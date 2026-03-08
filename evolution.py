"""Evolutionary algorithm for game-playing agent strategies.

- 10 agents, each with a system_prompt + strategy_prompt (template with code/variables)
- Round-robin tournament on any game
- Top 5 survive, bottom 5 replaced by mutations/crossovers
- Mutation = LLM rewrites system_prompt + strategy_prompt
- ELO rating tracks true skill across generations
"""

import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable

from agent import play_game, query_llm, DEFAULT_SYSTEM_PROMPT

# --- ELO Rating ---
ELO_K = 32
ELO_DEFAULT = 1200


def elo_expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400))


def elo_update(ra: int, rb: int, sa: float, sb: float) -> tuple[int, int]:
    """Update ELO ratings. sa/sb: 1=win, 0.5=draw, 0=loss."""
    ea = elo_expected(ra, rb)
    eb = elo_expected(rb, ra)
    return round(ra + ELO_K * (sa - ea)), round(rb + ELO_K * (sb - eb))


# ═══════════════════════════════════════════════════════════════
# Agent dataclass
# ═══════════════════════════════════════════════════════════════

@dataclass
class Agent:
    id: int
    system_prompt: str
    strategy_prompt: str
    game_memory: dict = field(default_factory=dict)
    tournament_memory: dict = field(default_factory=dict)
    wins: int = 0
    losses: int = 0
    draws: int = 0
    elo: int = ELO_DEFAULT
    game_notes: list[str] = field(default_factory=list)

    @property
    def total(self): return self.wins + self.losses + self.draws

    @property
    def winrate(self): return self.wins / max(1, self.total)

    @property
    def score(self): return self.wins * 3 + self.draws * 1

    def reset_stats(self):
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.game_notes = []

    def reset_game_memory(self):
        self.game_memory = {}


@dataclass
class Champion:
    """A preserved top agent from a previous generation."""
    gen: int
    id: int
    system_prompt: str
    strategy_prompt: str
    elo: int = ELO_DEFAULT

    @property
    def label(self):
        return f"Champ(g{self.gen}#{self.id})"


# ═══════════════════════════════════════════════════════════════
# Seed strategies — demonstrate the template syntax
# ═══════════════════════════════════════════════════════════════

SEED_AGENTS = [
    {
        "system": "You are an aggressive player who prioritizes captures, material advantage, and constant pressure. Attack whenever possible.",
        "strategy": """Prioritize capturing and material advantage.
```python
advantage = state.get('my_pieces', 0) - state.get('opponent_pieces', 0)
phase = 'endgame' if move_number > 40 else 'opening' if move_number < 10 else 'midgame'
```
Material advantage: {advantage}. Game phase: {phase}.
If ahead, force trades. If behind, look for tactical shots.
Always pick the most aggressive legal move available.""",
    },
    {
        "system": "You are a patient, defensive player. Build solid positions and wait for the opponent to overextend before striking.",
        "strategy": """Play defensively. Build a solid position before attacking.
Keep pieces protected and avoid unnecessary risk.
Move number: {move_number}. Moves available: {num_moves}.
If the opponent is aggressive early, stay solid and counter-attack later.
Avoid trades unless clearly favorable.""",
    },
    {
        "system": "You are a strategic player focused on positional advantage and board control rather than raw material.",
        "strategy": """Focus on controlling key squares and having good piece placement.
```python
recent = history[-5:] if len(history) > 5 else history
activity = sum(1 for h in recent if '->' in h)
```
Recent activity level: {activity} moves in last 5 turns.
Position over material — control the board center, not just piece count.
Restrict the opponent's options whenever possible.""",
    },
    {
        "system": "You are an adaptive player who reads the opponent's style and adjusts. You use memory to track patterns across turns.",
        "strategy": """Adapt to the opponent's play style using memory.
Opponent style: {memory.get('opponent_style', 'unknown')}.
```python
if len(history) >= 6:
    opp_moves = [h for h in history if not h.startswith(my_color)]
    aggression = len(opp_moves)
else:
    aggression = 0
```
Opponent aggression indicator: {aggression}.
If opponent is passive, push forward. If aggressive, play solid and counter.
Update memory with your observations about the opponent each turn.""",
    },
    {
        "system": "You are a tempo-focused player. You aim to always have the initiative and force the opponent to respond to your threats.",
        "strategy": """Maintain tempo and initiative. Make moves that force responses.
Move {move_number}: look for moves that create multiple threats simultaneously.
Don't let the opponent dictate the pace.
{num_moves} options available — every move should either attack or improve position.
The best move puts the opponent on the defensive.""",
    },
    {
        "system": "You are an endgame specialist. You play conservatively early and excel at converting small advantages in the late game.",
        "strategy": """Play conservatively early. Excel in endgame conversions.
```python
my_p = state.get('my_pieces', 0)
opp_p = state.get('opponent_pieces', 0)
total_pieces = my_p + opp_p
is_endgame = total_pieces < 10 or move_number > 50
advantage = my_p - opp_p
```
Phase: {'ENDGAME — activate!' if is_endgame else 'Not endgame yet — be patient'}.
Advantage: {advantage}.
In endgame, every piece matters — maximize value, no wasteful trades.""",
    },
    {
        "system": "You are a tricky, tactical player who loves sacrifices and combinations. You look for surprising moves that catch opponents off guard.",
        "strategy": """Look for tactical combinations and surprising moves.
Sometimes sacrificing material leads to a bigger gain.
{num_moves} moves available — check ALL of them for hidden tactics.
The best move is often not the most obvious one.
Look for forcing sequences: captures, threats, and moves that limit opponent options.""",
    },
    {
        "system": "You are a methodical player who uses memory to build a detailed picture of the game. Track patterns and adapt.",
        "strategy": """Think methodically about each decision.
```python
turn_count = memory.get('turns_analyzed', 0) + 1
```
Turns analyzed: {turn_count}. Build on your observations.
Track what works and what fails in memory.
Update memory_update with: turns_analyzed, observed patterns, and opponent tendencies.
Use accumulated knowledge to make better decisions each turn.""",
    },
    {
        "system": "You are a balanced player who evaluates both tactical and positional factors. Adapt strategy based on the current game state.",
        "strategy": """Balance tactical and positional considerations.
```python
my_p = state.get('my_pieces', 0)
opp_p = state.get('opponent_pieces', 0)
if my_p > opp_p:
    mode = 'ahead — simplify and trade safely'
elif my_p < opp_p:
    mode = 'behind — complicate and avoid trades'
else:
    mode = 'equal — fight for small advantages'
```
Current mode: {mode}.
Legal moves: {num_moves}. More options = more flexibility.
Pick the move that best matches the current mode.""",
    },
    {
        "system": "You are a memory-driven player who builds detailed models of the opponent over the course of the game and exploits their weaknesses.",
        "strategy": """Build a mental model of the opponent using memory.
Opponent profile: {memory.get('profile', 'not yet built')}.
```python
profile = memory.get('profile', 'unknown')
games_tracked = memory.get('games_tracked', 0)
if isinstance(profile, dict):
    counter = profile.get('weakness', 'none identified')
else:
    counter = 'gathering data'
```
Counter-strategy: {counter}. Games tracked: {games_tracked}.
Every turn, update memory with new observations.
After enough data, exploit the opponent's patterns.""",
    },
]


# ═══════════════════════════════════════════════════════════════
# Mutation / Crossover prompts
# ═══════════════════════════════════════════════════════════════

MUTATE_SYSTEM = """You are an expert AI strategy architect who evolves game-playing agent strategies.

An agent has TWO components you must output:

1. SYSTEM PROMPT — The agent's identity and general approach (1-3 sentences).
   This becomes the LLM's system message and sets its persona.

2. STRATEGY PROMPT — A dynamic template executed EVERY TURN. This is powerful:

   a) Plain text: instructions, heuristics, and decision guidelines.

   b) Code blocks that compute live analysis:
      ```python
      advantage = state.get('my_pieces', 0) - state.get('opponent_pieces', 0)
      phase = 'endgame' if move_number > 50 else 'midgame'
      ```

   c) Variable references showing live values: {advantage}, {phase}

   d) Memory references: {memory.get('opponent_style', 'unknown')}

   Available variables in strategy templates:
   - state: dict with game features (board, pieces, game-specific computed keys)
   - memory: dict that persists across turns within a game
   - history: list of past move strings
   - legal_moves: list of available move strings
   - num_moves: count of legal moves
   - move_number: current turn number
   - my_color: which player the agent is

   The agent can READ memory via {memory.get('key', default)} in the template.
   The agent can WRITE memory by including "memory_update" in its JSON response.

   The strategy template can be AS LONG AS NEEDED. Use code blocks to compute
   position analysis. Use memory to track opponent patterns across turns.
   Use .get() with defaults in code blocks so they never crash.

RULES:
- Write a NEW and DIFFERENT strategy — never copy input word-for-word
- Use code blocks for computation — they make strategies much stronger
- Use memory tracking to adapt within a game
- Keep code blocks short and robust (use .get() with defaults)
- Strategy can reference any keys in the state dict

Output ONLY valid JSON (no markdown, no explanation):
{"system_prompt": "...", "strategy_prompt": "..."}"""

CROSSOVER_SYSTEM = """You are an AI strategy architect. Combine two game-playing agent strategies into one NEW, stronger agent.

Each agent has a system_prompt (persona) and strategy_prompt (dynamic template with code/variables).

Strategy templates can contain:
- Plain text heuristics
- ```python ... ``` code blocks for live computation
- {expression} variable references evaluated each turn
- {memory.get('key', default)} for persistent memory

Available template variables: state, memory, history, legal_moves, num_moves, move_number, my_color.

Create a NEW agent that synthesizes the best ideas from both inputs.
The result should be DIFFERENT from both — take what works and improve it.

Output ONLY valid JSON:
{"system_prompt": "...", "strategy_prompt": "..."}"""


def _build_mutate_prompt(agent: Agent) -> str:
    """Build mutation prompt using f-string (safe for nested braces in strategy)."""
    notes = "\n".join(agent.game_notes[-6:]) if agent.game_notes else "No detailed notes."
    return f"""Current agent (ID {agent.id}):

=== SYSTEM PROMPT ===
{agent.system_prompt}

=== STRATEGY PROMPT ===
{agent.strategy_prompt}

=== RECENT RESULTS ===
Wins: {agent.wins}, Losses: {agent.losses}, Draws: {agent.draws}
Win rate: {agent.winrate:.0%}

=== GAME NOTES ===
{notes}

Write a NEW, IMPROVED version. Keep what works, fix what doesn't.
Add better code analysis, smarter memory usage, or new tactical ideas.

Output JSON: {{"system_prompt": "...", "strategy_prompt": "..."}}"""


def _build_crossover_prompt(a: Agent, b: Agent) -> str:
    """Build crossover prompt using f-string."""
    return f"""Agent A (win rate {a.winrate:.0%}):

SYSTEM: {a.system_prompt}

STRATEGY:
{a.strategy_prompt}

---

Agent B (win rate {b.winrate:.0%}):

SYSTEM: {b.system_prompt}

STRATEGY:
{b.strategy_prompt}

---

Combine the best elements into one NEW, stronger agent.
Output JSON: {{"system_prompt": "...", "strategy_prompt": "..."}}"""


def _extract_json(text: str) -> dict | None:
    """Extract the first valid JSON object from text, handling nested braces."""
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    start = None
                    continue
    return None


def _parse_mutation_response(response: str, fallback_sys: str,
                             fallback_strat: str) -> tuple[str, str]:
    """Parse LLM mutation/crossover response into (system_prompt, strategy_prompt)."""
    # Strip thinking tags
    cleaned = re.sub(r'<[^>]+>', '', response).strip()

    obj = _extract_json(cleaned)
    if obj:
        sys_p = obj.get("system_prompt", "").strip()
        strat_p = obj.get("strategy_prompt", "").strip()
        if len(sys_p) > 10 and len(strat_p) > 20:
            return sys_p, strat_p

    return fallback_sys, fallback_strat


def mutate_agent(agent: Agent) -> tuple[str, str]:
    """Mutate an agent's strategy. Returns (new_system_prompt, new_strategy_prompt)."""
    prompt = _build_mutate_prompt(agent)
    try:
        response = query_llm(MUTATE_SYSTEM, prompt, temperature=0.7, think=True,
                             max_tokens=1500)
        return _parse_mutation_response(response, agent.system_prompt, agent.strategy_prompt)
    except Exception as e:
        print(f"  Mutation failed for Agent {agent.id}: {e}")
    return agent.system_prompt, agent.strategy_prompt


def crossover_agents(a: Agent, b: Agent) -> tuple[str, str]:
    """Combine two agents. Returns (new_system_prompt, new_strategy_prompt)."""
    prompt = _build_crossover_prompt(a, b)
    try:
        response = query_llm(CROSSOVER_SYSTEM, prompt, temperature=0.7, think=True,
                             max_tokens=1500)
        return _parse_mutation_response(response, a.system_prompt, a.strategy_prompt)
    except Exception as e:
        print(f"  Crossover failed: {e}")
    return a.system_prompt, a.strategy_prompt


# ═══════════════════════════════════════════════════════════════
# Tournament
# ═══════════════════════════════════════════════════════════════

def _record_result(agent_a, agent_b, winner: str, moves: int,
                   p1: str, p2: str, label_a: str, label_b: str):
    """Update W/L/D, game notes, and ELO for both agents."""
    if winner == p1:
        sa, sb = 1.0, 0.0
        agent_a.wins += 1
        agent_b.losses += 1
        agent_a.game_notes.append(f"Won as {p1} vs {label_b} in {moves} moves")
        agent_b.game_notes.append(f"Lost as {p2} vs {label_a} in {moves} moves")
    elif winner == p2:
        sa, sb = 0.0, 1.0
        agent_a.losses += 1
        agent_b.wins += 1
        agent_a.game_notes.append(f"Lost as {p1} vs {label_b} in {moves} moves")
        agent_b.game_notes.append(f"Won as {p2} vs {label_a} in {moves} moves")
    else:
        sa, sb = 0.5, 0.5
        agent_a.draws += 1
        agent_b.draws += 1
        agent_a.game_notes.append(f"Draw as {p1} vs {label_b} ({moves} moves)")
        agent_b.game_notes.append(f"Draw as {p2} vs {label_a} ({moves} moves)")

    agent_a.elo, agent_b.elo = elo_update(agent_a.elo, agent_b.elo, sa, sb)


def run_tournament(agents: list[Agent], game_factory: Callable,
                   verbose: bool = True) -> list[Agent]:
    """Round-robin tournament. game_factory() creates a new game instance each match."""
    for a in agents:
        a.reset_stats()

    pairs = list(combinations(range(len(agents)), 2))
    total_games = len(pairs) * 2
    game_num = 0

    for i, j in pairs:
        for a_idx, b_idx in [(i, j), (j, i)]:
            game_num += 1
            agent_a = agents[a_idx]
            agent_b = agents[b_idx]

            game = game_factory()
            p1, p2 = game.players

            if verbose:
                print(f"  Game {game_num}/{total_games}: "
                      f"A{agent_a.id}({p1}) vs A{agent_b.id}({p2})",
                      end=" ", flush=True)

            # Fresh game memory for each match
            agent_a.reset_game_memory()
            agent_b.reset_game_memory()

            t0 = time.time()
            result = play_game(game, agent_a, agent_b, max_moves=150)
            elapsed = time.time() - t0

            _record_result(agent_a, agent_b, result["winner"], result["moves"],
                           p1, p2, f"Agent {agent_a.id}", f"Agent {agent_b.id}")

            if verbose:
                w = result["winner"]
                sym = "A" if w == p1 else ("B" if w == p2 else "D")
                print(f"-> {sym} ({result['moves']}mv, {elapsed:.1f}s)")

    return sorted(agents, key=lambda a: (a.score, a.elo), reverse=True)


# ═══════════════════════════════════════════════════════════════
# Evolution
# ═══════════════════════════════════════════════════════════════

def evolve(agents: list[Agent], verbose: bool = True) -> list[Agent]:
    """Top 5 survive. Bottom 5 replaced by mutations/crossovers of top 5."""
    survivors = agents[:5]
    new_agents = []

    if verbose:
        print("\n--- EVOLUTION ---")
        print("Survivors:")
        for a in survivors:
            print(f"  Agent {a.id}: score={a.score} W={a.wins} L={a.losses} D={a.draws}")

    for idx in range(5):
        old_id = agents[5 + idx].id

        if idx < 3:
            parent = survivors[idx]
            if verbose:
                print(f"  Mutating Agent {parent.id} -> new Agent {old_id}")
            new_sys, new_strat = mutate_agent(parent)
        else:
            p1, p2 = random.sample(survivors, 2)
            if verbose:
                print(f"  Crossover Agent {p1.id} x Agent {p2.id} -> new Agent {old_id}")
            new_sys, new_strat = crossover_agents(p1, p2)

        new_agents.append(Agent(id=old_id, system_prompt=new_sys,
                                strategy_prompt=new_strat))

    for a in survivors:
        a.reset_stats()

    return survivors + new_agents


# ═══════════════════════════════════════════════════════════════
# Population setup / persistence
# ═══════════════════════════════════════════════════════════════

def create_initial_population() -> list[Agent]:
    return [
        Agent(id=i, system_prompt=s["system"], strategy_prompt=s["strategy"])
        for i, s in enumerate(SEED_AGENTS)
    ]


def get_run_dir(resume_path: str | None = None) -> str:
    """Return the run directory. Creates a new timestamped one if not resuming."""
    if resume_path:
        run_dir = resume_path.rstrip("/")
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return run_dir
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_state(agents: list[Agent], generation: int, log: list[dict],
               hall_of_fame: list[Champion] | None = None, run_dir: str = "."):
    path = os.path.join(run_dir, "evo_state.json")
    data = {
        "generation": generation,
        "agents": [
            {
                "id": a.id,
                "system_prompt": a.system_prompt,
                "strategy_prompt": a.strategy_prompt,
                "wins": a.wins, "losses": a.losses, "draws": a.draws,
                "elo": a.elo,
            }
            for a in agents
        ],
        "hall_of_fame": [
            {
                "gen": c.gen, "id": c.id,
                "system_prompt": c.system_prompt,
                "strategy_prompt": c.strategy_prompt,
                "elo": c.elo,
            }
            for c in (hall_of_fame or [])
        ],
        "log": log,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_state(run_dir: str):
    path = os.path.join(run_dir, "evo_state.json")
    with open(path) as f:
        data = json.load(f)
    agents = []
    for a in data["agents"]:
        if "system_prompt" in a:
            agents.append(Agent(
                id=a["id"],
                system_prompt=a["system_prompt"],
                strategy_prompt=a["strategy_prompt"],
                elo=a.get("elo", ELO_DEFAULT),
            ))
        else:
            # Backward compat: old format had "memory" as a flat string
            agents.append(Agent(
                id=a["id"],
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                strategy_prompt=a.get("memory", ""),
                elo=a.get("elo", ELO_DEFAULT),
            ))
    hall_of_fame = []
    for c in data.get("hall_of_fame", []):
        if "system_prompt" in c:
            hall_of_fame.append(Champion(
                gen=c["gen"], id=c["id"],
                system_prompt=c["system_prompt"],
                strategy_prompt=c["strategy_prompt"],
                elo=c.get("elo", ELO_DEFAULT),
            ))
        else:
            hall_of_fame.append(Champion(
                gen=c["gen"], id=c["id"],
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                strategy_prompt=c.get("memory", ""),
                elo=c.get("elo", ELO_DEFAULT),
            ))
    return agents, data["generation"], data.get("log", []), hall_of_fame
