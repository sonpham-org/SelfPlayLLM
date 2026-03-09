"""REPL-enabled game agent with tool-integrated reasoning.

Each agent gets a multi-step REPL loop per turn with:
- python: execute analysis code with game state in scope
- simulate: try a move on a copy and see the result
- analyze: spawn subagents to evaluate specific moves
- move: submit final move (terminates loop)

Evolvable parameters:
- system_prompt, strategy_prompt (text — evolved by LLM)
- max_steps: REPL budget per turn (int, 1-15)
- subagent_budget: steps per subagent (int, 0-5)
- analysis_width: max moves to subagent-analyze (int, 0-5)
"""

import contextlib
import io
import json
import random
import re
import requests
import time
from dataclasses import dataclass, field
from typing import Any

from agent import (
    OLLAMA_URL, MODEL, SAFE_BUILTINS, render_strategy,
    DEFAULT_SYSTEM_PROMPT,
)

# ═══════════════════════════════════════════════════════════════
# REPL System Prompt
# ═══════════════════════════════════════════════════════════════

REPL_DEFAULT_SYSTEM = """You are a game-playing AI with access to analysis tools.

Each turn you can take ACTIONS to analyze the position before deciding.
Output ONLY valid JSON for each action — no other text.

AVAILABLE ACTIONS:

1. Run Python code to analyze the position:
   {"action": "python", "code": "your_code_here"}
   Variables in scope: state (dict), board (2D list), legal_moves (list of str),
   num_moves (int), move_number (int), my_color (str), history (list), memory (dict).
   Use print() to see results.

2. Simulate a move to preview the resulting position:
   {"action": "simulate", "move": "3"}
   Shows the board after your move, opponent's options, and threats.

3. Spawn subagent analysis for specific moves (if budget allows):
   {"action": "analyze", "moves": ["0", "3", "6"]}
   Returns an evaluation of each listed move.

4. Submit your final move (REQUIRED — must end with this):
   {"action": "move", "move": "3"}
   Optional: {"action": "move", "move": "3", "memory_update": {"key": "value"}}

RULES:
- You have a LIMITED number of steps. Use them wisely.
- You MUST end with a "move" action. If you run out of steps, a fallback move is chosen.
- For obvious positions (forced moves, clear wins), decide quickly.
- For complex positions, use python/simulate to analyze before deciding.
- Output ONLY valid JSON each step. No markdown, no explanation outside JSON.
"""


# ═══════════════════════════════════════════════════════════════
# REPLAgent dataclass
# ═══════════════════════════════════════════════════════════════

@dataclass
class REPLAgent:
    id: int
    system_prompt: str
    strategy_prompt: str
    max_steps: int = 5
    subagent_budget: int = 2
    analysis_width: int = 3
    game_memory: dict = field(default_factory=dict)
    tournament_memory: dict = field(default_factory=dict)
    wins: int = 0
    losses: int = 0
    draws: int = 0
    elo: int = 1200
    game_notes: list[str] = field(default_factory=list)
    total_llm_calls: int = 0

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
        self.total_llm_calls = 0

    def reset_game_memory(self):
        self.game_memory = {}


# ═══════════════════════════════════════════════════════════════
# REPL Environment
# ═══════════════════════════════════════════════════════════════

class REPLEnvironment:
    """Sandboxed execution environment for agent code."""

    def __init__(self, game, player, history, memory):
        self.game = game
        self.player = player
        self.state = game.get_state(player)
        self.legal_moves = game.get_legal_moves(player)
        self.legal_strs = [game.move_to_str(m) for m in self.legal_moves]
        self.history = history
        self.memory = memory

        self._initial_keys = set()
        self.namespace = {
            '__builtins__': SAFE_BUILTINS,
            'state': self.state,
            'board': self.state.get('board', []),
            'legal_moves': self.legal_strs,
            'num_moves': len(self.legal_strs),
            'move_number': self.state.get('move_number', game.move_count),
            'my_color': player,
            'history': history,
            'memory': memory,
        }
        self._initial_keys = set(self.namespace.keys())

    def execute(self, code: str) -> str:
        """Execute Python code in the sandbox. Returns output + new variables."""
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                exec(code, self.namespace)
            output = stdout.getvalue().strip()

            # Collect newly defined variables
            new_vars = {}
            for k, v in self.namespace.items():
                if k.startswith('_') or k in self._initial_keys:
                    continue
                try:
                    s = str(v)
                    new_vars[k] = s[:200] if len(s) > 200 else s
                except Exception:
                    new_vars[k] = "<unprintable>"

            parts = []
            if output:
                parts.append(output[:500])
            if new_vars:
                parts.append("Variables: " + json.dumps(new_vars, default=str)[:500])
            return "\n".join(parts) if parts else "(code ran, no output)"

        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    def simulate_move(self, move_str: str) -> str:
        """Simulate a move and return the resulting position."""
        move = self.game.parse_move(str(move_str), self.player)
        if move is None:
            return f"Invalid move: '{move_str}'. Legal moves: {self.legal_strs}"

        game_copy = self.game.copy()
        game_copy.apply_move(move)
        winner = game_copy.winner()

        if winner:
            return f"After {move_str}: GAME OVER — {winner}!"

        opp = game_copy.current_player
        opp_legal = game_copy.get_legal_moves(opp)
        opp_state = game_copy.get_state(opp)

        board_str = game_copy.render(perspective=self.player)
        lines = [
            f"After move {move_str}:",
            board_str,
            f"Winner: ongoing",
            f"Opponent ({opp}) has {len(opp_legal)} legal moves.",
        ]
        # Add threat info if available
        for key in ['my_threats_3', 'opponent_threats_3']:
            if key in opp_state:
                label = key.replace('_', ' ')
                lines.append(f"  {label}: {opp_state[key]}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# LLM multi-turn chat
# ═══════════════════════════════════════════════════════════════

def query_llm_chat(messages: list[dict], temperature: float = 0.3,
                   max_tokens: int = 500) -> str:
    """Multi-turn Ollama chat. Returns assistant content."""
    resp = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }, timeout=120)
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "")


# ═══════════════════════════════════════════════════════════════
# Action parsing
# ═══════════════════════════════════════════════════════════════

def _extract_json(text: str) -> dict | None:
    """Extract the first valid JSON object from text."""
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
    return None


def parse_action(response: str) -> dict:
    """Parse an action from LLM response. Returns dict with 'type' key."""
    cleaned = re.sub(r'<[^>]+>', '', response).strip()
    obj = _extract_json(cleaned)
    if not obj:
        return {"type": "invalid", "raw": response[:200]}

    action = obj.get("action", "").lower().strip()
    if action == "move":
        return {"type": "move", "move": str(obj.get("move", "")),
                "memory_update": obj.get("memory_update", {})}
    elif action == "python":
        return {"type": "python", "code": obj.get("code", "")}
    elif action == "simulate":
        return {"type": "simulate", "move": str(obj.get("move", ""))}
    elif action == "analyze":
        moves = obj.get("moves", [])
        return {"type": "analyze", "moves": [str(m) for m in moves]}
    else:
        # Try to interpret as a direct move (fallback)
        if "move" in obj:
            return {"type": "move", "move": str(obj["move"]),
                    "memory_update": obj.get("memory_update", {})}
        return {"type": "invalid", "raw": response[:200]}


# ═══════════════════════════════════════════════════════════════
# Subagent analysis
# ═══════════════════════════════════════════════════════════════

def run_subagent_analysis(game, player, history, memory,
                          moves_to_analyze: list[str],
                          budget: int) -> str:
    """Evaluate specific moves using focused subagent queries."""
    results = []
    for move_str in moves_to_analyze:
        move = game.parse_move(str(move_str), player)
        if move is None:
            results.append(f"  {move_str}: INVALID MOVE")
            continue

        game_copy = game.copy()
        game_copy.apply_move(move)
        winner = game_copy.winner()

        if winner:
            results.append(f"  {move_str}: GAME OVER — {winner}")
            continue

        opp = game_copy.current_player
        opp_legal = game_copy.get_legal_moves(opp)
        opp_state = game_copy.get_state(opp)
        board_str = game_copy.render(perspective=player)

        prompt = f"""Evaluate this position for {player} after playing move {move_str}:

{board_str}

Opponent ({opp}) has {len(opp_legal)} legal moves.
State: {json.dumps({k: v for k, v in opp_state.items() if k != 'board'}, default=str)[:300]}

Rate: WINNING / GOOD / NEUTRAL / BAD / LOSING. One sentence why."""

        messages = [
            {"role": "system", "content": "You are a game position evaluator. Be concise. Output format: RATING: explanation"},
            {"role": "user", "content": prompt},
        ]

        try:
            evaluation = query_llm_chat(messages, temperature=0.3, max_tokens=100)
            results.append(f"  {move_str}: {evaluation.strip()[:150]}")
        except Exception as e:
            results.append(f"  {move_str}: [eval failed: {e}]")

    return "\n".join(results)


# ═══════════════════════════════════════════════════════════════
# REPL move picking (core loop)
# ═══════════════════════════════════════════════════════════════

def repl_pick_move(game, player: str, system_prompt: str, strategy_prompt: str,
                   game_memory: dict, history: list[str],
                   max_steps: int = 5, subagent_budget: int = 2,
                   analysis_width: int = 3) -> tuple[Any, dict, dict]:
    """Multi-step REPL reasoning to pick a move.

    Returns (move, memory_update, stats_dict).
    """
    legal = game.get_legal_moves(player)
    stats = {"llm_calls": 0, "steps_used": 0, "actions": []}

    if not legal:
        return None, {}, stats
    if len(legal) == 1:
        return legal[0], {}, stats

    env = REPLEnvironment(game, player, history, game_memory)

    # Render strategy template
    rendered_strategy = render_strategy(strategy_prompt, env.namespace)

    # Build initial prompt
    board = game.render(perspective=player)
    feat = {k: v for k, v in env.state.items() if k != 'board'}
    hist_text = "\n".join(history[-10:]) if history else "(none)"

    initial_prompt = f"""=== YOUR STRATEGY ===
{rendered_strategy}

=== BOARD ===
You are: {player.upper()}
{board}

=== STATE FEATURES ===
{json.dumps(feat, indent=2, default=str)}

=== LEGAL MOVES ===
{chr(10).join(f'{i+1}. {s}' for i, s in enumerate(env.legal_strs))}

=== RECENT HISTORY ===
{hist_text}

=== YOUR MEMORY ===
{json.dumps(game_memory, indent=2, default=str) if game_memory else '(empty)'}

You have {max_steps} analysis steps. Choose your action."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_prompt},
    ]

    for step in range(max_steps):
        try:
            response = query_llm_chat(messages, temperature=0.3, max_tokens=500)
        except Exception as e:
            stats["actions"].append(f"llm_error:{e}")
            break

        stats["llm_calls"] += 1
        action = parse_action(response)
        stats["actions"].append(action["type"])

        remaining = max_steps - step - 1

        if action["type"] == "move":
            move = game.parse_move(action["move"], player)
            if move is not None:
                stats["steps_used"] = step + 1
                mem_update = action.get("memory_update", {})
                return move, (mem_update if isinstance(mem_update, dict) else {}), stats
            # Invalid move — give feedback
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content":
                f"Invalid move '{action['move']}'. Legal moves: {env.legal_strs}. "
                f"Try again. {remaining} steps remaining."})

        elif action["type"] == "python":
            result = env.execute(action["code"])
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content":
                f"```\n{result}\n```\n{remaining} steps remaining."})

        elif action["type"] == "simulate":
            result = env.simulate_move(action["move"])
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content":
                f"{result}\n\n{remaining} steps remaining."})

        elif action["type"] == "analyze":
            if subagent_budget > 0 and analysis_width > 0:
                moves = action["moves"][:analysis_width]
                analysis = run_subagent_analysis(
                    game, player, history, game_memory, moves, subagent_budget)
                stats["llm_calls"] += len(moves)
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content":
                    f"Move evaluations:\n{analysis}\n\n{remaining} steps remaining."})
            else:
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content":
                    "Subagent analysis not available (budget=0). "
                    f"Use python/simulate instead. {remaining} steps remaining."})

        else:
            # Invalid action
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content":
                f"Invalid action. Use: python, simulate, analyze, or move. "
                f"{remaining} steps remaining."})

    # Ran out of steps — fallback
    stats["steps_used"] = max_steps
    return legal[0], {}, stats


# ═══════════════════════════════════════════════════════════════
# Game loop (drop-in for agent.play_game)
# ═══════════════════════════════════════════════════════════════

def play_game_repl(game, agent_a, agent_b, max_moves: int = 150,
                   verbose: bool = False) -> dict:
    """Play a full game between two REPL agents.

    Compatible with agent.play_game() signature.
    """
    p1, p2 = game.players
    history = []

    def unpack(ag):
        if isinstance(ag, REPLAgent):
            return (ag.system_prompt, ag.strategy_prompt, dict(ag.game_memory),
                    ag.max_steps, ag.subagent_budget, ag.analysis_width)
        # Fallback for plain Agent or string
        if isinstance(ag, str):
            return (REPL_DEFAULT_SYSTEM, ag, {}, 3, 0, 0)
        return (getattr(ag, 'system_prompt', REPL_DEFAULT_SYSTEM),
                getattr(ag, 'strategy_prompt', ''), dict(ag.game_memory),
                getattr(ag, 'max_steps', 3),
                getattr(ag, 'subagent_budget', 0),
                getattr(ag, 'analysis_width', 0))

    sys1, strat1, mem1, steps1, sub1, width1 = unpack(agent_a)
    sys2, strat2, mem2, steps2, sub2, width2 = unpack(agent_b)

    parts = {
        p1: (sys1, strat1, mem1, steps1, sub1, width1),
        p2: (sys2, strat2, mem2, steps2, sub2, width2),
    }

    total_llm_calls = 0
    total_repl_steps = 0

    while game.winner() is None and game.move_count < max_moves:
        player = game.current_player
        sys_p, strat_p, mem_p, max_s, sub_b, a_w = parts[player]

        move, mem_update, stats = repl_pick_move(
            game, player, sys_p, strat_p, mem_p, history,
            max_steps=max_s, subagent_budget=sub_b, analysis_width=a_w,
        )
        total_llm_calls += stats["llm_calls"]
        total_repl_steps += stats["steps_used"]

        if move is None:
            break

        if mem_update:
            mem_p.update(mem_update)

        move_str = game.move_to_str(move)
        history.append(f"{player}: {move_str}")
        game.apply_move(move)

        if verbose:
            actions_str = ",".join(stats["actions"]) if stats["actions"] else "forced"
            print(f"  Move {game.move_count}: {player} plays {move_str} "
                  f"[{stats['steps_used']}steps, {stats['llm_calls']}calls: {actions_str}]")

    result = game.winner() or "draw"
    return {
        "winner": result,
        "moves": game.move_count,
        "history": history,
        "final_memories": {p1: mem1, p2: mem2},
        "total_llm_calls": total_llm_calls,
        "total_repl_steps": total_repl_steps,
    }


# ═══════════════════════════════════════════════════════════════
# Seed REPL agents
# ═══════════════════════════════════════════════════════════════

REPL_SEED_AGENTS = [
    {
        "system": REPL_DEFAULT_SYSTEM,
        "strategy": """Quick analysis: compute material balance, then decide.
Step 1: python — calculate advantage from state.
Step 2: move — pick the best move based on advantage.""",
        "max_steps": 2, "subagent_budget": 0, "analysis_width": 0,
    },
    {
        "system": REPL_DEFAULT_SYSTEM,
        "strategy": """Thorough simulation: test each promising move.
Step 1: simulate the center/key moves.
Step 2: simulate 1-2 more candidates.
Step 3: compare results and pick the best.
Prioritize moves that create threats or block opponent threats.""",
        "max_steps": 6, "subagent_budget": 0, "analysis_width": 0,
    },
    {
        "system": REPL_DEFAULT_SYSTEM,
        "strategy": """Threat-first analysis:
Step 1: python — check if opponent has any winning threats to block.
Step 2: python — check if we have any winning moves.
Step 3: If no immediate tactics, simulate top 2 moves and pick best.""",
        "max_steps": 4, "subagent_budget": 0, "analysis_width": 0,
    },
    {
        "system": REPL_DEFAULT_SYSTEM,
        "strategy": """Delegate analysis to subagents.
Step 1: identify the 3 most promising legal moves.
Step 2: analyze those 3 moves (spawns subagents).
Step 3: review evaluations and pick the highest-rated move.""",
        "max_steps": 4, "subagent_budget": 3, "analysis_width": 3,
    },
    {
        "system": REPL_DEFAULT_SYSTEM,
        "strategy": """Adaptive memory-driven analysis:
Step 1: python — check memory for opponent patterns and game phase.
Step 2: python — compute position features based on phase.
Step 3: simulate 1-2 candidate moves.
Step 4: update memory with observations, then pick move.""",
        "max_steps": 5, "subagent_budget": 2, "analysis_width": 2,
    },
    {
        "system": REPL_DEFAULT_SYSTEM,
        "strategy": """Deep thinker: maximize analysis depth.
Use python to compute threats, material, and mobility.
Simulate multiple moves to find the best continuation.
Check opponent responses after each simulation.
Only decide when confident.""",
        "max_steps": 8, "subagent_budget": 0, "analysis_width": 0,
    },
    {
        "system": REPL_DEFAULT_SYSTEM,
        "strategy": """Minimalist: one-shot decision.
Quickly assess the position and pick the most natural move.
Don't overthink — speed is a feature.""",
        "max_steps": 1, "subagent_budget": 0, "analysis_width": 0,
    },
    {
        "system": REPL_DEFAULT_SYSTEM,
        "strategy": """Pattern tracker: focus on opponent behavior.
Step 1: python — analyze opponent's recent moves from history.
Step 2: python — predict opponent's likely next move.
Step 3: simulate the counter-move.
Store patterns in memory for future turns.""",
        "max_steps": 4, "subagent_budget": 0, "analysis_width": 0,
    },
    {
        "system": REPL_DEFAULT_SYSTEM,
        "strategy": """Positional evaluator: focus on board control.
Step 1: python — count pieces in center, edges, and corners.
Step 2: python — compute mobility ratio (my moves vs opponent).
Step 3: simulate moves that improve positional score.
Step 4: pick the move with best positional outcome.""",
        "max_steps": 5, "subagent_budget": 0, "analysis_width": 0,
    },
    {
        "system": REPL_DEFAULT_SYSTEM,
        "strategy": """Full hybrid: use every tool available.
Step 1: python — quick tactical scan (threats, forced moves).
Step 2: If unclear, analyze top 3 moves with subagents.
Step 3: simulate the top-rated move to verify.
Step 4: decide based on combined analysis.
Adapt budget usage to position complexity.""",
        "max_steps": 7, "subagent_budget": 3, "analysis_width": 4,
    },
]


# ═══════════════════════════════════════════════════════════════
# Mutation / Crossover for REPL agents
# ═══════════════════════════════════════════════════════════════

MUTATE_REPL_SYSTEM = """You are an AI strategy architect evolving REPL-enabled game agents.

Each agent has:
1. SYSTEM PROMPT — identity and REPL tool-use policy
2. STRATEGY PROMPT — analysis plan template (guides how the agent uses its REPL tools)
3. MAX_STEPS — REPL budget per turn (1-15)
4. SUBAGENT_BUDGET — evaluation depth per subagent (0-5)
5. ANALYSIS_WIDTH — max moves to subagent-evaluate (0-5)

REPL actions available to agents each turn:
- {"action": "python", "code": "..."} — run analysis code
- {"action": "simulate", "move": "..."} — try a move, see resulting board
- {"action": "analyze", "moves": [...]} — spawn subagent evaluations
- {"action": "move", "move": "..."} — submit final move

Strategy prompts should describe a STEP-BY-STEP analysis plan.
More steps = deeper analysis but slower. Find the right balance.

Output ONLY valid JSON:
{"system_prompt": "...", "strategy_prompt": "...", "max_steps": N, "subagent_budget": N, "analysis_width": N}"""

CROSSOVER_REPL_SYSTEM = """You are an AI strategy architect. Combine two REPL-enabled game agents into one stronger agent.

Each agent has system_prompt, strategy_prompt, max_steps, subagent_budget, analysis_width.
Agents can use python, simulate, analyze, and move actions each turn.

Create a NEW agent that takes the best ideas from both.
Balance analysis depth (max_steps) with speed.

Output ONLY valid JSON:
{"system_prompt": "...", "strategy_prompt": "...", "max_steps": N, "subagent_budget": N, "analysis_width": N}"""


def mutate_repl_agent(agent: REPLAgent) -> REPLAgent:
    """Mutate a REPL agent. Returns a new REPLAgent with evolved parameters."""
    from agent import query_llm

    notes = "\n".join(agent.game_notes[-6:]) if agent.game_notes else "No notes."
    prompt = f"""Current agent (ID {agent.id}):

SYSTEM: {agent.system_prompt[:200]}
STRATEGY: {agent.strategy_prompt}
MAX_STEPS: {agent.max_steps}
SUBAGENT_BUDGET: {agent.subagent_budget}
ANALYSIS_WIDTH: {agent.analysis_width}

Results: W={agent.wins} L={agent.losses} D={agent.draws} ({agent.winrate:.0%})
LLM calls used: {agent.total_llm_calls}

Game notes: {notes}

Write a NEW, IMPROVED version. Keep what works, fix what doesn't.
Optimize the step budget — don't waste steps on obvious positions.

Output JSON: {{"system_prompt": "...", "strategy_prompt": "...", "max_steps": N, "subagent_budget": N, "analysis_width": N}}"""

    try:
        response = query_llm(MUTATE_REPL_SYSTEM, prompt, temperature=0.7,
                             think=False, max_tokens=1500)
        obj = _extract_json(re.sub(r'<[^>]+>', '', response).strip())
        if obj and obj.get("strategy_prompt"):
            return REPLAgent(
                id=agent.id,
                system_prompt=obj.get("system_prompt", agent.system_prompt),
                strategy_prompt=obj["strategy_prompt"],
                max_steps=_clamp(obj.get("max_steps", agent.max_steps), 1, 15),
                subagent_budget=_clamp(obj.get("subagent_budget", agent.subagent_budget), 0, 5),
                analysis_width=_clamp(obj.get("analysis_width", agent.analysis_width), 0, 5),
            )
    except Exception as e:
        print(f"  REPL mutation failed for Agent {agent.id}: {e}")

    # Fallback: perturb numeric params only
    return REPLAgent(
        id=agent.id,
        system_prompt=agent.system_prompt,
        strategy_prompt=agent.strategy_prompt,
        max_steps=_clamp(agent.max_steps + random.choice([-1, 0, 1]), 1, 15),
        subagent_budget=_clamp(agent.subagent_budget + random.choice([-1, 0, 1]), 0, 5),
        analysis_width=_clamp(agent.analysis_width + random.choice([-1, 0, 1]), 0, 5),
    )


def crossover_repl_agents(a: REPLAgent, b: REPLAgent) -> REPLAgent:
    """Combine two REPL agents."""
    from agent import query_llm

    prompt = f"""Agent A ({a.winrate:.0%} win rate, {a.total_llm_calls} LLM calls):
SYSTEM: {a.system_prompt[:150]}
STRATEGY: {a.strategy_prompt}
MAX_STEPS={a.max_steps}, SUBAGENT_BUDGET={a.subagent_budget}, ANALYSIS_WIDTH={a.analysis_width}

Agent B ({b.winrate:.0%} win rate, {b.total_llm_calls} LLM calls):
SYSTEM: {b.system_prompt[:150]}
STRATEGY: {b.strategy_prompt}
MAX_STEPS={b.max_steps}, SUBAGENT_BUDGET={b.subagent_budget}, ANALYSIS_WIDTH={b.analysis_width}

Combine into one NEW, stronger agent.
Output JSON: {{"system_prompt": "...", "strategy_prompt": "...", "max_steps": N, "subagent_budget": N, "analysis_width": N}}"""

    try:
        response = query_llm(CROSSOVER_REPL_SYSTEM, prompt, temperature=0.7,
                             think=False, max_tokens=1500)
        obj = _extract_json(re.sub(r'<[^>]+>', '', response).strip())
        if obj and obj.get("strategy_prompt"):
            return REPLAgent(
                id=a.id,
                system_prompt=obj.get("system_prompt", a.system_prompt),
                strategy_prompt=obj["strategy_prompt"],
                max_steps=_clamp(obj.get("max_steps", (a.max_steps + b.max_steps) // 2), 1, 15),
                subagent_budget=_clamp(obj.get("subagent_budget", max(a.subagent_budget, b.subagent_budget)), 0, 5),
                analysis_width=_clamp(obj.get("analysis_width", max(a.analysis_width, b.analysis_width)), 0, 5),
            )
    except Exception as e:
        print(f"  REPL crossover failed: {e}")

    # Fallback: take better agent's prompts, average numeric params
    better = a if a.winrate >= b.winrate else b
    return REPLAgent(
        id=a.id,
        system_prompt=better.system_prompt,
        strategy_prompt=better.strategy_prompt,
        max_steps=(a.max_steps + b.max_steps) // 2,
        subagent_budget=max(a.subagent_budget, b.subagent_budget),
        analysis_width=max(a.analysis_width, b.analysis_width),
    )


def _clamp(val, lo, hi):
    return max(lo, min(hi, int(val)))


# ═══════════════════════════════════════════════════════════════
# Population / persistence helpers
# ═══════════════════════════════════════════════════════════════

def create_repl_population(n: int = 10) -> list[REPLAgent]:
    agents = []
    for i in range(n):
        seed = REPL_SEED_AGENTS[i % len(REPL_SEED_AGENTS)]
        agents.append(REPLAgent(
            id=i,
            system_prompt=seed["system"],
            strategy_prompt=seed["strategy"],
            max_steps=seed["max_steps"],
            subagent_budget=seed["subagent_budget"],
            analysis_width=seed["analysis_width"],
        ))
    return agents


def evolve_repl(agents: list[REPLAgent], survive_ratio: float = 0.5,
                mutation_ratio: float = 0.6, verbose: bool = True) -> list[REPLAgent]:
    """Evolve REPL agents. Top half survives, bottom half replaced."""
    n = len(agents)
    n_survive = max(2, int(n * survive_ratio))
    n_replace = n - n_survive
    n_mutate = int(n_replace * mutation_ratio)

    survivors = agents[:n_survive]
    new_agents = []

    if verbose:
        print(f"\n--- REPL EVOLUTION ({n_survive} survive, {n_mutate} mutate, "
              f"{n_replace - n_mutate} crossover) ---")

    for idx in range(n_replace):
        old_id = agents[n_survive + idx].id

        if idx < n_mutate:
            parent = survivors[idx % n_survive]
            if verbose:
                print(f"  Mutating Agent {parent.id} -> new Agent {old_id}")
            new_ag = mutate_repl_agent(parent)
            new_ag.id = old_id
        else:
            p1, p2 = random.sample(survivors[:min(n_survive, 10)], 2)
            if verbose:
                print(f"  Crossover Agent {p1.id} x Agent {p2.id} -> new Agent {old_id}")
            new_ag = crossover_repl_agents(p1, p2)
            new_ag.id = old_id

        new_agents.append(new_ag)

    for a in survivors:
        a.reset_stats()

    return survivors + new_agents


def save_repl_state(agents: list[REPLAgent], generation: int, log: list,
                    run_dir: str = "."):
    """Save REPL agent state to JSON."""
    import os
    path = os.path.join(run_dir, "evo_state.json")
    data = {
        "generation": generation,
        "agent_type": "repl",
        "agents": [
            {
                "id": a.id,
                "system_prompt": a.system_prompt,
                "strategy_prompt": a.strategy_prompt,
                "max_steps": a.max_steps,
                "subagent_budget": a.subagent_budget,
                "analysis_width": a.analysis_width,
                "wins": a.wins, "losses": a.losses, "draws": a.draws,
                "elo": a.elo,
                "total_llm_calls": a.total_llm_calls,
            }
            for a in agents
        ],
        "log": log,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_repl_state(run_dir: str):
    """Load REPL agent state from JSON."""
    import os
    path = os.path.join(run_dir, "evo_state.json")
    with open(path) as f:
        data = json.load(f)
    agents = []
    for a in data["agents"]:
        agents.append(REPLAgent(
            id=a["id"],
            system_prompt=a.get("system_prompt", REPL_DEFAULT_SYSTEM),
            strategy_prompt=a.get("strategy_prompt", ""),
            max_steps=a.get("max_steps", 5),
            subagent_budget=a.get("subagent_budget", 2),
            analysis_width=a.get("analysis_width", 3),
            elo=a.get("elo", 1200),
        ))
    return agents, data["generation"], data.get("log", [])
