"""Game-agnostic LLM agent with rich strategy support.

Each agent has:
- system_prompt: static identity/approach (becomes the LLM system message)
- strategy_prompt: dynamic template rendered every turn with live game variables
- game_memory: dict persisting across turns within a single game

Strategy templates can contain:
- Plain text instructions and heuristics
- ```python ... ``` code blocks that compute values into the template namespace
- {expression} references that get evaluated with live game state
"""

import json
import re
import requests
from typing import Any

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:4b"

DEFAULT_SYSTEM_PROMPT = """You are a game-playing AI. Each turn you receive:
1. Your STRATEGY — dynamic instructions with live game analysis
2. The current board state and computed features
3. Legal moves to choose from
4. Recent move history
5. Your persistent memory (you can read and update it)

CAPABILITIES:
- Integrated reasoning: think step-by-step before deciding
- Game state features: a dict of computed features about the current position
- Memory: persistent storage across turns — track patterns, opponent tendencies, plans

OUTPUT FORMAT — respond with valid JSON only:
{"move": "<chosen move from legal moves>"}

You may also include optional fields:
{"move": "...", "memory_update": {"key": "value"}, "reasoning": "brief thought"}

Pick from the legal moves list. Output valid JSON only, no other text."""

# --- Safe builtins for strategy template code execution ---
SAFE_BUILTINS = {
    'abs': abs, 'len': len, 'max': max, 'min': min, 'sum': sum,
    'round': round, 'sorted': sorted, 'range': range, 'enumerate': enumerate,
    'zip': zip, 'int': int, 'float': float, 'str': str, 'bool': bool,
    'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
    'True': True, 'False': False, 'None': None, 'any': any, 'all': all,
    'isinstance': isinstance, 'map': map, 'filter': filter,
}


def render_strategy(template: str, namespace: dict) -> str:
    """Render a strategy prompt template with code execution and variable substitution.

    1. ```python ... ``` blocks are executed; variables they define enter the namespace
    2. {expression} references are evaluated and replaced with their string values
    3. If anything fails, it degrades gracefully (error message or raw text preserved)
    """
    if not template:
        return "(no strategy defined)"

    exec_ns = {'__builtins__': SAFE_BUILTINS}
    exec_ns.update(namespace)

    # Phase 1: execute ```python ... ``` code blocks
    code_pattern = re.compile(r'```python\s*\n(.*?)```', re.DOTALL)

    def exec_code_block(match):
        code = match.group(1)
        try:
            exec(code, exec_ns)
            return ""  # code ran, variables now in exec_ns
        except Exception as e:
            return f"[code error: {e}]"

    rendered = code_pattern.sub(exec_code_block, template)

    # Phase 2: substitute {expression} references
    def eval_expression(match):
        expr = match.group(1).strip()
        # Skip things that look like JSON or format specifiers
        if not expr or expr.startswith('"') or expr.startswith("'"):
            return match.group(0)
        try:
            result = eval(expr, {'__builtins__': SAFE_BUILTINS}, exec_ns)
            return str(result)
        except Exception:
            return match.group(0)  # leave unresolved

    rendered = re.sub(r'\{([^{}]+)\}', eval_expression, rendered)
    return rendered


def build_prompt(game, player: str, system_prompt: str, strategy_prompt: str,
                 game_memory: dict, history: list[str]) -> tuple[str, str]:
    """Build (system_message, user_prompt) for the LLM.

    Renders the strategy template with live game state variables.
    Returns the system prompt and rendered user prompt.
    """
    state = game.get_state(player)
    legal = game.get_legal_moves(player)
    legal_strs = [game.move_to_str(m) for m in legal]

    # Namespace available inside strategy templates
    namespace = {
        'state': state,
        'memory': game_memory,
        'history': history,
        'legal_moves': legal_strs,
        'num_moves': len(legal_strs),
        'move_number': state.get('move_number', game.move_count),
        'my_color': player,
    }

    # Render strategy template with live values
    rendered_strategy = render_strategy(strategy_prompt, namespace)

    board = game.render(perspective=player)
    hist_text = "\n".join(history[-10:]) if history else "(none)"

    # State features (exclude raw board to avoid duplication)
    feat = {k: v for k, v in state.items() if k != 'board'}

    user_prompt = f"""=== STRATEGY ===
{rendered_strategy}

=== BOARD ===
You are: {player.upper()}
{board}

=== STATE FEATURES ===
{json.dumps(feat, indent=2, default=str)}

=== LEGAL MOVES ===
{chr(10).join(f'{i+1}. {s}' for i, s in enumerate(legal_strs))}

=== RECENT HISTORY ===
{hist_text}

=== YOUR MEMORY ===
{json.dumps(game_memory, indent=2, default=str) if game_memory else '(empty — you can store observations here)'}

Pick the best move. Output JSON: {{"move": "<chosen move>"}}
Optional: {{"move": "...", "memory_update": {{"key": "value"}}, "reasoning": "..."}}"""

    return system_prompt, user_prompt


def query_llm(system: str, prompt: str, temperature: float = 0.3, think: bool = False,
              max_tokens: int | None = None) -> str:
    """Call Ollama chat API. Returns the response text."""
    if max_tokens is None:
        max_tokens = 120 if not think else 800
    resp = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "think": think,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data.get("message", {}).get("content", "")


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


def agent_pick_move(game, player: str, system_prompt: str, strategy_prompt: str,
                    game_memory: dict, history: list[str]) -> tuple[Any, dict]:
    """Ask the LLM to pick a move. Returns (move, memory_update_dict)."""
    legal = game.get_legal_moves(player)
    if not legal:
        return None, {}
    if len(legal) == 1:
        return legal[0], {}

    sys_msg, user_prompt = build_prompt(
        game, player, system_prompt, strategy_prompt, game_memory, history
    )

    for attempt in range(3):
        try:
            response = query_llm(sys_msg, user_prompt, temperature=0.3 + attempt * 0.2)
            obj = _extract_json(response)
            if obj:
                move_str = obj.get("move", "")
                move = game.parse_move(str(move_str), player)
                if move is not None:
                    mem_update = obj.get("memory_update", {})
                    return move, mem_update if isinstance(mem_update, dict) else {}
        except Exception:
            pass

    # Fallback: first legal move
    return legal[0], {}


def play_game(game, agent_a, agent_b, max_moves: int = 150, verbose: bool = False) -> dict:
    """Play a full game between two agents.

    agent_a plays as players[0], agent_b plays as players[1].
    Each agent should have: .system_prompt, .strategy_prompt, .game_memory
    Or be a plain string (backward compat: treated as strategy_prompt with defaults).
    """
    p1, p2 = game.players
    history = []

    def unpack(agent):
        """Extract (system_prompt, strategy_prompt, game_memory) from agent."""
        if isinstance(agent, str):
            return DEFAULT_SYSTEM_PROMPT, agent, {}
        return agent.system_prompt, agent.strategy_prompt, dict(agent.game_memory)

    sys1, strat1, mem1 = unpack(agent_a)
    sys2, strat2, mem2 = unpack(agent_b)

    parts = {
        p1: (sys1, strat1, mem1),
        p2: (sys2, strat2, mem2),
    }

    while game.winner() is None and game.move_count < max_moves:
        player = game.current_player
        sys_p, strat_p, mem_p = parts[player]

        move, mem_update = agent_pick_move(
            game, player, sys_p, strat_p, mem_p, history
        )
        if move is None:
            break

        # Apply memory update from the agent's response
        if mem_update:
            mem_p.update(mem_update)

        move_str = game.move_to_str(move)
        history.append(f"{player}: {move_str}")
        game.apply_move(move)

        if verbose:
            print(f"Move {game.move_count}: {player} plays {move_str}")

    result = game.winner() or "draw"
    if verbose:
        print(f"\nGame over: {result} ({game.move_count} moves)")
        print(game.render())

    return {
        "winner": result,
        "moves": game.move_count,
        "history": history,
        "final_memories": {p1: mem1, p2: mem2},
    }
