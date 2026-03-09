#!/usr/bin/env python3
"""Small smoke test: 4 agents, 1 generation, Connect Four.

Verifies the full pipeline:
1. Agent creation
2. Strategy template rendering
3. LLM move picking (via Ollama)
4. Game play
5. Tournament
6. Mutation + Crossover
7. State saving
"""

import sys
import os
import time

# Ensure we're in the right directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from agent import query_llm, play_game, DEFAULT_SYSTEM_PROMPT
from evolution import (
    Agent, create_initial_population, run_tournament, evolve,
    mutate_agent, crossover_agents, save_state, get_run_dir,
)
from games.connect_four import ConnectFourGame


def test_llm_alive():
    """Step 0: verify ollama is responding."""
    print("=" * 60)
    print("TEST 0: LLM connectivity")
    print("=" * 60)
    try:
        resp = query_llm(
            "You are a helpful assistant.",
            'Respond with ONLY: {"status": "ok"}',
            temperature=0.1, max_tokens=20
        )
        print(f"  LLM response: {resp.strip()[:80]}")
        assert "ok" in resp.lower() or "{" in resp, "LLM didn't respond properly"
        print("  PASSED\n")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        print("  Make sure ollama is running: ollama serve")
        return False


def test_single_game():
    """Step 1: play one game between two agents."""
    print("=" * 60)
    print("TEST 1: Single game (Connect Four)")
    print("=" * 60)

    game = ConnectFourGame()
    a1 = Agent(id=0, system_prompt=DEFAULT_SYSTEM_PROMPT,
               strategy_prompt="Play center column (3) whenever possible. Control the center.")
    a2 = Agent(id=1, system_prompt=DEFAULT_SYSTEM_PROMPT,
               strategy_prompt="Block opponent threats. If opponent has 3 in a row, block immediately.")

    t0 = time.time()
    result = play_game(game, a1, a2, max_moves=50, verbose=True)
    elapsed = time.time() - t0

    print(f"\n  Result: {result['winner']} in {result['moves']} moves ({elapsed:.1f}s)")
    print(f"  Final memories: {result.get('final_memories', {})}")
    assert result["winner"] in ("red", "yellow", "draw"), f"Unexpected winner: {result['winner']}"
    print("  PASSED\n")
    return elapsed


def test_mini_tournament():
    """Step 2: 4-agent tournament."""
    print("=" * 60)
    print("TEST 2: Mini tournament (4 agents, Connect Four)")
    print("=" * 60)

    agents = create_initial_population(n=4)
    print(f"  Created {len(agents)} agents")
    for a in agents:
        sys_p = a.system_prompt[:50].replace('\n', ' ')
        print(f"    #{a.id}: {sys_p}...")

    t0 = time.time()
    ranked = run_tournament(agents, ConnectFourGame, verbose=True)
    elapsed = time.time() - t0

    print(f"\n  Tournament took {elapsed:.1f}s")
    print(f"  Rankings:")
    for r, a in enumerate(ranked, 1):
        print(f"    {r}. Agent #{a.id}: W={a.wins} L={a.losses} D={a.draws} score={a.score}")

    assert len(ranked) == 4
    assert all(a.total > 0 for a in ranked), "Some agents didn't play"
    print("  PASSED\n")
    return elapsed


def test_mutation():
    """Step 3: mutation."""
    print("=" * 60)
    print("TEST 3: Mutation")
    print("=" * 60)

    agent = Agent(id=0,
                  system_prompt="You are aggressive.",
                  strategy_prompt="Always capture. Material advantage wins.")
    agent.wins = 3
    agent.losses = 2
    agent.draws = 1
    agent.game_notes = ["Won as red vs Agent 1 in 20 moves",
                        "Lost as black vs Agent 2 in 35 moves"]

    t0 = time.time()
    new_sys, new_strat = mutate_agent(agent)
    elapsed = time.time() - t0

    print(f"  Mutation took {elapsed:.1f}s")
    print(f"  Old system: {agent.system_prompt[:60]}...")
    print(f"  New system: {new_sys[:60]}...")
    print(f"  Old strategy: {agent.strategy_prompt[:60]}...")
    print(f"  New strategy: {new_strat[:80]}...")

    # Check it actually changed (or at least didn't crash)
    assert len(new_sys) > 5, "System prompt too short"
    assert len(new_strat) > 10, "Strategy prompt too short"
    print("  PASSED\n")
    return elapsed


def test_crossover():
    """Step 4: crossover."""
    print("=" * 60)
    print("TEST 4: Crossover")
    print("=" * 60)

    a = Agent(id=0, system_prompt="You are aggressive.",
              strategy_prompt="Always attack. Capture whenever possible.")
    a.wins, a.losses = 4, 2

    b = Agent(id=1, system_prompt="You are defensive.",
              strategy_prompt="Play safe. Protect pieces. Wait for mistakes.")
    b.wins, b.losses = 3, 3

    t0 = time.time()
    new_sys, new_strat = crossover_agents(a, b)
    elapsed = time.time() - t0

    print(f"  Crossover took {elapsed:.1f}s")
    print(f"  Result system: {new_sys[:60]}...")
    print(f"  Result strategy: {new_strat[:80]}...")

    assert len(new_sys) > 5
    assert len(new_strat) > 10
    print("  PASSED\n")
    return elapsed


def test_full_generation():
    """Step 5: full generation (tournament + evolution)."""
    print("=" * 60)
    print("TEST 5: Full generation (4 agents, tournament + evolve)")
    print("=" * 60)

    agents = create_initial_population(n=4)

    t0 = time.time()
    ranked = run_tournament(agents, ConnectFourGame, verbose=True)
    t_tournament = time.time() - t0

    print(f"\n  Tournament: {t_tournament:.1f}s")
    for r, a in enumerate(ranked, 1):
        print(f"    {r}. #{a.id}: score={a.score} W={a.wins} L={a.losses} D={a.draws}")

    t0 = time.time()
    new_pop = evolve(ranked, verbose=True)
    t_evolve = time.time() - t0

    print(f"\n  Evolution: {t_evolve:.1f}s")
    print(f"  New population ({len(new_pop)} agents):")
    for a in new_pop:
        sys_p = a.system_prompt[:40].replace('\n', ' ')
        strat_p = a.strategy_prompt[:50].replace('\n', ' ')
        print(f"    #{a.id}: SYS=[{sys_p}...] STRAT=[{strat_p}...]")

    # Save state
    run_dir = get_run_dir()
    save_state(new_pop, 1, [], run_dir=run_dir)
    print(f"\n  State saved to {run_dir}")

    total = t_tournament + t_evolve
    print(f"\n  Total generation time: {total:.1f}s")
    print("  PASSED\n")
    return total


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("  SELFPLAYLLM SMOKE TEST")
    print("#" * 60 + "\n")

    # Step 0: LLM
    if not test_llm_alive():
        print("ABORT: LLM not available")
        sys.exit(1)

    # Step 1: single game
    game_time = test_single_game()

    # Step 2: mini tournament
    tournament_time = test_mini_tournament()

    # Step 3: mutation
    mutation_time = test_mutation()

    # Step 4: crossover
    crossover_time = test_crossover()

    # Step 5: full generation
    gen_time = test_full_generation()

    # Summary + estimate for big run
    print("=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)
    print(f"\n  Timing summary:")
    print(f"    Single game:   {game_time:.1f}s")
    print(f"    Tournament(4): {tournament_time:.1f}s ({tournament_time/12:.1f}s/game)")
    print(f"    Mutation:      {mutation_time:.1f}s")
    print(f"    Crossover:     {crossover_time:.1f}s")
    print(f"    Full gen(4):   {gen_time:.1f}s")

    # Estimate for 20 agents x 20 generations
    per_game = tournament_time / 12  # 4 agents = C(4,2)*2 = 12 games
    games_per_gen_20 = 20 * 19  # C(20,2)*2 = 380 games
    evo_ops_20 = 10  # 6 mutations + 4 crossovers
    evo_time_est = evo_ops_20 * max(mutation_time, crossover_time)
    gen_time_20 = games_per_gen_20 * per_game + evo_time_est
    total_20x20 = gen_time_20 * 20

    print(f"\n  Estimated for 20 agents x 20 generations:")
    print(f"    Per game:      {per_game:.1f}s")
    print(f"    Games/gen:     {games_per_gen_20}")
    print(f"    Per gen:       {gen_time_20/60:.1f} min")
    print(f"    Total:         {total_20x20/3600:.1f} hours")
