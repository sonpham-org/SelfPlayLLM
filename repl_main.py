#!/usr/bin/env python3
"""Evolving REPL Game Agents — main runner.

Like main.py but agents get a multi-step REPL loop per turn:
- python code execution
- move simulation
- subagent analysis spawning

Evolution optimizes both prompts AND numeric REPL parameters
(max_steps, subagent_budget, analysis_width).
"""

import argparse
import time

import agent

from evolution import run_tournament, get_run_dir, elo_update
from repl_agent import (
    REPLAgent, play_game_repl, create_repl_population,
    evolve_repl, save_repl_state, load_repl_state,
)

# Game registry
GAMES = {}


def register_games():
    from games.checkers_wrap import CheckersGame
    from games.othello import OthelloGame
    from games.connect_four import ConnectFourGame
    from games.mancala import MancalaGame
    from games.hex_game import HexGame
    from games.dots_and_boxes import DotsAndBoxesGame
    from games.quoridor import QuoridorGame
    from games.chess_game import ChessGame
    from games.battleship import BattleshipGame
    from games.poker import PokerGame
    from games.stratego import StrategoGame
    from games.go import GoGame

    GAMES.update({
        "checkers":   CheckersGame,
        "othello":    OthelloGame,
        "connect4":   ConnectFourGame,
        "mancala":    MancalaGame,
        "hex":        HexGame,
        "dots":       DotsAndBoxesGame,
        "quoridor":   QuoridorGame,
        "chess":      ChessGame,
        "battleship": lambda: BattleshipGame(seed=None),
        "poker":      lambda: PokerGame(seed=None),
        "stratego":   lambda: StrategoGame(seed=None),
        "go":         GoGame,
    })


def print_leaderboard(agents, gen, game_name):
    print(f"\n{'='*78}")
    print(f"  GENERATION {gen} — {game_name.upper()} (REPL AGENTS)")
    print(f"{'='*78}")
    print(f"  {'Rk':<4} {'Agent':<7} {'ELO':<6} {'Score':<6} {'W':<3} {'L':<3} {'D':<3} "
          f"{'WR':<6} {'Steps':<6} {'Sub':<4} {'Wid':<4} {'LLM':<5}")
    print(f"  {'-'*68}")
    for rank, a in enumerate(agents, 1):
        print(f"  {rank:<4} #{a.id:<6} {a.elo:<6} {a.score:<6} "
              f"{a.wins:<3} {a.losses:<3} {a.draws:<3} {a.winrate:<5.0%} "
              f"{a.max_steps:<6} {a.subagent_budget:<4} {a.analysis_width:<4} "
              f"{a.total_llm_calls:<5}")
    print()
    print("  Top strategies:")
    for a in agents[:5]:
        strat = a.strategy_prompt[:70].replace('\n', ' ')
        print(f"    #{a.id} (s={a.max_steps},sub={a.subagent_budget},w={a.analysis_width}): {strat}...")
    print(f"{'='*78}\n")


def main():
    register_games()

    parser = argparse.ArgumentParser(description="Evolving REPL Game Agents")
    parser.add_argument("--game", "-G", type=str, default="connect4",
                        choices=list(GAMES.keys()))
    parser.add_argument("--generations", "-g", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None, metavar="RUN_DIR")
    parser.add_argument("--population", "-p", type=int, default=10)
    parser.add_argument("--parallel", "-j", type=int, default=1)
    parser.add_argument("--model", "-m", type=str, default=None)
    parser.add_argument("--run-name", "-n", type=str, default=None)
    args = parser.parse_args()

    if args.model:
        agent.set_model(args.model)

    game_factory = GAMES[args.game]
    run_dir = get_run_dir(args.resume, name=args.run_name or f"repl_{args.game}")

    if args.resume:
        print(f"Resuming from {run_dir}")
        agents, start_gen, log = load_repl_state(run_dir)
        start_gen += 1
    else:
        agents = create_repl_population(n=args.population)
        start_gen = 1
        log = []

    print(f"Run directory: {run_dir}")
    print(f"Game: {args.game}")
    print(f"Model: {agent.MODEL}")
    print(f"Mode: REPL agents (tool-integrated reasoning)")
    print(f"Running {args.generations} generation(s) with {len(agents)} agents")
    rr_games = len(agents) * (len(agents) - 1)
    workers_str = f"{args.parallel} workers" if args.parallel > 1 else "sequential"
    print(f"Each generation: {rr_games} round-robin games ({workers_str})")
    print(f"\nAgent REPL configs:")
    for a in agents:
        print(f"  #{a.id}: max_steps={a.max_steps}, "
              f"subagent={a.subagent_budget}, width={a.analysis_width}")
    print()

    for gen in range(start_gen, start_gen + args.generations):
        print(f"\n{'#'*78}")
        print(f"  GENERATION {gen} — {args.game.upper()} (REPL)")
        print(f"{'#'*78}")

        t0 = time.time()

        # Tournament using REPL play function
        print("\nTournament:")
        _save = lambda: save_repl_state(agents, gen, log, run_dir)
        agents = run_tournament(
            agents, game_factory, verbose=True,
            max_workers=args.parallel,
            save_callback=_save, save_every=20,
            play_fn=play_game_repl,
        )
        elapsed = time.time() - t0

        print_leaderboard(agents, gen, args.game)

        gen_data = {
            "generation": gen,
            "game": args.game,
            "elapsed_s": round(elapsed, 1),
            "rankings": [
                {
                    "id": a.id, "score": a.score,
                    "wins": a.wins, "losses": a.losses, "draws": a.draws,
                    "elo": a.elo,
                    "max_steps": a.max_steps,
                    "subagent_budget": a.subagent_budget,
                    "analysis_width": a.analysis_width,
                    "total_llm_calls": a.total_llm_calls,
                    "strategy_prompt": a.strategy_prompt,
                }
                for a in agents
            ],
        }
        log.append(gen_data)

        # Evolve (except last generation)
        if gen < start_gen + args.generations - 1:
            agents = evolve_repl(agents, verbose=True)
            print("\nNew population:")
            for a in agents:
                strat = a.strategy_prompt[:60].replace('\n', ' ')
                print(f"  #{a.id}: ELO={a.elo} s={a.max_steps} "
                      f"sub={a.subagent_budget} w={a.analysis_width} — {strat}...")

        # Save after each generation
        save_repl_state(agents, gen, log, run_dir)
        print(f"\nState saved to {run_dir}. Generation took {elapsed:.1f}s")

    print(f"\n{'='*78}")
    print("  REPL EVOLUTION COMPLETE")
    print(f"{'='*78}")
    best = agents[0]
    print(f"\nBest agent: #{best.id} (ELO {best.elo})")
    print(f"  max_steps={best.max_steps}, subagent_budget={best.subagent_budget}, "
          f"analysis_width={best.analysis_width}")
    print(f"System prompt: {best.system_prompt[:100]}...")
    print(f"Strategy:\n{best.strategy_prompt}")
    print(f"\nResults saved to {run_dir}/evo_state.json")


if __name__ == "__main__":
    main()
