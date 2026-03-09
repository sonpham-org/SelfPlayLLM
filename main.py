#!/usr/bin/env python3
"""Evolving Game Agents — main runner.

10 agents play round-robin on any supported game. Top 5 advance, bottom 5 get
replaced by mutated/crossover versions. System prompts + strategy templates
evolve over generations.
"""

import argparse
import time

import agent

from evolution import (
    create_initial_population, run_tournament, evolve,
    save_state, load_state, get_run_dir, Champion,
)

# Game registry — maps name to factory function
GAMES = {}


def register_games():
    """Import and register all available game engines."""
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


def print_leaderboard(agents, gen, game_name, hall_of_fame=None):
    print(f"\n{'='*70}")
    print(f"  GENERATION {gen} — {game_name.upper()}")
    print(f"{'='*70}")
    print(f"  {'Rank':<5} {'Agent':<8} {'ELO':<6} {'Score':<7} {'W':<4} {'L':<4} {'D':<4} {'WR':<7}")
    print(f"  {'-'*52}")
    for rank, a in enumerate(agents, 1):
        print(f"  {rank:<5} #{a.id:<7} {a.elo:<6} {a.score:<7} "
              f"{a.wins:<4} {a.losses:<4} {a.draws:<4} {a.winrate:.0%}")
    print()
    print("  Top 5 strategies:")
    for a in agents[:5]:
        sys_p = a.system_prompt[:40].replace('\n', ' ')
        strat_p = a.strategy_prompt[:60].replace('\n', ' ')
        print(f"    #{a.id}: SYS=[{sys_p}...] STRAT=[{strat_p}...]")
    if hall_of_fame:
        print(f"\n  Hall of Fame ({len(hall_of_fame)} champions):")
        for c in sorted(hall_of_fame, key=lambda c: c.elo, reverse=True)[:5]:
            strat_p = c.strategy_prompt[:60].replace('\n', ' ')
            print(f"    {c.label}: ELO {c.elo} — {strat_p}...")
    print(f"{'='*70}\n")


def main():
    register_games()

    parser = argparse.ArgumentParser(description="Evolving Game Agents")
    parser.add_argument("--game", "-G", type=str, default="checkers",
                        choices=list(GAMES.keys()),
                        help=f"Game to play ({', '.join(GAMES.keys())})")
    parser.add_argument("--generations", "-g", type=int, default=5,
                        help="Number of generations")
    parser.add_argument("--resume", type=str, default=None, metavar="RUN_DIR",
                        help="Resume from an existing run directory")
    parser.add_argument("--population", "-p", type=int, default=10,
                        help="Number of agents in population")
    parser.add_argument("--champions", "-c", type=int, default=0,
                        help="Number of past champions to challenge (0 = disabled)")
    parser.add_argument("--parallel", "-j", type=int, default=1,
                        help="Parallel game workers (default: 1 = sequential)")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="LLM model name for Ollama (default: qwen3.5:4b)")
    parser.add_argument("--run-name", "-n", type=str, default=None,
                        help="Name prefix for the run directory")
    args = parser.parse_args()

    if args.model:
        agent.set_model(args.model)

    game_factory = GAMES[args.game]
    run_dir = get_run_dir(args.resume, name=args.run_name)

    if args.resume:
        print(f"Resuming from {run_dir}")
        agents, start_gen, log, hall_of_fame = load_state(run_dir)
        start_gen += 1
    else:
        agents = create_initial_population(n=args.population)
        start_gen = 1
        log = []
        hall_of_fame = []

    print(f"Run directory: {run_dir}")
    print(f"Game: {args.game}")
    print(f"Model: {agent.MODEL}")
    print(f"Running {args.generations} generation(s) with {len(agents)} agents")
    rr_games = len(agents) * (len(agents) - 1)
    workers_str = f"{args.parallel} workers" if args.parallel > 1 else "sequential"
    print(f"Each generation: {rr_games} round-robin games ({workers_str})\n")

    for gen in range(start_gen, start_gen + args.generations):
        print(f"\n{'#'*70}")
        print(f"  GENERATION {gen} — {args.game.upper()}")
        print(f"{'#'*70}")

        t0 = time.time()

        # Round-robin tournament (with mid-tournament checkpoints)
        print("\nTournament:")
        _save = lambda: save_state(agents, gen, log, hall_of_fame, run_dir)
        agents = run_tournament(agents, game_factory, verbose=True,
                                max_workers=args.parallel,
                                save_callback=_save, save_every=20)
        elapsed = time.time() - t0

        print_leaderboard(agents, gen, args.game, hall_of_fame)

        # Record top 2 as new champions
        for a in agents[:2]:
            hall_of_fame.append(Champion(
                gen=gen, id=a.id,
                system_prompt=a.system_prompt,
                strategy_prompt=a.strategy_prompt,
                elo=a.elo,
            ))

        gen_data = {
            "generation": gen,
            "game": args.game,
            "elapsed_s": round(elapsed, 1),
            "rankings": [
                {
                    "id": a.id, "score": a.score,
                    "wins": a.wins, "losses": a.losses, "draws": a.draws,
                    "elo": a.elo,
                    "system_prompt": a.system_prompt,
                    "strategy_prompt": a.strategy_prompt,
                }
                for a in agents
            ],
        }
        log.append(gen_data)

        # Evolve (except last generation)
        if gen < start_gen + args.generations - 1:
            agents = evolve(agents, verbose=True)
            print("\nNew population:")
            for a in agents:
                sys_p = a.system_prompt[:50].replace('\n', ' ')
                strat_p = a.strategy_prompt[:60].replace('\n', ' ')
                print(f"  #{a.id}: ELO {a.elo} — SYS=[{sys_p}...] STRAT=[{strat_p}...]")

        # Save after each generation
        save_state(agents, gen, log, hall_of_fame, run_dir)
        print(f"\nState saved to {run_dir}. Generation took {elapsed:.1f}s")

    print(f"\n{'='*70}")
    print("  EVOLUTION COMPLETE")
    print(f"{'='*70}")
    print(f"\nBest agent: #{agents[0].id} (ELO {agents[0].elo})")
    print(f"System prompt: {agents[0].system_prompt}")
    print(f"Strategy prompt:\n{agents[0].strategy_prompt}")
    if hall_of_fame:
        best_champ = max(hall_of_fame, key=lambda c: c.elo)
        print(f"\nAll-time best champion: {best_champ.label} (ELO {best_champ.elo})")
    print(f"\nFull results saved to {run_dir}/evo_state.json")


if __name__ == "__main__":
    main()
