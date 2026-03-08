# SelfPlayLLM

Evolutionary self-play framework where LLM agents compete in board games. Agents carry text "strategy memories" that evolve over generations through mutation and crossover — powered entirely by LLM inference.

## How It Works

1. **10 agents** start with seed strategy memories (e.g., "Control the center", "Get kings early")
2. **Round-robin tournament** — every agent plays every other agent twice (once as each color)
3. **Champion challenges** — agents also play against a hall of fame of past winners
4. **Selection** — top 5 agents survive, bottom 5 are eliminated
5. **Evolution** — the LLM mutates/crosses over surviving strategies to create 5 new agents
6. **Repeat** — ELO ratings track skill across generations

The LLM does everything: picking moves during games, and rewriting strategies during evolution.

## Supported Games

- Checkers (primary)
- Connect Four, Othello, Hex, Mancala, Go
- Chess, Battleship, Stratego, Quoridor, Dots and Boxes, Poker

## Quick Start

```bash
# Run locally with Ollama
ollama pull qwen3.5:4b
python main.py --generations 10

# Resume a previous run
python main.py --generations 5 --resume runs/20260308_143000
```

## Architecture

```
main.py          — CLI runner, generation loop
agent.py         — LLM agent (move selection via Ollama/vLLM)
evolution.py     — Tournament, ELO ratings, mutation/crossover
checkers.py      — Checkers game engine
games/           — Additional game implementations
```

## Cloud Deployment

For large-scale runs, deploy on a GCP T4 spot instance (~$0.35/hr) with vLLM serving Qwen3.5-9B-AWQ:

```bash
vllm serve QuantTrio/Qwen3.5-9B-AWQ \
  --port 8000 \
  --max-model-len 4096 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

With async parallelism, a single T4 can handle 10-15 concurrent games, enabling 1,000+ generations for ~$100.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.ai) (local) or vLLM (cloud)
- A Qwen3/3.5 model (tested with qwen3.5:4b and Qwen3.5-9B)
