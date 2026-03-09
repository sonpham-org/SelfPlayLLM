# SelfPlayLLM Experimental Plan

## Machine: RTX 4090 (24GB VRAM)

## Measured Benchmarks (Qwen 3.5, RTX 4090)

### 4B Model (3.4GB download, 7.2GB VRAM loaded)
| Metric | Value |
|---|---|
| Game move (think=False, warm, realistic prompt) | 0.39-0.79s, avg ~0.5s |
| Mutation/crossover (think=False) | ~3.1s |
| Parallel throughput (8 workers) | ~2x sequential |

### 27B Model (Q4_K_M, 17GB download, ~20GB VRAM loaded)
| Metric | Value |
|---|---|
| Game move (think=False, warm) | ~0.44s |
| Slowdown vs 4B | ~2.7x |

### Critical Fix Applied
- `think=True` produces **empty content** with Qwen 3.5 (thinking tokens exhaust budget)
- Fix: Use `think=False` for all LLM calls (mutations, crossovers)

### Parallelism Results (4B, short prompts)
| Workers | Wall time (8 queries) | Effective speedup |
|---|---|---|
| 1 (seq) | 1.28s | 1.0x |
| 4 | 0.38s | 3.4x |
| 8 | 0.66s | 1.9x |
| 16 | 1.25s | 1.0x (saturated) |

Note: Games are internally sequential (move N depends on N-1).
Parallelism = running different games concurrently. Real game speedup ~1.8x.

---

## Plans (Doubled Generations)

### Plan A: Quick Validation
- **Config**: `--game connect4 -g 10 -p 10` (sequential, 4B)
- **Purpose**: Validate full pipeline end-to-end
- **Games/gen**: 90 (10 agents round-robin)
- **Est. time**: ~2.5 hours

### Plan B: Parallelized Benchmark
- **Config**: `--game connect4 -g 20 -p 10 -j 8` (8 workers, 4B)
- **Purpose**: Measure parallelism speedup, observe 20-gen evolution
- **Est. time**: ~2.9 hours

### Plan C: Model Comparison
- **C-4B**: `--game connect4 -g 10 -p 10 -j 8 --model qwen3.5:4b`
- **C-27B**: `--game connect4 -g 10 -p 10 -j 4 --model qwen3.5:27b`
- **Purpose**: Compare strategy quality between model sizes
- **Est. time**: ~1.4 hr (4B) + ~4.7 hr (27B) = ~6.1 hours

### Plan D: Multi-Game Evolution
- **Games**: othello, checkers (connect4 covered by Plan B)
- **Config**: `-g 20 -p 10 -j 8` (4B, 8 workers)
- **Purpose**: Test if evolved strategies are game-specific
- **Est. time**: ~5.6 hr (othello) + ~6.75 hr (checkers) = ~12.4 hours

---

## Total Estimated Runtime: ~24 hours

## Run Command
```bash
python run_all_plans.py
```

## Time Estimation Method
- Per LLM call (4B, game move): ~0.4s average
- Per LLM call (27B, game move): ~1.1s average
- Avg Connect Four game: ~25 moves = 25 LLM calls
- Avg Othello game: ~50 moves
- Avg Checkers game: ~60 moves
- Parallel speedup: 1.8x (8 workers, 4B), 1.5x (4 workers, 27B)
- Evolution overhead: ~15s/gen (4B), ~40s/gen (27B)
