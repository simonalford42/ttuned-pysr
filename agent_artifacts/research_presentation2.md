# Neural SR: Research Overview, Connections to Sandia, and Current Status

## Project Overview
This research project explores using transformer neural networks to enhance symbolic regression algorithms, specifically implementing "neural algorithmic tuning" to optimize algorithm performance beyond traditional hyperparameter optimization. The core concept is to train transformers on execution traces of symbolic regression algorithms and then improve them via reinforcement learning.

## Goals & Motivation

- Objective: Use transformers to tune symbolic regression (SR) algorithms at the algorithmic level — not just hyperparameters. Start with a simple, inspectable evolutionary SR (BasicSR), then distill and improve it with a transformer, and finally port the approach back to PySR.
- Hypothesis: A small transformer trained on evolution traces can learn “what to do next” during search (selection/mutation/crossover choices and candidates) to accelerate discovery and/or improve solution quality.
- Inspiration: Learning to imitate and then improve search/plan algorithms from traces and self‑generated data.
  - Stream of Search (SoS): learn to search from heuristic traces; STaR improves performance.
  - Beyond A*: imitate A* expansions, then bootstrap via shorter solutions.

## System Overview

- BasicSR (this repo): Minimal evolutionary SR in Python with explicit, readable operators and population dynamics. Used to generate trajectories and baselines.
- Trajectory Collection: Record full population state per generation (expressions + fitness, best/avg/diversity) to create rich training data.
- One‑Step Transformer Training: Convert trajectories into supervised examples of the form “context + current population → next expression”. Train GPT‑Neo variants on these.
- NeuralSR: Drop‑in replacement for BasicSR’s population update that samples new expressions from a trained transformer in parallel; falls back to evolutionary ops when suggestions are invalid.
- Benchmark Problems: Tiered suite from ultra‑simple to harder multi‑variate/rational forms for development, evaluation, and training data.

## Problem Benchmarks

- Ultra‑Simple (debugging): identity/constant/affine/quadratic like `y=x`, `y=2`, `y=x+1`, `y=2x`, `y=x^2`.
- Simple (10 problems): quadratics, cubics, rational, bi/tri‑variate products and sums, quartic, mixed polynomial, complex rational.
- Harder (5 problems): `pythagorean_3d`, `quadratic_formula_discriminant`, `mixed_polynomial`, `polynomial_product`, `surface_area_sphere_approx`.

## Implementation Highlights

- Core representation: Binary expression tree (`Node`) with operators `+ - * /`, constants `[1.0, 2.0]`, and variables `x0..xN`. Safe evaluation with numeric guards.
- Evolution loop (BasicSR):
  - Initialization: diverse seeds (variables, constants, x±c, x*c, x^2, plus random trees respecting `max_depth`/`max_size`).
  - Fitness: negative MSE with a simple complexity penalty (0.01·size).
  - Selection: tournament of size `S` with geometric preference; elitism keeps the best.
  - Variation: subtree crossover or targeted mutation with size constraints.
  - Early stopping: stop when MSE ≤ 3e‑16; optional wall‑clock `time_limit` per run.
- Reuse and clarity:
  - `generate_child_via_evolution(...)`: single place for tournament + crossover/mutation, used by both BasicSR and fallbacks in NeuralSR.
  - `record_population_state(...)`: captures expressions, fitnesses, best/avg/diversity each generation when `collect_trajectory=True`.
- Expression parsing: `expression_parser.py` converts model‑generated text back into ASTs to evaluate candidates.

## Data Pipeline for Training

- Trajectory to training examples (one‑step): `training/convert_trajectories.py`
  - Extracts header context (variables, operators, constants) from the run.
  - For each generation transition, formats: `<CONTEXT>{ctx}<POPULATION>{expr_i <FITNESS>fi}… <TARGET>{next_expr}`
  - Emits JSONL with fields: `context`, `population`, `target`, with metadata for problem name, generation, run id, and target index.
  - Splits into train/val sets (default 90/10).
- Shared formatting utils: `training/format_utils.py` to keep training and inference consistent.
- Training script: `training/train_one_step.py`
  - Tokenization: only the target tokens contribute loss (inputs masked to -100).
  - Models: GPT‑Neo tiny and small via config (`training/gpt-neo-*.json`).
  - Configs: `onestep-tiny.json` and `onestep-s.json` control data files, optimizer, logging, W&B integration, and checkpointing.
  - Outputs: Hugging Face checkpoints under `training/checkpoints/...` and a `final_model/` snapshot.

## NeuralSR Integration

- Interface: `NeuralSR` subclasses `BasicSR`, swapping the population update step.
- Parallel sampling: For a population of size N, sample N‑1 suggestions in one `generate()` call using `num_return_sequences` and `temperature`.
- Robustness: Track well‑formedness; if parsing fails or format is wrong, fall back to `generate_child_via_evolution(...)`.
- No size cap on neural suggestions: accept any parsed, well‑formed expression; size/depth caps still apply to fallback evolutionary ops.

## Experiments & Findings

1) Initialization vs Evolution (15 problems)
- Question: Does evolution provide real value beyond a good initialization?
- Result summary (evolution_vs_initialization.md):
  - Total: 15 problems (5 ultra‑simple + 10 simple)
  - Evolution helped: 9/15 (60.0%)
  - Same: 1/15 (6.7%)
  - Worse: 5/15 (most of these were already perfect at init)
  - Perfect solutions: 5 → 11 (120% increase)
- Takeaway: Evolution discovers structure not present in the initial population. Clear upgrades on many non‑trivial cases (adding missing terms, discovering factorization/interaction structure, correcting functional forms).

2) Improvement Trajectories under Time Budgets (Harder problems)
- Setup: Run each problem for 60s with time limit and MSE early stopping; parse progress logs into trajectories.
- Results (improvement_trajectories.md):
  - Pythagorean 3D: solved to ~7e‑31 MSE within ~0.3s, 14 tracked improvements, front‑loaded progress.
  - Quadratic discriminant: ~3e‑31 MSE in ~1.3s, 19 improvements, front‑loaded progress.
  - Mixed polynomial: ~1.7e‑31 MSE in ~0.2s, 12 improvements.
  - Polynomial product: ~1.5e‑31 MSE, 6 improvements, very fast.
  - Sphere area approx: best ~4.65e‑01 MSE in ~22s with steady progress and multiple breakthroughs.
- Takeaway: Many tasks are solved very quickly; one hard case shows sustained search improvements. These traces are rich supervision for “what to try next.”

3) Trajectory Collection at Scale (Harder problems)
- `collect_trajectories.py` gathers full populations per generation across multiple runs per problem with metadata (time, best/avg fitness, diversity).
- Outputs are standardized and feed directly into the training converter.

4) One‑Step Transformer Training
- Data: Converted BasicSR trajectories for the harder set into JSONL (`..._one_step_train/val.jsonl`).
- Models/configs:
  - Tiny config (`gpt-neo-tiny.json`): 256 hidden, 4 layers; batch 4.
  - Small config (`gpt-neo-s.json`): 1024 hidden, 16 layers; batch 2 with grad‑accum 2; reduced LR 3e‑4.
- Checkpoints: `training/checkpoints/onestep-full_20250811_145316/checkpoint-*` and `final_model/` present for tiny variant; S‑size runs scaffolded.
- Logging: W&B integrated via config.

5) NeuralSR Execution
- Status: Implemented parallel sampling path, format‑consistent prompting, and robust fallback. Comparison script (`neural_comparison.py`) supports head‑to‑head BasicSR vs NeuralSR with optional trajectory saving.
- Early metric to track: well‑formed suggestion rate (now recorded in `NeuralSR.save_trajectory`).

## Current Status

- BasicSR: Stable, readable, and solves most benchmark problems quickly. Early stopping and time limit working; evolution empirically helpful where needed.
- Trajectory/Conversion: End‑to‑end pipeline ready; produces clean one‑step examples with consistent formatting for training/inference.
- Training: Tiny model trained end‑to‑end with checkpoints and W&B. S‑size configuration prepared; multiple output dirs created as runs proceed.
- NeuralSR: Integrated with batch generation and fallbacks; parser‑based validation in place. Ready to run head‑to‑head comparisons against BasicSR using trained checkpoints.

## Roadmap (Short‑Term)

- Validate NeuralSR end‑to‑end
  - Run `neural_comparison.py` across the harder problems with the trained tiny model; record MSE/time and well‑formed rates.
  - Inspect failures; tune sampling parameters (temperature, max tokens) and target extraction.
- Improve data and targets
  - Balance per‑problem sampling; consider curriculum from ultra‑simple → harder.
  - Add richer features: generation index, simple summary stats for expressions (e.g., range/variance), or problem hints.
- Model and training tweaks
  - Train the S‑size config to convergence on current data; watch eval loss and well‑formedness proxy.
  - Consider smaller custom architectures for efficiency if needed.

## Roadmap (Mid‑Term, PySR Integration)

- Identify low‑latency “neural patch” points in PySR where a 1‑step suggestion is most impactful (selection heuristics, mutation/crossover choices, acceptance rules).
- Collect PySR‑native trajectories; unify the converter to the same input/target scheme.
- Validate latency: keep model small and fast; consider distillation or specialized tokenization for expressions/fitness.

## Risks & Mitigations

- Well‑formedness of generations: Mitigated with parser validation + fallbacks; can add constrained decoding or structural guidance.
- Overfitting to narrow distributions: Use broader/evolving problem sets and holdout tasks; evaluate generalization to new variable counts and operator mixes.
- Latency/regression risk in PySR: Keep models small; stage‑gate with ablations (how much does each neural hook help?).

## Repository Guide (Key Files)

- `basic_sr.py`: BasicSR implementation; `NeuralSR` subclass; AST `Node` type; training‑friendly logging and trajectory capture.
- `expression_parser.py`: Robust parser from string back to AST for model outputs.
- `problems.py` / `problem_splits.py`: Benchmarks and train/val/test splitting utilities.
- `collect_trajectories.py`: Multi‑run trajectory collection with standardized metadata.
- `time_comparison.py`: Time‑budgeted improvement trajectory analysis; exports `improvement_trajectories.md/json`.
- `training/convert_trajectories.py`: Trajectory → JSONL one‑step examples + train/val split.
- `training/train_one_step.py`: HF Trainer setup with target‑only loss masking; configs under `training/*.json`.
- `neural_comparison.py`: Head‑to‑head BasicSR vs NeuralSR runner, optional trajectory saving.

## Representative Results (Quick View)

- Evolution adds value
  - Helped on 60% of non‑trivial problems; perfect solutions rose from 5 to 11/15.
- Time‑trajectory behavior
  - Most tasks solved in <2s with front‑loaded improvements; one hard case exhibits steady, multi‑stage progress.
- Training assets
  - Checkpoints present for tiny model; S‑size training configs prepared with gradient accumulation and lower LR for stability.

## What To Run Next

- Train/validate the S‑size model with `training/onestep-s.json`, then compare with `neural_comparison.py --checkpoint <final_model>` on all 5 harder problems; capture well‑formed rate and MSE/time deltas.
- Sensitivity: vary temperature and max_new_tokens; try nucleus sampling; log parse failure modes.
- Data growth: add more trajectory runs and incorporate “harder” composition problems and mild noise to test robustness.

---

Notes and context originate from `readme.md`, `history.md`, improvement analyses, and the current codebase. This document is presentation‑ready for a project update: motivation, approach, implementation, experiments, and a clear near‑term plan.

