Embedder Architecture Improvement Plan

Overview
- Task: Improve point embedder for direct expression prediction from (X, y) samples, aiming for higher R^2 and direct accuracy on mixed-complexity datasets.
- Baseline: E2EPointEmbedder — per-point 2-layer MLP, returns one prefix token per point, no pooling/statistics, no mask awareness, no normalization.

Data Inspection (spot checks)
- Direct JSONL entries contain: `X_data` (N×V), `y_data` (N), and `target` (string expression).
- V varies up to 4–10 in current sets; N varies per example (tens to hundreds). Magnitude ranges across both inputs and outputs vary widely (negatives, positives, near-zero), suggesting the need for per-sample normalization.
- Ordering of points is arbitrary from generators; the model should not depend on point order.

Baseline Limitations
- Order sensitivity: treating the set of points as a sequence may cause overfitting to arbitrary ordering.
- Scale sensitivity: no normalization; large magnitude variance across samples likely hurts generalization and makes the LM’s job harder.
- Inefficient prefix budget: one token per point consumes context; many padded tokens carry no information and can distract the LM.
- No pooling/statistics: lacks global cues (moments, trends) and permutation invariance.

Design Goals
- Permutation invariance (or robustness) to point order.
- Scale invariance via per-sample normalization.
- Compact, informative prefix using a fixed number of summary tokens (small prefix_len) to free LM capacity for decoding.
- General-purpose features (avoid operator-specific hacks).

Candidate Architectures
1) Set Encoder + Cross-Attention Pooling (proposed: SetPointEmbedder)
   - Per-point MLP → TransformerEncoder over points (mask-aware).
   - Learnable queries attend to point encodings (Perceiver/SetTransformer PMA-like) to produce P summary tokens.
   - Optional random Fourier features for richer signal; per-sample normalization for scale.
   - Outputs P prefix tokens (P ≪ N), enabling longer target sequences and stronger semantics per token.

2) Point-Transformer Prefix (sequence encoder)
   - Per-point MLP + positional encodings (randomized or sinusoidal) + Transformer over the point sequence.
   - Keep K first tokens or pool to K tokens.
   - Pros: simple; Cons: order-sensitive unless augmented with randomization; less principled invariance.

3) Moment/Kernel Embeddings
   - Compute permutation-invariant statistics: per-dim means/stds/covariances; random Fourier feature means of φ(x), φ(x)·y, etc.
   - Concatenate into a small set of tokens via MLP.
   - Pros: very simple, robust; Cons: may underfit complex structures without attention.

4) Hybrid (summary + sparse raw points)
   - Emit P summary tokens (as in 1) + a small R of raw per-point tokens (reservoir-sampled) to retain fine detail.
   - Trade-off between invariance and fidelity.

Initial Choice and Rationale
- Start with (1) SetPointEmbedder: compact, permutation-robust, mask-aware, simple to implement.
- Add optional Fourier features (low-dimensional) for nonlinearity without assuming specific operators.
- Normalize per-sample to control scale.

Implementation Snapshot
- SetPointEmbedder in `modules.py`:
  - Input: `(points_X, points_y, points_mask)`.
  - Build features: pad/truncate X to max_input_dim; concat y; optional per-sample normalization; optional random Fourier features; point MLP.
  - Point encoder: `nn.TransformerEncoder` with mask.
  - Pooling: learnable queries + `nn.MultiheadAttention` to produce `prefix_len` tokens.
  - Output: project to LM hidden size + LayerNorm.
- Wiring:
  - `train.py`: new `input_embedder: "set"`, reserves `prefix_len` tokens (vs. `max_points`), passes `points_mask` from collator.
  - `utils.py`: save/load `embedder_config.json` to reconstruct embedder class and hparams.
  - Backward compatible with E2E embedder.

Experiment Plan (3–7 days)
- Metrics: Direct prediction mean R^2, MSE, and within-tolerance accuracy.

Phase A — Fast iteration (Day 1–2)
- Dataset: `arith_10k_b3` train, `arith_100_b3` val; config: `configs/direct-set-small.json`.
- Ablations (short runs, ~40 epochs tiny model):
  1. Base SetPointEmbedder: prefix_len=16, no Fourier.
  2. Fourier features: fourier_features ∈ {2, 4}.
  3. Vary prefix_len ∈ {8, 16, 24} to study budget vs. quality.
  4. Normalize on/off comparison.
- Compare against E2E baseline on same setup.

Phase B — Scale up (Day 3–5)
- If gains hold, try a medium run (fewer epochs) on mixed-complexity set, e.g. a 10k–20k slice of `arith_100k_c05`.
- Tune: d_model (hidden_size/4 … hidden_size/2), layers {1,2,3}, heads {4,8}.

Phase C — Long run (Day 5–7)
- Full `arith_100k_c05` for 30–40 epochs if compute allows; freeze the best hyperparameters from Phase B.

Evaluation & Commands
- Train (example):
  - `python train.py --config configs/direct-set-small.json`
- Evaluate:
  - `python eval_direct.py --checkpoint training/checkpoints/direct-set-small_<RUN>/final_model --num_samples 20`

Risk Mitigation & Notes
- Avoid reward hacking: do not encode operator-specific signals; Fourier features and normalization are general-purpose.
- Keep architecture simple: 2-layer encoder + single cross-attn is understandable and not the paper’s focus.
- If order leakage remains a concern, consider shuffling points per batch during collate (optional future step).

Next Steps
- Run baseline vs. SetPointEmbedder on 10k_b3.
- Based on metrics, sweep Fourier features and prefix_len.
- If promising, proceed to mixed-complexity subsets and then to the 100k run.

