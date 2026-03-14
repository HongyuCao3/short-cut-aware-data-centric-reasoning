# VRAM Reduction Plan for ShortcutScore Gradient Computation

## Problem Statement

The current implementation of `compute_sample_gradients_batched()` in `src/methods.py`
causes CUDA OOM errors during the ShortcutScore computation phase. The root cause is that
per-sample gradient vectors are computed via backpropagation and accumulated in GPU memory
before being stacked.

### Memory Breakdown (Synthetic Config)

| Component | Size | VRAM |
|-----------|------|------|
| Model parameters | ~19M params | 76 MB |
| Single gradient vector | 19M floats × 4 bytes | 76 MB |
| score_batch_size=64, one type | 64 × 76 MB | 4.86 GB |
| Three types stacked (full/ans/reason) | 3 × 4.86 GB | **14.6 GB** |
| + model + activations + optimizer | — | **~25-28 GB** |

The exact OOM point (confirmed in `hp_optuna/optuna.log`) is `methods.py:187`:
```python
return torch.stack(g_fulls), torch.stack(g_anss), torch.stack(g_reasons)
```

The NL config is worse: ~163M params → 652 MB per gradient vector.

---

## Two Proposed Directions

---

### Direction 1: MeZO-style Training (Phase 1 / 2 / 3)

**Scope:** Applies to the standard training loops in `trainer.py` (Phases 1, 2, 3),
**not** to the ShortcutScore computation phase.

**Idea:**

Replace the Adam optimizer + backpropagation in training phases with the MeZO
(Memory-Efficient Zeroth-Order) optimizer from the paper:

> Malladi et al., "Fine-Tuning Language Models with Just Forward Passes", NeurIPS 2023.

MeZO estimates gradients using only two forward passes via SPSA
(Simultaneous Perturbation Stochastic Approximation):

```
projected_grad = (L(θ + εz; B) - L(θ - εz; B)) / (2ε)
θ_i ← θ_i - η * projected_grad * z_i     (in-place update)
```

The key trick: by re-seeding the same random number generator, `z` is never materialized
as a full vector — parameters are perturbed and updated one-by-one in place.

**Memory savings:**

| Component | With Adam (current) | With MeZO |
|-----------|-------------------|-----------|
| Model params | 76 MB | 76 MB |
| Gradient buffer | 76 MB | 0 MB |
| Optimizer states (Adam m, v) | ~152 MB | 0 MB |
| Activation cache (backprop) | varies | 0 MB |
| **Total training overhead** | **~3-4× inference** | **~1× inference** |

**Tradeoffs:**

| Aspect | Detail |
|--------|--------|
| Memory reduction | ~3× in training phases |
| Convergence speed | Slower per-step convergence; requires more steps |
| Implementation complexity | Moderate — need to rewrite training loop |
| Correctness risk | MeZO requires prompt-based fine-tuning to work well (see paper Section A); needs validation on synthetic tasks |
| Scope | Does NOT solve the ShortcutScore OOM — that is a separate phase |

**Affected files:**
- `src/trainer.py` — training loop in `train_phase1()`, `train_phase2()`, `train_phase3()`
- `src/config.py` — add MeZO-specific hyperparameters (`epsilon`, `mezo_lr`)

---

### Direction 2: Random Projection Sketch for ShortcutScore

**Scope:** Directly targets the OOM source — the `compute_sample_gradients_batched()`
function in `src/methods.py`.

**Idea:**

Instead of storing and comparing full gradient vectors `g ∈ R^d`, use random projections
to compress each gradient into a sketch `s ∈ R^k` (where `k << d`), then approximate
cosine similarities from the sketches.

This is theoretically grounded by the **Johnson-Lindenstrauss Lemma**: for random unit
vectors `z_1, ..., z_k`, the projected values `{z_j⊤g}` preserve pairwise inner products
(and thus cosine similarities) with high probability when k is large enough.

**Algorithm sketch:**

```
# Step 1: Compute validation gradient sketch (once, shared across all samples)
for j in range(k):
    z_j ~ N(0, I_d)                                    # random direction
    s_V[j] = (L_val(θ + ε*z_j) - L_val(θ - ε*z_j)) / (2ε)   # 2 fwd passes

# Step 2: For each training sample i, compute per-sample sketches
for each sample i:
    for j in range(k):
        z_j ~ N(0, I_d)   # same z_j reused via fixed seed
        s_full[i,j] = (L_full_i(θ + ε*z_j) - L_full_i(θ - ε*z_j)) / (2ε)
        s_ans[i,j]  = (L_ans_i(θ + ε*z_j)  - L_ans_i(θ - ε*z_j))  / (2ε)
        s_reason[i,j]= (L_reason_i(θ + ε*z_j)- L_reason_i(θ - ε*z_j))/ (2ε)

# Step 3: Approximate cosine similarities from sketches
cos(g_full_i, g_V)      ≈ s_full[i] · s_V / (‖s_full[i]‖ · ‖s_V‖)
cos(g_ans_i, g_reason_i) ≈ s_ans[i] · s_reason[i] / (‖s_ans[i]‖ · ‖s_reason[i]‖)
```

**Memory savings:**

| Component | Current | With Sketch (k=512) |
|-----------|---------|---------------------|
| Per-sample gradient vector | 76 MB (19M floats) | 2 KB (512 floats) |
| Batch of 64 × 3 types | 14.6 GB | ~0.4 MB |
| **Reduction factor** | — | **~36,000×** |

For the NL config (163M params): 652 MB per vector → still 2 KB with k=512.

**Compute cost:**

- Current: 3 backward passes per sample
- Sketch: 2k forward passes per sample (k=512 → 1024 forward passes)
- Forward pass is ~2-3× faster than backward pass, so wall-clock cost is roughly
  `1024 / (3 * 0.4) ≈ 853×` slower per sample without batching optimizations.
- With batching over samples and sharing z_j across samples in a batch, this can
  be reduced significantly.

**Key implementation detail — sharing z across samples:**

The most important optimization: for a fixed `z_j`, compute `L(θ + ε*z_j)` for the
entire batch simultaneously (one forward pass for all samples), instead of per sample.
This keeps the 2k forward passes total (not per sample), making the compute overhead
similar to the current approach.

```
for j in range(k):
    z_j ~ N(0, I_d)
    perturb model by +ε*z_j  (in-place, no extra memory)
    compute L_full, L_ans, L_reason for ALL samples in batch  # 1 batched fwd pass
    perturb model by -2ε*z_j
    compute L_full, L_ans, L_reason for ALL samples in batch  # 1 batched fwd pass
    reset model by +ε*z_j
    update sketches: s_full[:,j] = (L+_full - L-_full) / (2ε)  # B scalars
```

With this batching, total forward passes = `2k` (independent of batch size B).

**Tradeoffs:**

| Aspect | Detail |
|--------|--------|
| Memory reduction | ~36,000× for gradient storage (the OOM bottleneck) |
| Approximation error | Cosine similarity estimated with error O(1/√k); k=512 gives ~4% error (JL bound) |
| Effect on ShortcutScore quality | **Unknown — requires empirical validation.** The downstream task is binary classification of shortcut samples, which may tolerate approximation error well |
| Compute cost (naive) | Much slower without batching optimizations |
| Compute cost (batched z) | Similar to current if k is chosen carefully |
| Implementation complexity | High — requires rewriting `compute_sample_gradients_batched()` and `compute_shortcut_scores_batched()` |
| Affected files | `src/methods.py`, `src/trainer.py`, `src/config.py` |

**Hyperparameter introduced:** `k` (sketch dimension), suggested search range: {64, 128, 256, 512, 1024}.

---

## Comparison and Recommendation

| | Direction 1 (MeZO Training) | Direction 2 (Sketch Score) |
|---|---|---|
| Solves the OOM | No (different phase) | Yes (directly) |
| Memory reduction | ~3× in training | ~36,000× in scoring |
| Algorithmic change | Training optimizer only | Core scoring algorithm |
| Correctness risk | Low for training loss | Medium (approximation) |
| Implementation effort | Moderate | High |
| Dependency | Requires validation on synthetic tasks | Requires k-tuning experiments |

**Recommended approach:**

1. **First implement Direction 2** — it directly eliminates the OOM and is the critical blocker.
   Start with a small k (e.g., 128) to verify the approach doesn't destroy ShortcutScore quality,
   then tune k upward.

2. **Optionally add Direction 1** as a secondary optimization for training phases once
   Direction 2 is validated.

---

## Open Questions

1. Does approximate ShortcutScore (via sketch) preserve the ranking of shortcut samples
   well enough to produce equivalent reweighting/filtering decisions?
2. What is the minimum k that produces acceptable downstream performance?
3. For the NL config (real language tasks), does the approximation degrade more due to
   higher dimensionality or concentration of measure effects?
4. Can the sketch vectors z_j be shared across the validation set computation and the
   per-sample computation to further reduce passes?

---

## References

- Malladi et al., "Fine-Tuning Language Models with Just Forward Passes", NeurIPS 2023.
  [arXiv:2305.17333](https://arxiv.org/abs/2305.17333)
- Johnson & Lindenstrauss (1984) — random projection dimension reduction lemma.
- `src/methods.py` — `compute_sample_gradients_batched()`, `compute_shortcut_scores_batched()`
- `src/trainer.py` — `_compute_sample_scores()`, `train_our_method()`
- `hp_optuna/optuna.log` — OOM traces confirming exact failure location
