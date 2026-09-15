# Experiment 5b: fictional fixed-capacity handoffs

- **Design:** 1 fictional dossiers; each dossier is the independent unit.
- **Manipulation:** generic versus one-question-conditioned selection into exactly K evidence slots (`K=[4]`).
- **Isolation:** hidden questions are never shown to either selector; the fixed answerer receives only one sealed handoff and one evaluation query.
- **Interpretation:** this is a controlled closed-world evidence-allocation test. The published natural-prose Experiment 5 supplies the complementary naturalistic result.

## Primary capacity: K=4

| Arm | Immediate token F1 | Reusable token F1 |
|---|---:|---:|
| generic | 0.667 [0.667, 0.667] | 0.667 [0.667, 0.667] |
| conditioned | 1.000 [1.000, 1.000] | 0.617 [0.617, 0.617] |

### Conditioned minus generic

| Endpoint | Delta | 95% CI | p |
|---|---:|---:|---:|
| immediate | +0.333 | [+0.333, +0.333] | 0.000 |
| reusable | -0.050 | [-0.050, -0.050] | 0.000 |

Specialization interaction: **+0.383** [+0.383, +0.383]. This is secondary: a trade-off requires a positive immediate delta and a negative reusable delta separately.

## Baseline checks

| Material | Metric | Mean | 95% CI |
|---|---|---:|---:|
| direct_source | f1 | 1.000 | [1.000, 1.000] |
| direct_cards | f1 | 1.000 | [1.000, 1.000] |
| closed_book | f1 | 0.000 | [0.000, 0.000] |

## Integrity checks

- Valid selector outputs: 1.000.
- Rotation rows with exactly K selected cards: 1.000.
- Confidence intervals bootstrap dossier means, not question rotations.
- Invalid selections remain in the intention-to-treat result as zero-answer technical failures; inspect `selections.jsonl` for parse diagnostics.

## Artifacts

- `metrics.csv`: dossier-clustered arm estimates.
- `contrasts.csv`: paired conditioned-minus-generic effects.
- `rotation_rows.csv`: auditable per-rotation outcomes and exact selections.
- `baseline_metrics.csv`: source, full-card, and closed-book controls.
- `fixed_capacity_generalization.png`: immediate/reusable capacity curves.
