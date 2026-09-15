# Experiment 5b: fictional fixed-capacity handoffs

- **Design:** 20 fictional dossiers; each dossier is the independent unit.
- **Manipulation:** generic versus one-question-conditioned selection into exactly K evidence slots (`K=[2, 4, 6]`).
- **Isolation:** hidden questions are never shown to either selector; the fixed answerer receives only one sealed handoff and one evaluation query.
- **Interpretation:** this is a controlled closed-world evidence-allocation test. The published natural-prose Experiment 5 supplies the complementary naturalistic result.

## Primary capacity: K=4

| Arm | Immediate judge accuracy | Reusable judge accuracy |
|---|---:|---:|
| generic | 0.675 [0.667, 0.692] | 0.675 [0.667, 0.692] |
| conditioned | 0.992 [0.975, 1.000] | 0.613 [0.603, 0.625] |

### Conditioned minus generic

| Endpoint | Delta | 95% CI | p |
|---|---:|---:|---:|
| immediate | +0.317 | [+0.292, +0.333] | <0.001 |
| reusable | -0.062 | [-0.077, -0.048] | <0.001 |

Specialization interaction: **+0.378** [+0.355, +0.395]. This is secondary: a trade-off requires a positive immediate delta and a negative reusable delta separately.

## Baseline checks

| Material | Metric | Mean | 95% CI |
|---|---|---:|---:|
| direct_source | f1 | 0.921 | [0.883, 0.955] |
| direct_source | judge_correct | 0.992 | [0.975, 1.000] |
| direct_cards | f1 | 0.911 | [0.860, 0.954] |
| direct_cards | judge_correct | 1.000 | [1.000, 1.000] |
| closed_book | f1 | 0.000 | [0.000, 0.001] |
| closed_book | judge_correct | 0.000 | [0.000, 0.000] |

## Integrity checks

- Valid selector outputs: 0.997.
- Rotation rows with exactly K selected cards: 0.997.
- Confidence intervals bootstrap dossier means, not question rotations.
- Invalid selections remain in the intention-to-treat result as zero-answer technical failures; inspect `selections.jsonl` for parse diagnostics.

## Paired-compliant no-selection control

At K=6, valid conditioned and generic packets both contain all six cards. Their reusable judge accuracy delta is **+0.000** [+0.000, +0.000] over 118 paired rotations.

## Artifacts

- `metrics.csv`: dossier-clustered arm estimates.
- `contrasts.csv`: paired conditioned-minus-generic effects.
- `rotation_rows.csv`: auditable per-rotation outcomes and exact selections.
- `baseline_metrics.csv`: source, full-card, and closed-book controls.
- `fixed_capacity_generalization.png`: immediate/reusable capacity curves.
