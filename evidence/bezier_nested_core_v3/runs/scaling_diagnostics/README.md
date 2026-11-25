# Core learning diagnostic

These probes used CPU only (two threads), the enlarged **235,210-parameter**
core, and the expanded corpus's 16,384 training examples and 512 validation
examples. Test and longer-program examples were not evaluated. No model source
was changed for these probes.

The direct syntax-only path was trained independently to distinguish its
learnability from interference with outer prediction and control losses. The
core retained its exact 256 → 458 → 256 architecture and fixed projections.
Each configuration used seed 7, batch size 256, AdamW weight decay 0.01, gradient
clipping at 1, and 2,000 updates. Checkpoints were compared by validation NLL.

| Learning rate | Best validation step | Train accuracy | Validation accuracy | Validation NLL |
| ---: | ---: | ---: | ---: | ---: |
| 0.001 | 1,500 | 63.39% | 58.20% | 0.68337 |
| 0.003 | 500 | 59.03% | 55.27% | 0.67921 |

Uniform binary prediction has NLL 0.69315 and 50% accuracy. This is modest
learning, not reliable logical execution. The final 2,000-update validation
NLLs worsened to 0.69629 and 0.72680 respectively even though training scores
improved. Simply taking the last warmup checkpoint would erase the observed
validation advantage.

A minimal next recipe is bounded direct-core warmup with validation checkpoint
selection, followed by joint training. Learning rate 0.003 at roughly 500
updates is the best NLL candidate here; 0.001 around 1,500 updates gives higher
validation accuracy but slightly worse NLL. This one-seed comparison does not
establish robust superiority. The full experiment should preserve a separate
warmup sampling RNG and retain the best validation state rather than the last.

The earlier 1,312-parameter diagnostic is preserved separately in the v2
experiment. RMS normalization, training-set standardization, and a simple
depth curriculum did not improve that probe's best validation NLL. Those
normalization variants have not been promoted into this architecture.

`scaled_core_probes.json` contains all measured checkpoints.
`probe_scaled_core.py` reproduces the CPU-only comparison.
`core235210_seed7_lr*_best.pt` contain diagnostic core-only states; they are
not full-model checkpoints or evidence of joint control competence.
