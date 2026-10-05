# Model Card

## RadEncoder-XL

RadEncoder is a denoising autoencoder for single-cell gene expression. ConditionTransport applies a condition-dependent residual transformation to its latent vectors.

| Setting | Value |
|---|---:|
| Input genes | 14,817 |
| Hidden dimension | 6,144 |
| Encoder blocks | 10 |
| Latent dimension | 1,536 |
| Parameters | 2,164,362,147 |

## Inputs and outputs

Counts are normalized to 10,000 per cell and transformed with `log1p`. Input columns follow [vocabulary.tsv](../reference/vocabulary.tsv).

ConditionTransport uses dose in Gy, time after irradiation in hours, fraction number and a radiation-type code. The `condition_vector` function in [radfusion_gpu_common.py](../src/radfusion_gpu_common.py) converts these inputs to five features.

Inference returns cell embeddings. Gene ordering, radiation codes and encoder/transport checkpoints must match the training configuration.

## Training data

The [reference specification](../evidence/RadFusion_R3_REFERENCE_V1.json) lists 3,113,467 cells from 15 breast/mammary datasets, stored in 163 sparse matrix shards.

The [main training summary](../evidence/encoder_summary.json) records 3,157,795 steps over approximately 60 hours. Holdout evaluation records cover GSE162931, GSE255800 and GSE310219; aggregate benchmarks are not included in this release.

## Reproducibility

- Checkpoint downloads are pending.
- End-to-end inference with trained checkpoints has not been tested for this release.
- Dependency versions are not pinned.
- Training scripts contain environment-specific paths.
- The published pretraining script does not implement the VICReg option used in the main run.

The model is intended for research and has not been validated for clinical use.

## License

Project code: [Apache-2.0](../LICENSE). Source datasets and dependencies retain their respective terms.
