# RadFusion

Single-cell gene expression modeling for radiation response.

RadFusion combines **RadEncoder**, a denoising autoencoder for gene expression, with **ConditionTransport**, a model that predicts changes in latent representations from radiation dose, time after irradiation, fraction number and radiation type.

## Installation

Install PyTorch for your CPU or CUDA environment, then install the remaining dependencies:

```sh
pip install -r requirements.txt
```

## Inference

Provide raw counts as a SciPy CSR `.npz` matrix. Columns must match the order in [reference/vocabulary.tsv](reference/vocabulary.tsv).

Extract cell embeddings:

```sh
python src/infer.py \
  --counts matrix.npz \
  --encoder checkpoints/encoder.pt \
  --out embeddings.npy
```

Predict embeddings under an irradiation condition:

```sh
python src/infer.py \
  --counts matrix.npz \
  --encoder checkpoints/encoder.pt \
  --transport checkpoints/transport.pt \
  --dose-gy 2 \
  --time-h 24 \
  --fraction 1 \
  --radiation-code 0 \
  --device cuda \
  --out predicted_embeddings.npy
```

Use a transport checkpoint trained with the supplied encoder. The values above are examples; radiation codes must match the training data. Outputs are NumPy arrays with one latent vector per cell.

**Weights are not yet available for download.** These commands require local checkpoints. See the [checkpoint inventory](docs/weights_manifest.csv).

## Code

| File | Purpose |
|---|---|
| `GPU00_prepare_radiation_npz.py` | Convert radiation datasets to sparse matrix bundles |
| `GPU00_io_benchmark.py` | Benchmark data loading and GPU execution |
| `GPU01_radencoder_pretrain.py` | Pretrain RadEncoder |
| `GPU02_condition_train.py` | Train ConditionTransport |
| `GPU03_eval_holdout.py` | Evaluate a held-out context |
| `infer.py` | Extract or predict cell embeddings |
| `radfusion_gpu_common.py` | Model definitions and shared utilities |

Scripts are in `src/`. Update the default data paths before running the GPU training scripts.

## Model and data

RadEncoder-XL has 2.16 billion parameters and maps 14,817 genes to 1,536-dimensional embeddings. The reference dataset contains 3,113,467 cells from 15 breast/mammary datasets.

See the [model card](docs/MODEL_CARD.md) for architecture, preprocessing and reproducibility notes, and [release status](docs/RELEASE_STATUS.md) for available files.

## License

[Apache-2.0](LICENSE). External datasets and dependencies retain their own licenses.
