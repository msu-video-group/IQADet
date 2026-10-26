# IQADet

IQADet is a multi-task adversarial attack detector for no-reference image quality assessment (NR-IQA). It combines visual features with quantile-based representations of IQA scores to jointly predict whether an image has been attacked, the attack type, and the victim IQA metric.

## Installation

Use the provided Dockerfile to set up the environment. To enable GPU access, configure Docker with the NVIDIA Container Toolkit. From the repository root, build the image and start a container with the project and data directories mounted:

```bash
docker build -t iqadet .
docker run --rm -it --gpus all --ipc=host -p 8888:8888 \
  -v "$PWD":/workspace/IQADet -v /absolute/path/to/data:/data \
  -w /workspace/IQADet iqadet bash
```

Replace `/absolute/path/to/data` with your local data directory. Run the following commands inside the container.

## Data

Configure `DATASET_PATH = "/data"` and `MODEL_PATH = "models"` in `common.py`. The dataset loader expects the following directory structure:

| Path under `/data/framework/` | Contents |
| --- | --- |
| `clean-<dataset>/<file>` | Unattacked images |
| `attacked-<dataset>/preset_<n>/<attack>/<metric>/<file>` | Adversarially modified images, organized by strength preset, attack method, and victim IQA metric |
| `quantiles.csv` | Precomputed quantile indices for each image sample |

The CSV must contain the columns `file_name`, `dataset`, `preset`, `attack`, and `metric`, along with one quantile-index column for each IQA metric listed in `METRICS` (excluding `clean`). Quantile indices are integers from 0 to 100 inclusive. For unattacked images, `preset`, `attack`, and `metric` must all be set to `clean`. Dataset identifiers must match the directory suffixes and the `--datasets` argument. Attack and metric identifiers must match the names in `common.py`.

## Experiments

The following example trains the multi-task detector on NIPS 2017, using 85% of the data for training and 15% for validation, then evaluates the selected checkpoint on the full KonIQ-10K dataset.

```bash
python train.py \
  --cls-type meta --model swin_v2_b --pretrained \
  --datasets nips --batch 16 --learning-rate 1e-5 --epochs 42 \
  --balanced-sampler --augment-resize \
  --train-ratio 0.85 --val-ratio 0.15
```

To evaluate, create `models-meta.json` at the repository root. Replace `<run>` with the training timestamp; the directory must contain checkpoints and `lightning_logs/version_0/hparams.yaml`:

```json
{"swin_v2_b": "models/meta/swin_v2_b/pretrained/<run>"}
```

```bash
mkdir -p test/res
PYTHONPATH=. python test/test.py --paths-file models-meta.json --cls-type meta --datasets koniq
```

Evaluation selects the checkpoint with the lowest recorded validation loss and saves outputs under `test/res/`.

For image-only models, use `--cls-type classes --binary` for binary attack detection, `--cls-type classes` for attack-type classification, or `--cls-type metrics` for victim-metric classification. Use the corresponding configuration during evaluation.

## Demo

Download `weights.ckpt` (approximately 340 MB) from this repository's **Releases** page and place it in the repository root, next to `demo.ipynb`.

The demo uses this checkpoint, `samples/quantile_idxs.csv`, and `samples/images/` to plot actual and predicted labels:

```bash
jupyter notebook demo.ipynb --ip=0.0.0.0 --allow-root
```
