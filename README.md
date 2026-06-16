# BBdiffusion-FC

Implementation of a Brownian Bridge diffusion framework for age-conditional longitudinal functional connectivity prediction.

## Structure

```text
configs/
  train.yaml
  eval.yaml
  data/
  model/
  trainer/
  paths/
  experiment/
src/
  models/
    ae.py
    bbdm.py
  utils/
    data_utils.py
  train_ae.py
  train_bridge_subject_cv.py
```

## Main components

- `src/models/ae.py`: GAT-based autoencoder for encoding FC matrices into node-level latent representations.
- `src/models/bbdm.py`: Brownian Bridge diffusion model, bridge schedule, residual prediction training, and DDIM-style bridge sampling.
- `src/utils/data_utils.py`: data loading, subject/session parsing, FC metrics, and visualization utilities.
- `src/train_ae.py`: autoencoder training and latent export entry point.
- `src/train_bridge_subject_cv.py`: subject-level 5-fold bridge training and evaluation.

## Expected input format

The bridge script expects a pair CSV containing at least:

```text
fc_file_early,fc_file_late
```

The age TSV should contain:

```text
Participant, Session, ScanAge
```

Latent files are expected to follow the naming convention:

```text
<fc_file_name>_z.npy
```

## Usage

Autoencoder training:

```bash
python -m src.train_ae \
  --data_dir <FC_DIR> \
  --tsv_file <AGE_TSV> \
  --save_dir <AE_OUTPUT_DIR>
```

Subject-level Brownian Bridge cross-validation:

```bash
python -m src.train_bridge_subject_cv \
  --pairs_csv <PAIR_CSV> \
  --age_tsv <AGE_TSV> \
  --z_dir <LATENT_DIR> \
  --fc_dir <FC_DIR> \
  --ae_checkpoint <AE_CHECKPOINT> \
  --save_dir <OUTPUT_DIR>
```

Optional YAML files under `configs/` provide a lightweight template for organizing paths, data settings, model settings, and training settings.
