# BBdiffusion-FC

Implementation of a Brownian Bridge diffusion framework for age-conditional longitudinal functional connectivity prediction.

## Structure

```text
BBdiffusion-FC/
  configs/
    data/
    experiment/
    model/
    trainer/

  src/
    models/
      __init__.py
      ae.py
      bbdm.py

    scripts/

    train/
      __init__.py
      train_ae.py
      train_bridge_subject_cv.py

    utils/
      __init__.py
      data_utils.py
```

## Main components

* `configs/`: configuration templates for data settings, model settings, training settings, and experiment organization.
* `src/models/ae.py`: GAT-based autoencoder for encoding FC matrices into node-level latent representations.
* `src/models/bbdm.py`: Brownian Bridge diffusion model, bridge schedule, residual prediction training, and DDIM-style bridge sampling.
* `src/utils/data_utils.py`: data loading, subject/session parsing, FC evaluation metrics, and visualization utilities.
* `src/train/train_ae.py`: autoencoder training and latent representation export.
* `src/train/train_bridge_subject_cv.py`: subject-level 5-fold Brownian Bridge training and evaluation.
* `src/scripts/`: auxiliary scripts for project-level processing or analysis.

## Expected input format

The autoencoder training script expects FC matrices stored as `.npy` files. Each FC matrix should have shape:

```text
116 x 116
```

The age file should contain the following fields:

```text
Participant, Session, ScanAge
```

The Brownian Bridge training script expects a pair CSV containing at least:

```text
fc_file_early,fc_file_late
```

Latent files are expected to follow the naming convention:

```text
<fc_file_name>_z.npy
```

## Usage

### Autoencoder training

```bash
python -m src.train.train_ae \
  --data_dir <FC_DIR> \
  --tsv_file <AGE_TSV> \
  --save_dir <AE_OUTPUT_DIR>
```

### Subject-level Brownian Bridge cross-validation

```bash
python -m src.train.train_bridge_subject_cv \
  --pairs_csv <PAIR_CSV> \
  --age_tsv <AGE_TSV> \
  --z_dir <LATENT_DIR> \
  --fc_dir <FC_DIR> \
  --ae_checkpoint <AE_CHECKPOINT> \
  --save_dir <OUTPUT_DIR>
```

## Configuration

The `configs/` directory provides lightweight templates for organizing experiment settings:

```text
configs/
  data/        dataset-related settings
  model/       model-related settings
  trainer/     training-related settings
  experiment/  experiment-level settings
```

These files are intended to keep path, model, and training options organized when running different experimental settings.

## Citation

If you use this repository, please cite the corresponding paper.
