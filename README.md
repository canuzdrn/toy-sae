# toy-sae

Toy experiments for studying split sparse autoencoders on foreground-colored
MNIST embeddings.

The project asks whether a learned representation can be split into:

- `z_good`: semantic/content information, ideally digit or digit-group
  information without color.
- `z_bad`: nuisance/domain information, especially the red/green foreground
  color.

The short version of the current result is:

```text
The model achieves partial disentanglement, not clean disentanglement.

z_bad becomes a clean color channel.
z_good still carries color, often in digit-conditioned ways.
The in-training adversarial head can be confused without removing
post-hoc probe-accessible color information.
```

This README is the top-level map. Folder-level README files give more local
detail for `scripts/`, `src/`, `data/`, `checkpoints/`, `outputs/`, and
`figures/`.

## Environment

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate toy-sae
```

The environment includes PyTorch, torchvision, scikit-learn, matplotlib,
Jupyter, and `umap-learn`. Most scripts automatically use CUDA if available,
then Apple MPS, then CPU.

Run commands from the repository root:

```bash
cd /path/to/toy-sae
```

## Repository Layout

```text
configs/       JSON configs for single runs and sweeps.
data/          Generated ColoredMNIST files and frozen base-AE embeddings.
checkpoints/   Trained model checkpoints, configs, histories, and plots.
figures/       Architecture diagrams and report figures.
notebooks/     Exploratory notebooks.
outputs/       Post-hoc probe JSON files and diagnostic plots.
scripts/       Executable training, probing, debugging, and sweep scripts.
src/toy_sae/   Reusable datasets, models, and utilities.
```

The `scripts/` folder is organized by task:

```text
scripts/train/   Training entry points.
scripts/probe/   Quantitative post-hoc probes.
scripts/debug/   Qualitative/debug visualizations.
scripts/sweep/   Sequential sweep runners.
```

Two scripts intentionally remain directly under `scripts/` because they are
pipeline setup steps:

```text
scripts/generate_colored_mnist.py
scripts/extract_base_embeddings.py
```

## Pipeline Overview

The full experiment pipeline is:

```text
1. Generate ColoredMNIST splits.
2. Train a base convolutional autoencoder on balanced colored digits.
3. Extract frozen base-AE embeddings for every split.
4. Train split SAEs on biased base-AE embeddings.
5. Probe frozen z_good and z_bad latents post hoc.
6. Run diagnostic tests for color leakage.
```

The split-SAE models do not train on images directly. They train on frozen
64-dimensional embeddings produced by the base autoencoder.

## Data Splits

The data-generation step creates RGB MNIST digits with black background and
red/green foreground strokes.

```bash
python scripts/generate_colored_mnist.py
```

Generated files live under `data/colored_mnist/`:

```text
ae_train_balanced.npz       Balanced colors for base-AE training.
ae_val_balanced.npz         Balanced validation split for base-AE tuning.
split_train_biased.npz      Biased split-SAE training split.
split_val_biased.npz        Biased split-SAE validation split.
test_id_biased.npz          Test split with the same shortcut as training.
test_balanced.npz           Test split with balanced colors per digit.
test_reversed.npz           Test split with the shortcut reversed.
```

Each `.npz` file contains:

```text
images        RGB tensors in NCHW layout, values in [0, 1].
digits        MNIST labels 0-9.
colors        0=red, 1=green.
digit_groups  0 for digits 0-4, 1 for digits 5-9.
metadata      JSON metadata for provenance.
```

The biased split is intentionally shortcut-heavy. It is useful for training
the split SAE, but it is not enough to evaluate invariance. For that, the
important splits are usually `test_balanced` and `test_reversed`.

## Base Autoencoder

Train the base convolutional autoencoder:

```bash
python scripts/train/train_base_autoencoder.py
```

Default inputs:

```text
data/colored_mnist/ae_train_balanced.npz
data/colored_mnist/ae_val_balanced.npz
```

Default outputs:

```text
checkpoints/base_ae/best.pt
checkpoints/base_ae/latest.pt
checkpoints/base_ae/config.json
checkpoints/base_ae/history.json
checkpoints/base_ae/reconstructions.png
```

The base AE is a small convolutional autoencoder defined in
`src/toy_sae/models/base_autoencoder.py`. Its encoder produces the frozen
embedding `x` used by the split-SAE experiments. The default embedding
dimension is `64`.

After training, extract embeddings:

```bash
python scripts/extract_base_embeddings.py
```

This reads `checkpoints/base_ae/best.pt` and writes one embedding `.npz` file
per split under:

```text
data/base_ae_embeddings/
```

To sanity-check the base embedding, run:

```bash
python scripts/probe/probe_base_embeddings.py \
  --out-dir outputs/base_ae_probes_balanced_train \
  --train-split ae_train_balanced
```

and optionally:

```bash
python scripts/probe/probe_base_embeddings.py \
  --out-dir outputs/base_ae_probes_biased_train \
  --train-split split_train_biased
```

These probes ask whether the frozen base embedding contains digit, digit-group,
and color information before any split-SAE training.

## Split-SAE Architectures

There are two active split-SAE architectures.

### Shared-Encoder Split SAE

Defined in:

```text
src/toy_sae/models/split_sae.py
```

Class:

```text
SplitSparseAutoencoder
```

Architecture:

```text
x -> shared encoder -> h
                    -> z_good -> good decoder -> good_reconstruction
                    -> z_bad  -> bad decoder  -> bad_reconstruction

reconstruction = good_reconstruction + bad_reconstruction
```

The old shared `z_good`-only reconstruction variant has been removed from the
active code. The shared model now always uses additive full reconstruction.

Train it with:

```bash
python scripts/train/train_split_sae.py \
  --embedding-dir data/base_ae_embeddings \
  --checkpoint-dir checkpoints/split_sae/example_shared \
  --epochs 100 \
  --batch-size 256 \
  --lr 1e-3 \
  --lambda-recon 1.0 \
  --lambda-good-recon 0.0 \
  --lambda-badcon 0.1 \
  --lambda-adv 1.0 \
  --grl-lambda 1.0 \
  --lambda-dom 1.0 \
  --lambda-sparse-good 1e-3 \
  --lambda-sparse-bad 5e-3 \
  --deterministic
```

The shared architecture is useful historically, but it has a leakage path:
both latent branches are computed from the same hidden state `h`.

### Separate-Encoder Split SAE

Defined in:

```text
src/toy_sae/models/separate_split_sae.py
```

Class:

```text
SeparateEncoderSAE
```

Architecture:

```text
x -> good_encoder -> z_good -> good_decoder -> good_reconstruction
x -> bad_encoder  -> z_bad  -> bad_decoder  -> bad_reconstruction

reconstruction = good_reconstruction + bad_reconstruction
```

This is the main architecture used in the current diagnosis work. It removes
the shared hidden state and lets the good and bad branches encode the input
independently.

The strongest baseline setting so far is:

```bash
python scripts/train/train_separate_split_sae.py \
  --embedding-dir data/base_ae_embeddings \
  --checkpoint-dir checkpoints/separate_split_sae/sweep_default_adv5_grl5_badcon0_dom1_lr3e4_seed1 \
  --epochs 100 \
  --batch-size 256 \
  --lr 3e-4 \
  --lambda-recon 1.0 \
  --lambda-good-recon 0.0 \
  --lambda-badcon 0.0 \
  --lambda-adv 5.0 \
  --grl-lambda 5.0 \
  --lambda-dom 1.0 \
  --lambda-sparse-good 0.0 \
  --lambda-sparse-bad 0.0 \
  --seed 1 \
  --deterministic
```

Checkpoint outputs include:

```text
best_recon.pt          Best validation reconstruction MSE.
best_total.pt          Best total validation loss.
latest.pt              Final checkpoint.
config.json            Training arguments.
embedding_scaler.npz   Mean/std fitted on the train embedding split.
history.json           Per-epoch train/validation metrics.
plots/                 Automatically generated metric curves.
```

## Loss Terms

The split-SAE training scripts log and optionally optimize these terms:

```text
recon_mse          Full reconstruction MSE for good_rec + bad_rec.
good_recon_mse     Reconstruction MSE using only the good branch.
badcon_loss        Mean squared magnitude of the bad reconstruction.
good_color_loss    Color CE loss from z_good through gradient reversal.
bad_color_loss     Normal color CE loss from z_bad.
good_sparsity      Mean absolute z_good activation.
bad_sparsity       Mean absolute z_bad activation.
```

The main objective is controlled by lambdas:

```text
lambda_recon          weight on full reconstruction.
lambda_good_recon     optional weight on good-only reconstruction.
lambda_badcon         optional penalty on bad reconstruction magnitude.
lambda_adv            weight on adversarial good-color loss.
grl_lambda            gradient reversal strength into the encoder.
lambda_dom            weight on z_bad color/domain classification.
lambda_sparse_good    sparsity penalty for z_good.
lambda_sparse_bad     sparsity penalty for z_bad.
```

Important distinction:

```text
lambda_adv affects the loss weight and trains the color head more strongly.
grl_lambda scales the reversed gradient that reaches the encoder.
```

A metric can appear in `history.json` even if its lambda is zero. For example,
`badcon_loss` can be logged even when `lambda_badcon=0`; it just is not
contributing to the optimized objective.

## Checkpoint Selection

The normal training scripts save `best_recon.pt`, `best_total.pt`, and
`latest.pt`. In practice, `best_recon.pt` is usually the most useful default
for downstream probing because `best_total.pt` and `latest.pt` can be noisy in
adversarial runs.

There is also an experimental checkpointing script:

```text
scripts/train/train_separate_split_sae_posthoc_invariance.py
```

It periodically trains a small fresh linear color probe during training and
saves post-hoc-selected checkpoints such as:

```text
best_linear_posthoc_invariance.pt
best_linear_posthoc_tradeoff.pt
```

This script is useful for studying checkpoint selection, but it does not solve
the underlying representation problem by itself. It can only choose among the
representations that the training objective already produced.

## Evaluation Philosophy

The core lesson of the experiments is that the in-training adversarial color
head is not a reliable proof of invariance.

Bad but common situation:

```text
val_good_color_acc ~= 0.5
post-hoc z_good color accuracy is still high
```

This means the model confused the particular adversarial head trained with the
SAE, but did not remove color information from `z_good`.

The stronger test is:

```text
Freeze the trained SAE.
Extract z_good and z_bad.
Train fresh post-hoc probes from scratch.
Ask what information is recoverable.
```

The desired pattern is:

```text
z_good digit/group accuracy: high enough to preserve semantics
z_good color accuracy: near chance
z_bad color accuracy: high
```

What we usually observe instead:

```text
z_bad color accuracy is near perfect.
z_good still leaks color.
Some leakage is local/digit-conditioned rather than one clean global axis.
```

## Standard Post-Hoc Probes

For the separate-encoder model:

```bash
python scripts/probe/probe_separate_split_sae_latents.py \
  --checkpoint checkpoints/separate_split_sae/sweep_default_adv5_grl5_badcon0_dom1_lr3e4_seed1/best_recon.pt \
  --out-dir outputs/separate_split_sae_latent_probes/sweep_default_adv5_grl5_badcon0_dom1_lr3e4_seed1_best_recon
```

For the shared-encoder model:

```bash
python scripts/probe/probe_split_sae_latents.py \
  --checkpoint checkpoints/split_sae/residual_nosparse_grl5_badcon01_adv5/best_recon.pt \
  --out-dir outputs/split_sae_latent_probes/residual_nosparse_grl5_badcon01_adv5_best_recon
```

These scripts train fresh logistic regression probes for:

```text
z_good -> digit
z_good -> digit_group
z_good -> color
z_bad  -> digit
z_bad  -> digit_group
z_bad  -> color
```

The most important fields in a probe result JSON are:

```text
results.<split>.probe_accuracy.z_good.colors
results.<split>.probe_accuracy.z_good.digits
results.<split>.probe_accuracy.z_good.digit_groups
results.<split>.probe_accuracy.z_bad.colors
```

`test_balanced` is usually the cleanest color-leakage split. `test_reversed`
checks robustness under shortcut reversal.

## Diagnostic Scripts

The diagnosis scripts are as important as the basic probes. They helped reveal
that leakage is not just a single global color axis.

### Latent Color Separability Visualization

```bash
python scripts/debug/visualize_latent_color_separability.py \
  --checkpoint-dir checkpoints/separate_split_sae/sweep_default_adv5_grl5_badcon0_dom1_lr3e4_seed1 \
  --checkpoint-name best_recon
```

This extracts `z_good` and `z_bad`, then saves PCA, t-SNE, and UMAP plots under:

```text
outputs/latent_color_separability/<checkpoint-folder>/<method>/
```

For each method it plots both 2D and 3D projections, colored by true color or
digit group. PCA plots also report the variance captured by the displayed
principal components.

Interpretation:

```text
Clear red/green separation in z_good: visible color leakage.
Mixed red/green points in z_good: weaker global leakage, but not proof of invariance.
Within-digit plots are more diagnostic than global plots.
```

### Subgroup Color Probes

```bash
python scripts/probe/probe_latent_color_subgroups.py \
  --checkpoint-dir checkpoints/separate_split_sae/sweep_default_adv5_grl5_badcon0_dom1_lr3e4_seed1 \
  --checkpoint-name best_recon
```

This trains color probes:

```text
globally
within digit_group = 0
within digit_group = 1
within each digit 0-9
```

It also computes:

- per-digit color-probe direction cosine similarities,
- cross-digit probe transfer matrices,
- same-group vs cross-group transfer summaries,
- a permuted-within-digit label control.

The key question is:

```text
If we hold digit or digit group fixed, can color still be predicted from z_good?
```

If yes, color leakage is not only caused by the biased digit/group-color
correlation. It is directly present inside the representation.

The permuted-within-digit control is a sanity check. With real labels, subgroup
color probes should be high if leakage is real. With labels permuted inside
each digit during probe training, within-digit performance should drop toward
chance.

### Residualized Color Probes

```bash
python scripts/probe/probe_residualized_color.py \
  --checkpoint-dir checkpoints/separate_split_sae/sweep_default_adv5_grl5_badcon0_dom1_lr3e4_seed1 \
  --checkpoint-name best_recon
```

This asks:

```text
Is color recoverable from z_good only because z_good contains digit/group
information, or does z_good contain color information beyond digit/group?
```

Procedure:

```text
1. Fit a linear digit-group or digit classifier on z_good.
2. Remove the row-space of that classifier from z_good.
3. Train fresh probes on the residualized z_good.
4. Check whether color is still recoverable.
```

If color remains predictable after removing digit or digit-group linear
subspaces, that is strong evidence of direct residual color leakage.

## Sweeps

Sequentially train several separate-encoder configurations:

```bash
python scripts/sweep/run_separate_split_sae_sweep.py \
  --config configs/separate_split_sae_adv5_grl5_lr3e4_dom1_badcon0_sweep.json
```

The sweep runner:

- reads a JSON list of configs,
- trains each config one after another,
- writes checkpoints under `checkpoints/separate_split_sae/<config-name>/`,
- writes `train.log`,
- skips completed runs unless `--overwrite` is passed.

Batch-probe separate-encoder runs:

```bash
python scripts/sweep/run_separate_split_sae_probe_sweep.py \
  --checkpoint-root checkpoints/separate_split_sae \
  --checkpoints best_recon
```

The probe sweep skips runs whose `probe_results.json` already exists unless
`--overwrite` is passed.

## Current Retained Checkpoints

The repository is currently organized around a few retained checkpoints:

```text
checkpoints/base_ae/
checkpoints/split_sae/residual_nosparse_grl5_badcon01_adv5/
checkpoints/separate_split_sae/sweep_default_adv5_grl5_badcon0_dom1_lr3e4_seed0/
checkpoints/separate_split_sae/sweep_default_adv5_grl5_badcon0_dom1_lr3e4_seed1/
```

There may be additional generated outputs from previous sweeps under
`outputs/`. Those are useful for comparison, but the main current diagnosis
focuses on the separate-encoder `adv5/grl5/badcon0/dom1/lr3e-4` setting,
especially seed 1 as a more representative leakage case.

## What We Learned

The experiments moved through several stages.

### 1. Weak adversarial pressure failed immediately

Early runs had:

```text
val_good_color_acc ~= 1.0
```

The in-training adversarial color head could predict color from `z_good`
almost perfectly. This clearly failed the goal.

### 2. Stronger adversarial pressure confused the training head

Increasing `lambda_adv`, `grl_lambda`, and tuning learning rate could make:

```text
val_good_color_acc ~= 0.5
```

This looked promising, but it was not enough.

### 3. Post-hoc probes revealed remaining leakage

Fresh probes trained after the SAE was frozen often recovered color from
`z_good` with high accuracy, even when the in-training adversarial head was
near chance.

This is the central failure mode:

```text
Training-time adversary confusion != representation-level invariance.
```

### 4. Separate encoders helped but did not solve it

The separate-encoder architecture made `z_bad` a very clean color channel, but
`z_good` still retained color.

The remaining leakage is often conditional:

```text
Globally, z_good may not show one obvious color axis.
Within fixed digits, color can still be highly predictable.
```

### 5. Diagnostics refined the story

Subgroup probes, cross-digit transfer, residualized probes, and label
permutation controls support this interpretation:

```text
The model routes a clean global color representation into z_bad.
z_good no longer always has an obvious single global color axis.
But z_good still contains strong color information inside fixed digits.
Therefore the failure is residual conditional leakage, not total failure.
```

This is a useful scientific result. The method is doing something real, but it
has not achieved clean disentanglement.

## Suggested Reading Order

If you are coming to the project cold, read in this order:

```text
1. README.md
2. scripts/README.md
3. src/toy_sae/models/README.md
4. src/toy_sae/models/separate_split_sae.py
5. scripts/train/train_separate_split_sae.py
6. scripts/probe/probe_separate_split_sae_latents.py
7. scripts/probe/probe_latent_color_subgroups.py
8. scripts/probe/probe_residualized_color.py
9. outputs/README.md
```

For the data story, also open:

```text
notebooks/dataset_overview.ipynb
```

For diagrams, see:

```text
figures/conv_ae.png
figures/residual_split_sae_1.png
figures/separate_split_sae.png
```

## Practical Notes

- Always run training and probing commands from the project root.
- Check `history.json` and `plots/` for training dynamics.
- Use post-hoc probes, not only training-head accuracy, to judge invariance.
- Prefer `test_balanced` and subgroup probes for color-leakage claims.
- Treat visualizations as debugging evidence, not final metrics.
- Keep checkpoint-specific scalers with their checkpoints. The probe scripts
  load `embedding_scaler.npz` to evaluate in the same standardized space.
- Some older output names may contain historical labels such as `residual`.
  The current shared model class is simply `SplitSparseAutoencoder`.

## Reference

The colored shortcut setup is inspired by the ColoredMNIST experiment from the
Invariant Risk Minimization repository:

https://github.com/facebookresearch/InvariantRiskMinimization/blob/main/code/colored_mnist/main.py
