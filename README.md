# Privacy-Preserving Facial Expression Recognition

Python project for studying facial expression recognition under simple privacy-preserving transformations.

The project compares classification utility, visual de-identification and ViT attention behaviour on a grayscale RAF-DB style dataset.

## Project Structure

```text
privacy-fer/
|-- data/
|   |-- raw/
|   |-- processed/
|-- notebooks/
|   |-- 01_EDA.ipynb
|   |-- 02_Modeling.ipynb
|   |-- 03_Privacy.ipynb
|   |-- 04_AttentionMaps.ipynb
|-- results/
|   |-- models/
|   |-- tables/
|   |   |-- attention/
|   |   |-- final/
|   |   |-- intermediate/
|   |   |-- validation/
|   |-- plots/
|   |   |-- attention/
|   |   |-- final/
|   |   |-- training/
|-- src/
|   |-- configs.py
|   |-- attention/
|   |   |-- analysis.py
|   |   |-- landmarks.py
|   |   |-- reporting.py
|   |-- data/
|   |   |-- download_dataset.py
|   |   |-- loader.py
|   |-- eda/
|   |   |-- reporting.py
|   |-- modeling/
|   |   |-- reporting.py
|   |   |-- training.py
|   |-- privacy/
|   |   |-- filters.py
|   |   |-- reporting.py
|-- scripts/
|   |-- evaluation/
|   |   |-- evaluate.py
|   |-- reporting/
|   |   |-- make_plots.py
|   |-- training/
|   |   |-- train.py
|   |-- validation/
|   |   |-- check_data_loader.py
|   |   |-- validate_fairness.py
|-- requirements.txt
```

## Official Pipeline

Use these files as the reproducible project pipeline:

- `scripts/training/train.py`: train ResNet18, MobileNetV3-Large, Swin-T or ViT-B/16.
- `scripts/validation/check_data_loader.py`: run a quick sanity check for `RAFDataset`, privacy filters and DataLoader batches.
- `scripts/evaluation/evaluate.py`: evaluate a saved checkpoint on `train`, `val` or `test`.
- `scripts/reporting/make_plots.py`: regenerate the final reporting tables and figures.
- `scripts/validation/validate_fairness.py`: verify split consistency, transformation consistency and validation-first selection evidence.
- `notebooks/01_EDA.ipynb`: inspect raw dataset structure, balance and image properties.
- `notebooks/02_Modeling.ipynb`: inspect plots and optionally launch baseline/fine-tuning runs using `src/modeling/reporting.py`.
- `notebooks/03_Privacy.ipynb`: inspect fixed de-identification experiments using `src/privacy/reporting.py`.
- `notebooks/04_AttentionMaps.ipynb`: qualitative and quantitative ViT attention/landmark analysis.

Generated checkpoints, plots and evaluation scratch files are ignored by Git. The report-ready numeric outputs are kept in `results/tables/final/`.

## Environment Setup

In Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If you use VS Code notebooks, select the `.venv` kernel.

## Dataset Setup

The dataset used in this project is:

```text
dollyprajapati182/balanced-raf-db-dataset-7575-grayscale
```

Download and copy it into `data/raw`:

```powershell
.\.venv\Scripts\python.exe .\src\data\download_dataset.py
```

Expected structure:

```text
data/raw/balanced-raf-db-dataset-7575-grayscale/
|-- train/
|-- val/
|-- test/
```

Run the DataLoader sanity check after downloading the dataset:

```powershell
.\.venv\Scripts\python.exe .\scripts\validation\check_data_loader.py
```

This checks split sizes, class ordering, metadata, privacy filters and one train-ready batch.

## Training

Use `scripts/training/train.py` as the main training entrypoint.

Example clean ResNet18 run:

```powershell
.\.venv\Scripts\python.exe .\scripts\training\train.py `
  --model resnet18 `
  --weights pretrained `
  --privacy-mode none `
  --privacy-intensity 0 `
  --epochs 10 `
  --batch-size 64 `
  --num-workers 4 `
  --pin-memory `
  --persistent-workers `
  --prefetch-factor 2 `
  --run-suffix cnn_baseline
```

Example crop/context removal run:

```powershell
.\.venv\Scripts\python.exe .\scripts\training\train.py `
  --model resnet18 `
  --weights pretrained `
  --privacy-mode crop `
  --privacy-intensity 0.75 `
  --epochs 10 `
  --batch-size 64 `
  --num-workers 4 `
  --pin-memory `
  --persistent-workers `
  --prefetch-factor 2 `
  --run-suffix fixed_deid
```

Quick smoke test:

```powershell
.\.venv\Scripts\python.exe .\scripts\training\train.py `
  --model resnet18 `
  --epochs 1 `
  --batch-size 8 `
  --max-samples-per-split 64 `
  --run-suffix smokecheck
```

Each training run saves:

```text
results/models/<run_name>_best.pt
results/models/<run_name>_metrics.json
results/models/<run_name>_config.json
results/models/<run_name>_classification_report.txt
results/plots/training/<run_name>_metrics.png
```

## Evaluation

Use `scripts/evaluation/evaluate.py` to evaluate an existing checkpoint on `train`, `val` or `test`.

```powershell
.\.venv\Scripts\python.exe .\scripts\evaluation\evaluate.py `
  --checkpoint .\results\models\resnet18_none_0p0_cnn_baseline_best.pt `
  --split test
```

Evaluation outputs:

```text
results/evaluations/<run_name>_<split>_evaluation_metrics.json
results/evaluations/<run_name>_<split>_evaluation_classification_report.txt
results/evaluations/<run_name>_<split>_evaluation_confusion_matrix.csv
results/tables/evaluation_runs.csv
```

These evaluation files are treated as scratch outputs and are ignored by Git. Use `scripts/reporting/make_plots.py` for the final consolidated tables.

## Final Tables And Plots

Use `scripts/reporting/make_plots.py` to regenerate the final CSV tables and final figures from saved metrics.

```powershell
.\.venv\Scripts\python.exe .\scripts\reporting\make_plots.py
```

Generated tables:

```text
results/tables/final/results_summary.csv
results/tables/final/final_model_results.csv
results/tables/final/final_results_by_transformation.csv
results/tables/final/best_run_per_model_and_transformation.csv
```

`results_summary.csv` and `final_model_results.csv` include:

```text
model, accuracy, precision, recall, macro_f1, transformation, privacy_mode, privacy_intensity
```

The transformation groups are:

```text
baseline_clean
crop_context_removal
blur
mosaic
canny
noise
```

Generated plots:

```text
results/plots/final/precision_recall_f1_by_model.png
results/plots/final/confusion_matrix_by_emotion.png
results/plots/final/recall_by_emotion.png
results/plots/final/macro_f1_drop_by_transformation.png
results/plots/final/transformation_examples.png
```

## Fairness Validation

Use `scripts/validation/validate_fairness.py` to check whether comparisons use the same split samples and the same transformation pipeline.

```powershell
.\.venv\Scripts\python.exe .\scripts\validation\validate_fairness.py
```

Generated validation reports:

```text
results/tables/validation/fairness_validation.csv
results/tables/validation/fairness_validation_report.json
results/tables/validation/fairness_validation_report.md
```

The validation checks:

- the same train/val/test samples are used across transformations;
- class distributions are unchanged by privacy filters;
- `results/tables/final/results_summary.csv` has the required reporting columns;
- validation metrics are available for privacy parameter selection;
- test metrics are separated for final reporting.

## Attention And Landmark Analysis

The main attention analysis is in:

```text
notebooks/04_AttentionMaps.ipynb
```

It uses:

```text
src/attention/analysis.py
src/attention/reporting.py
src/attention/landmarks.py
```

Optional MediaPipe Face Landmarker setup:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
mkdir models
```

Place the model file here:

```text
models/face_landmarker.task
```

The attention notebook can generate:

```text
results/tables/attention/vit_attention_predictions.csv
results/tables/attention/vit_attention_selected_examples_class_balanced.csv
results/tables/attention/vit_attention_metrics_class_balanced.csv
results/tables/attention/vit_inverse_attention_masking_class_balanced.csv
results/tables/attention/vit_landmark_region_attention_class_balanced.csv
results/plots/attention/vit_attention_analysis/class_balanced/
results/plots/attention/vit_attention_analysis/class_balanced/inverse_attention_masking/
```

## Reproducibility Notes

- Training uses fixed seeds through `--seed`.
- Each training run saves a separate `*_config.json`.
- Metrics are saved in JSON and consolidated into CSV tables.
- Final plots can be regenerated with `scripts/reporting/make_plots.py`.
- Avoid selecting final hyperparameters from the test set. Use validation results for selection and reserve test results for final reporting.

## Current Main Finding

The current strongest result is that `crop/context removal` preserves classification performance best while still applying a simple de-identification transformation. `Mosaic` is useful as a strong perturbation, but it reduces utility too much to be the main solution.
