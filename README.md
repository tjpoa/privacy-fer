# Privacy-Preserving Facial Expression Recognition

Base Python project for researching facial expression recognition with privacy-preserving filters applied to the input images.

## Objective

This repository is a starting point for Machine Learning experiments focused on:

- facial expression classification;
- applying privacy transformations before training or inference;
- comparing model utility with visual privacy preservation.

## Current Structure

```text
privacy-fer/
|-- data/
|   |-- raw/
|   |-- processed/
|-- notebooks/
|-- results/
|   |-- plots/
|   |-- checkpoints/
|-- src/
|   |-- privacy_filters.py
|-- download_dataset.py
|-- requirements.txt
|-- .gitignore
```

## Virtual Environment

In Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dependencies

The project currently uses:

- `opencv-python`
- `torch`
- `torchvision`
- `timm`
- `matplotlib`
- `pandas`
- `scikit-learn`
- `kagglehub`

## Dataset

The current dataset is downloaded from Kaggle:

- dataset: `dollyprajapati182/balanced-raf-db-dataset-7575-grayscale`
- script: [`download_dataset.py`](./download_dataset.py)

To download and copy the dataset into `data/raw`:

```powershell
.\.venv\Scripts\Activate.ps1
python download_dataset.py
```

Expected destination:

```text
data/raw/balanced-raf-db-dataset-7575-grayscale/
|-- train/
|-- val/
|-- test/
```

## Base Code

- [`src/privacy_filters.py`](./src/privacy_filters.py) contains the initial placeholders for privacy filters such as blur, canny edges, and diffusion noise.

## Next Steps

- implement the privacy filters;
- create a data loading and preprocessing pipeline;
- define the baseline model for expression classification;
- store metrics, checkpoints, and plots in `results/`.
