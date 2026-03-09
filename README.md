# dsti-deep-learning

Deep Learning group project for **AG News topic classification**.

## Overview

This project compares a classical baseline with transformer-based improvements for **4-class news classification** on the **AG News** dataset.

### Task
Predict the topic of a news article from the following classes:

- World
- Sports
- Business
- Sci/Tech

### Models
- **Baseline:** TF-IDF + Logistic Regression
- **Improvement #1:** DistilBERT
- **Improvement #2:** RoBERTa

### Metrics
- **Accuracy**
- **Macro-F1**

---

## Repository Structure

```text
configs/      YAML configuration files
data/         Raw, split, and processed data
docs/         Experiment design and modelling notes
notebooks/    Sanity-check and EDA notebooks
report/       Tables and figures for the final PDF
runs/         Saved run artefacts (metrics, predictions, logs)
src/          Reusable source code for data, training, and evaluation
README.md
requirements.txt
```

---

## Configuration

All key parameters live in:

```text
configs/data.yaml
```

This includes:
- dataset and task settings
- text-column construction
- split strategy
- batch size and max sequence length
- training hyperparameters
- output paths

---

## Expected Data Format

The pipeline expects CSV files with the following columns:

- `label`
- `title`
- `description`

Expected paths:

```text
data/raw/train.csv
data/splits/
data/processed/train.csv
data/processed/val.csv
```

---

## Setup (Windows PowerShell)

### 1. Create a virtual environment

```powershell
py -m venv venv
```

### 2. Allow PowerShell activation scripts (one-time setup)

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 3. Activate the virtual environment

```powershell
.\venv\Scripts\Activate.ps1
```

> Note: the execution-policy command does **not** auto-activate the environment.  
> You still need to run `.\venv\Scripts\Activate.ps1` each time you open a new terminal.

### 4. Install dependencies

```powershell
pip install -r requirements.txt
pip install accelerate
```

Optional, for notebooks:

```powershell
pip install notebook
```

---

## Workflow

### 1. Data Sanity Check and EDA

Run the notebooks in order:

```text
notebooks/01_data_load_sanity.ipynb
notebooks/02_eda.ipynb
```

### 2. Prepare Processed Train/Validation Data

Before training, make sure these files exist:

```text
data/processed/train.csv
data/processed/val.csv
```

The split should be frozen and saved under:

```text
data/splits/
```

### 3. Train the Baseline

```powershell
python -m src.training.train_baseline --config configs/data.yaml
```

Outputs are saved to:

```text
runs/baseline_*/
```

### 4. Train DistilBERT

```powershell
python -m src.training.train_transformer distilbert-base-uncased --config configs/data.yaml
```

Outputs are saved to:

```text
runs/distilbert_*/
```

### 5. Train RoBERTa

```powershell
python -m src.training.train_transformer roberta-base --config configs/data.yaml
```

Outputs are saved to:

```text
runs/roberta_*/
```

### 6. Run an Ablation Study

Example: learning-rate ablation for DistilBERT

```powershell
python -m src.training.train_ablation --config configs/data.yaml --model distilbert-base-uncased --param learning_rate --values 1e-5 2e-5 --epochs 1
```

Outputs are saved to:

```text
runs/ablation_*/
```

---

## Run Artefacts

Heavy checkpoints and model weights are intentionally **not committed**.

The repository tracks lightweight artefacts needed for evaluation and reporting, such as:

- `metrics_val.json`
- `predictions_val.csv`
- `config_snapshot.yaml`
- `log_history.json`
- `overrides.json` where applicable

---

## Current Validation Results

| Model | Setting | Accuracy | Macro-F1 |
|------|---------|----------|----------|
| TF-IDF + Logistic Regression | Baseline | 0.9223 | 0.9220 |
| DistilBERT | 1 epoch | 0.9408 | 0.9408 |
| DistilBERT | 2 epochs | **0.9467** | **0.9467** |
| RoBERTa | 1 epoch | 0.9437 | 0.9438 |

### Ablation: DistilBERT Learning Rate

| Learning Rate | Epochs | Accuracy | Macro-F1 |
|--------------|--------|----------|----------|
| 1e-5 | 1 | 0.9350 | 0.9349 |
| 2e-5 | 1 | **0.9408** | **0.9408** |

**Best current model:** DistilBERT trained for 2 epochs.

---

## Evaluation and Report Outputs

Report-ready summaries are stored in:

```text
report/tables/
report/figures/
docs/modeling_notes.md
```

These files support:
- evaluation and plotting
- report drafting
- comparison across runs

---

## Reproducibility Notes

- The train/validation split is frozen and saved to JSON.
- Labels are normalised for transformer compatibility during split generation.
- Training is controlled through `configs/data.yaml`.
- Results should be reproducible from a clean environment using the steps above.

---

## Final Reproducibility Check

Before final submission:

- verify notebooks run cleanly
- verify training commands work from a clean environment
- confirm reported metrics match the saved run artefacts
- ensure report tables and figures match the latest tracked outputs