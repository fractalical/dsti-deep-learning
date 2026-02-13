# Experiment Design — Deep Learning Project (2026)
**Task:** NLP — Text Classification (News Topic Classification)  
**Team:** Charles Effah, Mohamed Abdelkarim, Othmane Khettar, Artur Zakirov  
**Date:** 2026-02-13

## 1) Problem Statement
We address supervised **multi-class text classification**: given a short news text, predict its topic label among 4 classes.

## 2) Dataset (frozen choice)
**Dataset:** AG News (Hugging Face Datasets: `ag_news`)  
- **Splits:** train = 120,000 samples; test = 7,600 samples. :contentReference[oaicite:0]{index=0}  
- **Labels (4 classes):** World (0), Sports (1), Business (2), Sci/Tech (3). :contentReference[oaicite:1]{index=1}  
- **Fields:** `text` (string), `label` (int). :contentReference[oaicite:2]{index=2}

### Rationale
AG News is a standard benchmark for topic classification: large enough for transformer fine-tuning, simple multi-class setup, and suitable for clear error analysis.

## 3) Data Split Strategy (fixed)
We keep the **official dataset test split** untouched for final evaluation. :contentReference[oaicite:3]{index=3}  
From the original training split (120k), we create:
- **train:** 90%
- **validation:** 10% (stratified by label)

**Random seed:** 42 (fixed across codebase).  
**Note:** test split is used **once** at the end, after model selection.

## 4) Baseline + Improvements (explicit)
### Baseline (must be simple)
**Baseline:** TF-IDF + Logistic Regression (scikit-learn)  
Purpose: provide a strong non-neural reference and quantify the gain from transformer fine-tuning.

### Improvement #1 (efficient transformer)
**Model:** `distilbert-base-uncased` fine-tuned for sequence classification  
Purpose: speed/efficiency baseline for transformers.

### Improvement #2 (strong transformer)
**Model:** `roberta-base` fine-tuned for sequence classification  
Purpose: maximize accuracy and macro-F1 on AG News; expected to outperform DistilBERT.

**Reference target:** a public fine-tuned RoBERTa checkpoint on AG News reports **~0.94697 accuracy** on evaluation. :contentReference[oaicite:4]{index=4}  
We aim to match or approach this within our compute constraints.

*(Optional, only if time/compute allows: DeBERTa-v3-base fine-tuning as an extra strong model.)*

## 5) Metrics (fixed)
### Primary metric
- **Accuracy** (overall correctness). Accuracy is a standard metric and is widely used for classification. :contentReference[oaicite:5]{index=5}

### Secondary metric
- **Macro-F1** (unweighted mean of per-class F1)  
Rationale: macro averaging treats all classes equally and helps detect if a model performs poorly on a specific class.

### Diagnostics (reporting/analysis)
- Confusion Matrix
- Per-class Precision/Recall/F1

## 6) Training & Model Selection Protocol
- Train on **train**, tune hyperparameters and early stopping on **validation**.
- Choose the best checkpoint by **validation macro-F1** (tie-breaker: validation accuracy).
- Report final numbers on **test** once per model.
- Log all runs with config snapshots (YAML) and fixed seeds.

## 7) Research Hypotheses
H1. Fine-tuned transformer models (DistilBERT/RoBERTa) will outperform the classical baseline in both Accuracy and Macro-F1.  
H2. RoBERTa will outperform DistilBERT due to stronger pretraining and capacity, at the cost of compute.  
H3. Most confusions will occur between **Business** and **Sci/Tech** due to overlapping vocabulary and topics.

## 8) Acceptance Criteria (success targets)
- Baseline (TF-IDF + LR): reasonable performance and stable training (used as reference).
- DistilBERT fine-tuned: clear improvement over baseline.
- RoBERTa fine-tuned: best overall; target to approach **~0.95 accuracy** (reference ~0.94697). :contentReference[oaicite:6]{index=6}
- Reproducibility: a clean run from README reproduces metrics within small tolerance.

## 9) Deliverables produced from this design
- Reproducible code + configs for:
  - Data loading + EDA
  - Preprocessing/tokenization
  - Training (baseline + two improvements)
  - Unified evaluation script + plots/tables for the report
