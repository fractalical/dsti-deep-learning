# Modeling Notes (Mohamed Abdelkarim)



## Overview

This section summarizes the modeling and training results for AG News topic classification.



## Models compared

- Baseline: TF-IDF + Logistic Regression

- Improvement #1: DistilBERT

- Improvement #2: RoBERTa



## Validation results

- Baseline: Accuracy 0.9223, Macro-F1 0.9220

- DistilBERT (1 epoch): Accuracy 0.9408, Macro-F1 0.9408

- DistilBERT (2 epochs): Accuracy 0.9467, Macro-F1 0.9467

- RoBERTa (1 epoch): Accuracy 0.9437, Macro-F1 0.9438



## Best model

The best current model is DistilBERT trained for 2 epochs, with validation Accuracy 0.9467 and Macro-F1 0.9467.



## Ablation study

A learning-rate ablation was run on DistilBERT for 1 epoch:

- 1e-5 -> Accuracy 0.9350, Macro-F1 0.9349

- 2e-5 -> Accuracy 0.9408, Macro-F1 0.9408



Takeaway: 2e-5 performed better than 1e-5 on this setup.



## Interpretation

- DistilBERT improved clearly over the TF-IDF baseline.

- RoBERTa was competitive, but under the current setup it did not outperform the best DistilBERT run.

- The learning-rate ablation suggests that 2e-5 is the better choice for this configuration.



## Notes for evaluation

Detailed run artifacts are available under `runs/`, including:

- metrics_val.json

- predictions_val.csv

- config_snapshot.yaml

- log_history.json

- overrides.json for ablation runs

