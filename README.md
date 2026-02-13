# dsti-deep-learning

# 

# Task

# Text classification on AG News. Baseline plus two transformer improvements. Metrics: Accuracy and Macro-F1.

# 

# Setup (Windows PowerShell)

# 

# Create and activate environment

# py -m venv venv

# .\\venv\\Scripts\\activate

# 

# Install dependencies

# pip install -r requirements.txt

# 

# Config

# All key parameters live in configs\\config.yaml.

# 

# Project structure

# docs: experiment scope and design

# notebooks: sanity checks and EDA

# src: reusable code (data, training, evaluation)

# runs: saved models, predictions, logs

# report: figures and tables for the PDF

# 

# Run order

# 

# Data sanity and EDA

# Run notebooks in order:

# notebooks\\01\_data\_load\_sanity.ipynb

# notebooks\\02\_eda.ipynb

# 

# Train baseline

# Outputs saved to runs\\baseline\_\*

# 

# Evaluate baseline

# Outputs saved to report\\figures and report\\tables

# 

# Train improvements

# DistilBERT outputs to runs\\distilbert\_\*

# RoBERTa or DeBERTa outputs to runs\\roberta\_\* or runs\\deberta\_\*

# 

# Final reproducibility check

# Follow the README from a clean environment and confirm metrics match.

