# /experiments

This directory contains output files from model training and evaluation. It is organized by component to prevent clutter.

## Subdirectories:

- `gan_models/`:  
  Serialized weights or checkpoints for GAN training (e.g., `.pkl`, `.pt`, `.json`). May also include training logs.

- `ids_models/`:  
  Trained intrusion detection models (e.g., `.joblib`, `.pkl`). Optionally include model validation scores or metadata.

- `results/`:  
  Confusion matrices, accuracy plots, ROC curves, class distributions, feature importance diagrams, etc.

> 📌 This folder is NOT version-controlled — it should be added to `.gitignore`.
