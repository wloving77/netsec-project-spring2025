# /notebooks

This directory contains exploratory Jupyter Notebooks for prototyping and testing code prior to script formalization.

## Suggested Notebooks:

- `01_eda_baseline.ipynb`:  
  Initial data exploration — distributions, class balance, missing values, etc.

- `02_gan_training.ipynb`:  
  CTGAN/TVAE training experiments. Visualizes learned distributions, sample quality, and mode diversity.

- `03_ids_training_eval.ipynb`:  
  IDS training on both original and augmented data. Tracks metrics (accuracy, precision, recall, confusion matrix).

> 🛠 Notebooks may be experimental and member-specific. Finalized pipelines should be migrated to the `src/` directory.
