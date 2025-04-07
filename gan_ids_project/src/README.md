# /src

This directory contains all Python scripts for the core logic of the project. Code here should be modular, reusable, and version-controlled.

## Structure:

- `preprocessing/`:  
  Scripts for cleaning, encoding, normalizing, and splitting the dataset. Also includes feature selection and correlation analysis.

- `gan/`:  
  Scripts to define, train, and sample from the GAN (e.g., CTGAN). Should include logging and synthetic sample saving.

- `ids/`:  
  Scripts to train and evaluate the intrusion detection model. Supports training on both original and GAN-augmented data.

- `utils/`:  
  Shared helper scripts — includes plotting utilities, data loaders, performance metrics, logging tools, etc.

> 📌 All scripts should follow the structure: `main()` function, argparse, clean outputs. Avoid hardcoding paths.
