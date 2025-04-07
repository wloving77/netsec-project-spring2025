# Network Security & Privacy Project Spring 2025

## Group Members:

- William Loving (wfl9zy)
- Tripp Mims (cef7dn)
- Joseph Okeno-Storms (compID)
- Anders Gyllenhoff (compID)

# Project Info:

## GAN-Enhanced Intrusion Detection System (IDS)

This project investigates how **Generative Adversarial Networks (GANs)** can be used to generate synthetic network attack traffic to improve the performance of an Intrusion Detection System (IDS) — especially on rare or underrepresented attacks.

We use datasets like **CICIDS2017** and **UNSW-NB15**, apply **CTGAN** or **TVAE** to generate synthetic attack samples, and train IDS models on both real and synthetic data. Our goal is to show that augmenting training with synthetic data improves detection of hard-to-spot threats.

---

## 📁 Project Directory Structure

```bash
gan_ids_project/
├── data/                      # Datasets
│   ├── raw/                   # Original datasets (CICIDS, etc.)
│   ├── processed/             # Cleaned + normalized data
│   └── synthetic/             # GAN-generated traffic samples
│
├── notebooks/                 # Prototyping & exploration
│   ├── 01_eda_baseline.ipynb
│   ├── 02_gan_training.ipynb
│   └── 03_ids_training_eval.ipynb
│
├── src/                       # All core Python scripts
│   ├── preprocessing/         # Cleaning & feature engineering
│   ├── gan/                   # GAN training, sampling
│   ├── ids/                   # IDS training & evaluation
│   └── utils/                 # Plotting & helpers
│
├── experiments/               # Outputs & model artifacts
│   ├── gan_models/
│   ├── ids_models/
│   └── results/
│
├── .gitignore
├── requirements.txt
├── setup.sh                   # One-click setup for the full environment
└── README.md                  # You're here!
```

