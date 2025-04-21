# Network Security & Privacy Project Spring 2025

## Group Members:

- William Loving (wfl9zy)
- Tripp Mims (cef7dn)
- Joseph Okeno-Storms (djm7rd)
- Anders Gyllenhoff (ujx4ab)

# Project Info:

## GAN-Enhanced Intrusion Detection System (IDS)

This project investigates how **Generative Adversarial Networks (GANs)** can be used to generate synthetic network attack traffic to improve the performance of an Intrusion Detection System (IDS) — especially on rare or underrepresented attacks.

We use datasets like **CICIDS2017** and **UNSW-NB15**, apply **CTGAN** or **TVAE** to generate synthetic attack samples, and train IDS models on both real and synthetic data. Our goal is to show that augmenting training with synthetic data improves detection of hard-to-spot threats.

---

## Instructions

Either use the setup.sh to create a conda enviroment or create you own virtial enviroment and use the requirements.txt under gans_ids_project. 

While in the root directory install the project module in editable mode with "pip install e .". This will permit you to use src modules in scripts in other parts of the repo. 

---

## 📁 Project Directory Structure

```bash
gan_ids_project/
├── data/                       # Datasets
│   ├── raw/                    # Original datasets
│   ├── processed/              # Cleaned + normalized data
│   └── synthetic/              # GAN-generated traffic samples
│
├── src/                        # Core scripts
│   ├── data_augmentors         # Classes for synthetic data prodiction
│   ├── models.py               # Model for binary (attack, not attack) OR
│                               # multi class (type of attack, not attack)
│   └── utils.py                # helpers: data loading, making synethic 
│                               # call synth data classes, plotting, model
│                               # training and evaluation
│
├── experiments/                # Outputs & model artifacts
│   ├── basic_test.py           # Currently all of the test results
│
│
├── .gitignore
├── LICENSE
├── README.md
├── setup.py                    # To download project module 
├──                             # (pip install -e .)
└── setup.sh                    # Enviroment setup
```