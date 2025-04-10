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



## 👥 Team Member CLI Responsibilities

This section breaks down each group member’s CLI tasks, inputs, and outputs. Follow your section to complete your part of the pipeline — when all jobs are done, the project is ready for analysis and presentation.

---

### Member 1 – Data Engineer  
**Goal:** Preprocess the raw dataset and generate clean training/test splits.

**Command:**
```bash
python src/preprocessing/preprocess.py \
    --input data/raw/CICIDS2017.csv \
    --output_dir data/processed/ \
    --attack_types 'DoS,GoldenEye,Heartbleed' \
    --train_split 0.8
```

**Inputs:**
- data/raw/CICIDS2017.csv — original dataset  
- --attack_types — rare attack types to retain  
- --train_split — ratio for splitting training/testing

**Outputs:**
- data/processed/train.csv  
- data/processed/test.csv  
- data/processed/columns.json

✅ Produces the cleaned dataset for GAN and IDS training.

---

### Member 2 – GAN Developer  
**Goal:** Train a GAN to generate synthetic samples for a selected rare attack.

**Command:**
```bash
python src/gan/train_gan.py \
    --input data/processed/train.csv \
    --output data/synthetic/generated.csv \
    --attack_type 'Heartbleed' \
    --model_out experiments/gan_models/heartbleed_ctgan.pkl \
    --samples 5000
```

**Inputs:**
- data/processed/train.csv — preprocessed data  
- --attack_type — rare class to model  
- --samples — how many synthetic rows to generate

**Outputs:**
- data/synthetic/generated.csv — synthetic samples  
- experiments/gan_models/heartbleed_ctgan.pkl — trained GAN model

✅ Provides augmented data to improve IDS training.

---

### Member 3 – IDS Modeler  
**Goal:** Train the IDS model on real + synthetic data and evaluate its performance.

**Command:**
```bash
python src/ids/train_ids.py \
    --real data/processed/train.csv \
    --synthetic data/synthetic/generated.csv \
    --test data/processed/test.csv \
    --model_out experiments/ids_models/random_forest.pkl \
    --results_dir experiments/results/
```

**Inputs:**
- data/processed/train.csv — real training data  
- data/synthetic/generated.csv — GAN-generated data  
- data/processed/test.csv — test split

**Outputs:**
- experiments/ids_models/random_forest.pkl — trained model  
- experiments/results/metrics.json — performance metrics  
- experiments/results/confusion_matrix.png — evaluation chart

✅ Shows how much GAN augmentation improves threat detection.

---

### Member 4 – Evaluator & Integrator  
**Goal:** Analyze the quality of generated data and visualize model results.

**Command:**
```bash
python src/utils/plot_utils.py \
    --real data/processed/train.csv \
    --synthetic data/synthetic/generated.csv \
    --metrics experiments/results/metrics.json \
    --out_dir experiments/results/
```

**Inputs:**
- data/processed/train.csv — baseline reference  
- data/synthetic/generated.csv — synthetic data  
- experiments/results/metrics.json — IDS evaluation results

**Outputs:**
- experiments/results/tsne_distribution.png — real vs. synthetic comparison  
- experiments/results/gan_vs_real_stats.json — stats report  
- experiments/results/summary_report.md — for presentation use

✅ Delivers final visualizations and analysis to report and present results.

---