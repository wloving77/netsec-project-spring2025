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

# Setup Guide:

- Initialize `git-lfs`
    - `apt install git-lfs` or `brew install git-lfs` or it comes with git for windows
    - `git-lfs install`
    - `git-lfs pull` 
- Create a virtual environment:
    - `python3 -m venv .venv`
    - `source .venv/bin/activate` activate it
    - `pip install -r requirements.txt` install dependencies
- Preprocess Data;
    - Navigate to `src/preprocessing/`
    - Run `python3 preprocess_data.py`
- Run Experiments
    - Navigate to `experiments/`
    - Run either of the two notebooks to see current performance #'s

# How to Spin up Dashboard:

After completing the setup guide above, follow the steps below to spin up the IDS dashboard:
- Ensure you have successfully installed all packages in `requirements.txt`
- Ensure you are in the root directory
- Run the commad `streamlit run gan_ids_project/ids_dashboard.py`
- You should now be able to access the dashboard via `http://localhost:8501/`
