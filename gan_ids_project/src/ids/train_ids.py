import argparse
import pandas as pd
import os
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.gan.models import NetworkAnomalyDetector, DeeperNetworkAnomalyDetector, WiderDeeperNetworkAnomalyDetector
from src.utils.utils import train_and_evaluate_model
import random

def load_split_data(x_path, y_path):
    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path)
    if "Attack" in y.columns:
        y = y["Attack"]
    elif "Label" in y.columns:
        y = y["Label"]
    else:
        y = y.iloc[:, 0]
    return X, y.apply(lambda x: str(x).lower())

def load_combined_data(path):
    df = pd.read_csv(path)
    X = df.drop("Attack", axis=1)
    y = df["Attack"].apply(lambda x: str(x).lower())
    return X, y

def train_ids(args):
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)


    # Load data
    if args.synthetic and args.x_real and args.y_real and args.x_test and args.y_test:
        X_real, y_real = load_split_data(args.x_real, args.y_real)
        X_synth, y_synth = load_combined_data(args.synthetic)
        X_test, y_test = load_split_data(args.x_test, args.y_test)

        # Combine real and synthetic training data
        X_train = pd.concat([X_real, X_synth], ignore_index=True)
        y_train = pd.concat([y_real, y_synth], ignore_index=True)
    elif args.x_real and args.y_real and args.x_test and args.y_test:
        X_real, y_real = load_split_data(args.x_real, args.y_real)
        X_test, y_test = load_split_data(args.x_test, args.y_test)

        # Use only real data for training
        X_train = X_real
        y_train = y_real
    elif args.synthetic and args.x_test and args.y_test:
        X_synth, y_synth = load_combined_data(args.synthetic)
        X_test, y_test = load_split_data(args.x_test, args.y_test)

        # Use only synthetic data for training
        X_train = X_synth
        y_train = y_synth
    print(f"Classes: {y_train.unique()}, Test Classes: {y_test.unique()}")
    # Encode string labels into integers automatically
    le_target = LabelEncoder()
    le_target.fit(pd.concat([y_train, y_test], ignore_index=True))  # Fit on both train + test labels
    y_train_enc = le_target.transform(y_train)
    y_test_enc = le_target.transform(y_test)

    # Choose model
    if args.model == "simple":
        model_fn = NetworkAnomalyDetector
    elif args.model == "deeper":
        model_fn = DeeperNetworkAnomalyDetector
    else:
        raise ValueError("Invalid model type: choose 'simple' or 'deeper'.")

    # Train and evaluate
    model, accuracy = train_and_evaluate_model(
        X_train=X_train,
        y_train=y_train_enc,
        X_test=X_test,
        y_test=y_test_enc,
        le_target=le_target,
        model=model_fn,
        epochs=50,
        batch_size=128,
        lr=0.001,
    )

    # Save model
    torch.save(model.state_dict(), args.model_out)

# Best Result
"""
Model: DeeperNetworkAnomalyDetector, Epochs: 30, Batch Size: 128, LR: 0.001
number of classes: 10, Classes: [0 1 2 3 4 5 6 7 8 9]
Loss function: Cross Entropy Loss
Epoch 1/30, Loss: 0.9284
Epoch 5/30, Loss: 0.8256
Epoch 10/30, Loss: 0.7935
Epoch 15/30, Loss: 0.7750
Epoch 20/30, Loss: 0.7653
Epoch 25/30, Loss: 0.7593
Epoch 30/30, Loss: 0.7537

Test Accuracy: 67.77%

"""
def random_search_training(x_real, y_real, synthetic_path, x_test, y_test, results_dir, n_trials=10):
    search_space = {
        "model_fn": [NetworkAnomalyDetector, DeeperNetworkAnomalyDetector, WiderDeeperNetworkAnomalyDetector],
        "epochs": [20, 30, 50, 75, 100],
        "batch_size": [32, 64, 128, 256],
        "lr": [0.003, 0.001, 0.0005, 0.0001, 0.00005],
    }


    # Load data
    X_real, y_real = load_split_data(x_real, y_real)
    X_synth, y_synth = load_combined_data(synthetic_path)
    X_test, y_test = load_split_data(x_test, y_test)

    X_train = pd.concat([X_real, X_synth], ignore_index=True)
    y_train = pd.concat([y_real, y_synth], ignore_index=True)

    le_target = LabelEncoder()
    y_train_enc = le_target.fit_transform(y_train)
    y_test_enc = le_target.transform(y_test)

    results = []

    for trial in range(n_trials):
        model_fn = random.choice(search_space["model_fn"])
        epochs = random.choice(search_space["epochs"])
        batch_size = random.choice(search_space["batch_size"])
        lr = random.choice(search_space["lr"])

        print(f"\n🚀 Trial {trial+1}/{n_trials}:")
        print(f"Model: {model_fn.__name__}, Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr}")

        model, accuracy = train_and_evaluate_model(
            X_train=X_train,
            y_train=y_train_enc,
            X_test=X_test,
            y_test=y_test_enc,
            le_target=le_target,
            model=model_fn,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
        )

        results.append({
            "trial": trial + 1,
            "model": model_fn.__name__,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "accuracy": accuracy,
        })

    return results

def run_ids_training(x_real, y_real, synthetic_path, x_test, y_test, model_out, results_dir, model="simple", synthetic=True, real=True):
    class Args:
        def __init__(self):
            self.x_real = x_real
            self.y_real = y_real
            self.synthetic = synthetic_path
            self.x_test = x_test
            self.y_test = y_test
            self.model_out = model_out
            self.results_dir = results_dir
            self.model = model

    train_ids(Args())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--x_real", required=False)
    parser.add_argument("--y_real", required=False)
    parser.add_argument("--synthetic", required=False)
    parser.add_argument("--x_test", required=False)
    parser.add_argument("--y_test", required=False)
    parser.add_argument("--model_out", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--model", choices=["simple", "deeper", "wide"], default="simple", help="Model type to use for training")
    args = parser.parse_args()
    train_ids(args)
