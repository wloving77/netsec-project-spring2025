import argparse
import pandas as pd
import os
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.gan.models import NetworkAnomalyDetector, DeeperNetworkAnomalyDetector
from src.utils.utils import train_and_evaluate_model

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
    X_real, y_real = load_split_data(args.x_real, args.y_real)
    X_synth, y_synth = load_combined_data(args.synthetic)
    X_test, y_test = load_split_data(args.x_test, args.y_test)

    # Combine real and synthetic training data
    X_train = pd.concat([X_real, X_synth], ignore_index=True)
    y_train = pd.concat([y_real, y_synth], ignore_index=True)
    print(f"Classes: {y_train.unique()}")
    # Encode string labels into integers automatically
    le_target = LabelEncoder()
    y_train_enc = le_target.fit_transform(y_train)
    y_test_enc = le_target.transform(y_test)

    # Choose model
    input_dim = X_train.shape[1]
    if args.model == "simple":
        model_fn = NetworkAnomalyDetector
    elif args.model == "deeper":
        model_fn = DeeperNetworkAnomalyDetector
    else:
        raise ValueError("Invalid model type: choose 'simple' or 'deeper'.")

    # Train and evaluate
    model = train_and_evaluate_model(
        X_train=X_train,
        y_train=y_train_enc,
        X_test=X_test,
        y_test=y_test_enc,
        le_target=le_target,
        model=model_fn,
        epochs=60,
        batch_size=64,
        lr=0.001,
    )

    # Save model
    torch.save(model.state_dict(), args.model_out)

def run_ids_training(x_real, y_real, synthetic_path, x_test, y_test, model_out, results_dir, model="simple"):
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
    parser.add_argument("--x_real", required=True)
    parser.add_argument("--y_real", required=True)
    parser.add_argument("--synthetic", required=True)
    parser.add_argument("--x_test", required=True)
    parser.add_argument("--y_test", required=True)
    parser.add_argument("--model_out", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--model", choices=["simple", "deeper"], default="simple", help="Model type to use for training")
    args = parser.parse_args()
    train_ids(args)
