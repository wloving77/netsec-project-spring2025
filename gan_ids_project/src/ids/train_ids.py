import argparse
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import joblib

def load_split_data(x_path, y_path):
    X = pd.read_csv(x_path)
    y_df = pd.read_csv(y_path)
    y = y_df["Label"] if "Label" in y_df.columns else y_df.iloc[:, 0]
    return X, y

def load_combined_data(path):
    df = pd.read_csv(path)
    X = df.drop("Attack", axis=1)
    y = df["Attack"]
    return X, y

def train_ids(args):
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # Load real training data from split files
    X_real, y_real = load_split_data(args.x_real, args.y_real)

    # Load synthetic data from combined file
    X_synth, y_synth = load_combined_data(args.synthetic)

    X_train = pd.concat([X_real, X_synth], ignore_index=True)
    y_train = pd.concat([y_real, y_synth], ignore_index=True)

    # Load test data from split files
    X_test, y_test = load_split_data(args.x_test, args.y_test)

    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train_enc)

    # Evaluate
    y_pred = clf.predict(X_test)
    report = classification_report(y_test_enc, y_pred, target_names=le.classes_, output_dict=True)
    cm = confusion_matrix(y_test_enc, y_pred)

    # Save model
    joblib.dump(clf, args.model_out)

    # Save metrics
    metrics_path = os.path.join(args.results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=2)

    # Save confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, "confusion_matrix.png"))

def run_ids_training(x_real, y_real, synthetic_path, x_test, y_test, model_out, results_dir):
    class Args:
        def __init__(self):
            self.x_real = x_real
            self.y_real = y_real
            self.synthetic = synthetic_path
            self.x_test = x_test
            self.y_test = y_test
            self.model_out = model_out
            self.results_dir = results_dir

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
    args = parser.parse_args()
    train_ids(args)