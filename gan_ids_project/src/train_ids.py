import argparse
import os
import pandas as pd
import joblib
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def load_data(real_path, synthetic_path, test_path):
    real_df = pd.read_csv(real_path)
    synthetic_df = pd.read_csv(synthetic_path)
    test_df = pd.read_csv(test_path)
    return real_df, synthetic_df, test_df

def prepare_features(df):
    X = df.drop(columns=["label"])
    y = df["label"]
    return X, y

def main(args):
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # Load and merge data
    real_df, synthetic_df, test_df = load_data(args.real, args.synthetic, args.test)
    train_df = pd.concat([real_df, synthetic_df], ignore_index=True)

    # Prepare features
    X_train, y_train = prepare_features(train_df)
    X_test, y_test = prepare_features(test_df)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, args.model_out)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    with open(os.path.join(args.results_dir, "metrics.json"), "w") as f:
        json.dump(report, f, indent=4)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(args.results_dir, "confusion_matrix.png"))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", required=True)
    parser.add_argument("--synthetic", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--model_out", required=True)
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()

    main(args)
