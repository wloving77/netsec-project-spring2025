import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import argparse


def split_csv(input_path, train_ratio=0.8):
    input_path = Path(input_path)
    df = pd.read_csv(input_path)

    train_df, test_df = train_test_split(
        df, train_size=train_ratio, random_state=42, shuffle=True
    )

    train_path = input_path.parent / "train.csv"
    test_path = input_path.parent / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train CSV saved to: {train_path}")
    print(f"Test CSV saved to: {test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a CSV into train/test sets.")
    parser.add_argument("input_csv", help="Path to the input CSV file")
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="Train set ratio (default: 0.8)"
    )

    args = parser.parse_args()
    split_csv(args.input_csv, args.train_ratio)
