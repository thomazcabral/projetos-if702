import argparse
from pathlib import Path

import pandas as pd


def split_no_replacement(df, train_frac, val_frac, test_frac, seed):
    if round(train_frac + val_frac + test_frac, 10) != 1.0:
        raise ValueError("Fractions must sum to 1.0")

    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_total = len(shuffled)
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)
    n_test = n_total - n_train - n_val

    train = shuffled.iloc[:n_train]
    val = shuffled.iloc[n_train : n_train + n_val]
    test = shuffled.iloc[n_train + n_val : n_train + n_val + n_test]

    return train, val, test


def oversample_minority(df, target_col, seed):
    counts = df[target_col].value_counts(dropna=False)
    if len(counts) != 2:
        raise ValueError("Target column must have exactly 2 classes")

    majority_label = counts.idxmax()
    minority_label = counts.idxmin()
    majority = df[df[target_col] == majority_label]
    minority = df[df[target_col] == minority_label]

    if len(minority) == len(majority):
        return df.copy(), majority_label, minority_label

    minority_upsampled = minority.sample(
        n=len(majority), replace=True, random_state=seed
    ).reset_index(drop=True)
    balanced = pd.concat([majority.reset_index(drop=True), minority_upsampled], ignore_index=True)
    return balanced, majority_label, minority_label


def save_split(output_dir, prefix, train, val, test):
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / f"{prefix}_train.csv", index=False)
    val.to_csv(output_dir / f"{prefix}_val.csv", index=False)
    test.to_csv(output_dir / f"{prefix}_test.csv", index=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Balance classes by Churn and split each class 50/25/25."
    )
    parser.add_argument(
        "--input",
        default="customer_churn_telecom_services.csv",
        help="Path to churn CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default="splits",
        help="Directory to write the split CSV files.",
    )
    parser.add_argument(
        "--target",
        default="Churn",
        help="Target column to split classes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    df = pd.read_csv(input_path)

    balanced, majority_label, minority_label = oversample_minority(df, args.target, args.seed)

    class_major = balanced[balanced[args.target] == majority_label]
    class_minor = balanced[balanced[args.target] == minority_label]

    train_major, val_major, test_major = split_no_replacement(
        class_major, 0.5, 0.25, 0.25, args.seed
    )
    train_minor, val_minor, test_minor = split_no_replacement(
        class_minor, 0.5, 0.25, 0.25, args.seed
    )

    save_split(output_dir, f"class_{majority_label}_major", train_major, val_major, test_major)
    save_split(output_dir, f"class_{minority_label}_minor", train_minor, val_minor, test_minor)

    print("Balanced counts:")
    print(balanced[args.target].value_counts())
    print("Majority class splits:")
    print(f"  train={len(train_major)} val={len(val_major)} test={len(test_major)}")
    print("Minority class splits:")
    print(f"  train={len(train_minor)} val={len(val_minor)} test={len(test_minor)}")


if __name__ == "__main__":
    main()
