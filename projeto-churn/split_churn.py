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


def oversample_to_match(minority_df, target_size, seed):
    if len(minority_df) >= target_size:
        return minority_df.copy()

    return minority_df.sample(n=target_size, replace=True, random_state=seed).reset_index(drop=True)


def save_split(output_dir, prefix, train, val, test):
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / f"{prefix}_train.csv", index=False)
    val.to_csv(output_dir / f"{prefix}_val.csv", index=False)
    test.to_csv(output_dir / f"{prefix}_test.csv", index=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Split each class 50/25/25, then oversample minority in train/val only."
        )
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

    counts = df[args.target].value_counts(dropna=False)
    if len(counts) != 2:
        raise ValueError("Target column must have exactly 2 classes")

    majority_label = counts.idxmax()
    minority_label = counts.idxmin()

    class_major = df[df[args.target] == majority_label]
    class_minor = df[df[args.target] == minority_label]

    train_major, val_major, test_major = split_no_replacement(
        class_major, 0.5, 0.25, 0.25, args.seed
    )
    train_minor, val_minor, test_minor = split_no_replacement(
        class_minor, 0.5, 0.25, 0.25, args.seed
    )

    train_minor_bal = oversample_to_match(train_minor, len(train_major), args.seed)
    val_minor_bal = oversample_to_match(val_minor, len(val_major), args.seed)

    train = pd.concat([train_major, train_minor_bal], ignore_index=True)
    val = pd.concat([val_major, val_minor_bal], ignore_index=True)
    test = pd.concat([test_major, test_minor], ignore_index=True)

    train = train.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    val = val.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    test = test.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    save_split(output_dir, args.target.lower(), train, val, test)

    print("Original counts:")
    print(df[args.target].value_counts())
    print("Train counts (balanced):")
    print(train[args.target].value_counts())
    print("Val counts (balanced):")
    print(val[args.target].value_counts())
    print("Test counts (original):")
    print(test[args.target].value_counts())


if __name__ == "__main__":
    main()
