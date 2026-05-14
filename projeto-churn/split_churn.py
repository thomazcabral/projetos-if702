
import pandas as pd


def split_no_replacement(df, train_frac, val_frac, test_frac, seed):
    if round(train_frac + val_frac + test_frac, 10) != 1.0:
        raise ValueError("Fractions must sum to 1.0")

    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_total = len(shuffled)
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)
    n_test = n_total - n_train - n_val

    # Extraindo os splits usando indexação
    train = shuffled.iloc[:n_train]
    val = shuffled.iloc[n_train : n_train + n_val]
    test = shuffled.iloc[n_train + n_val : n_train + n_val + n_test]

    return train, val, test


def oversample_to_match(minority_df, target_size, seed):
    if len(minority_df) >= target_size:
        return minority_df.copy()

    # Amostragem com reposição para alcançar o tamanho alvo
    return minority_df.sample(n=target_size, replace=True, random_state=seed).reset_index(drop=True)


def save_split(output_dir, prefix, train, val, test):
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / f"{prefix}_train.csv", index=False)
    val.to_csv(output_dir / f"{prefix}_val.csv", index=False)
    test.to_csv(output_dir / f"{prefix}_test.csv", index=False)




def main():
    input_path = "customer_churn_telecom_services.csv"
    output_dir = "splits"
    target = "Churn"
    seed = 42

    df = pd.read_csv(input_path)

    counts = df[target].value_counts(dropna=False)
    if len(counts) != 2:
        raise ValueError("Target column must have exactly 2 classes")

    # econtra as classes majoritária e minoritária
    majority_label = counts.idxmax()
    minority_label = counts.idxmin()

    class_major = df[df[target] == majority_label]
    class_minor = df[df[target] == minority_label]

    # realiza o split sem reposição para cada classe
    train_major, val_major, test_major = split_no_replacement(
        class_major, 0.5, 0.25, 0.25, seed
    )
    train_minor, val_minor, test_minor = split_no_replacement(
        class_minor, 0.5, 0.25, 0.25, seed
    )

    # oversample da classe minoritária no treino e validação
    train_minor_bal = oversample_to_match(train_minor, len(train_major), seed)
    val_minor_bal = oversample_to_match(val_minor, len(val_major), seed)

    # concatena e embaralha no final
    train = pd.concat([train_major, train_minor_bal], ignore_index=True)
    val = pd.concat([val_major, val_minor_bal], ignore_index=True)
    test = pd.concat([test_major, test_minor], ignore_index=True)

    train = train.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val = val.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test = test.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    save_split(output_dir, target.lower(), train, val, test)

    # verificação das contagens
    original_count_minority = len(class_minor)
    original_count_majority = len(class_major)
    
    print("Original counts:")
    print(f"No:   {original_count_majority} ({original_count_majority/len(df):.2%})")
    print(f"Yes:  {original_count_minority} ({original_count_minority/len(df):.2%})")

    print("Train counts (balanced):")
    print(f"No:   {len(train_major)} ({len(train_major)/len(train):.2%})")
    print(f"Yes:  {len(train_minor_bal)} ({len(train_minor_bal)/len(train):.2%})")

    print("Val counts (balanced):")
    print(f"No:   {len(val_major)} ({len(val_major)/len(val):.2%})")
    print(f"Yes:  {len(val_minor_bal)} ({len(val_minor_bal)/len(val):.2%})")

    print("Test counts (original):")
    print(f"No:   {len(test_major)} ({len(test_major)/len(test):.2%})")
    print(f"Yes:  {len(test_minor)} ({len(test_minor)/len(test):.2%})")


if __name__ == "__main__":
    main()
