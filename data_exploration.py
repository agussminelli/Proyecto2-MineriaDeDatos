import joblib
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

from simple_load_dataset import simple_load_dataset
from divide_data import create_stratified_partitions
from preprocess import preprocess_dataset


def explore_subgroups():
    # Same data and partitions as your baseline
    train_data, _, headers, _, _ = simple_load_dataset()
    train_df, dev_df, test_df = create_stratified_partitions(
        train_data, headers, target_col="gs_text34"
    )

    print("\nAVAILABLE COLUMNS:")
    print(list(train_df.columns))

    # Detect possible subgroup columns
    candidates = [
        c for c in train_df.columns
        if any(key in c.lower() for key in
               ["module", "site", "location", "country", "region"])
    ]
    print("\nPOSSIBLE SUBGROUP COLUMNS:", candidates)

    label = "gs_text34"

    print("\nOVERALL CAUSE DISTRIBUTION:")
    print(train_df[label].value_counts().to_string())

    print("\nMODULE DISTRIBUTION BY SITE:")
    print(pd.crosstab(
        train_df["site"],
        train_df["module"],
        margins=True
    ))

    print("\nMODULE DISTRIBUTION BY SITE (%):")
    print(
        pd.crosstab(
            train_df["site"],
            train_df["module"],
            normalize="index"
        ).mul(100).round(1)
    )

    # Inspect distributions for available subgroup columns
    for col in candidates:
        print(f"\n{'=' * 60}")
        print(f"SUBGROUP: {col}")

        counts = pd.crosstab(train_df[col], train_df[label])

        print("\nNumber of records:")
        print(train_df[col].value_counts(dropna=False))

        print("\nCause distribution (counts):")
        print(counts.to_string())

        print("\nCause distribution (row %):")
        print(
            pd.crosstab(
                train_df[col], train_df[label],
                normalize="index"
            ).mul(100).round(1).to_string()
        )

        # Causes absent from each subgroup
        print("\nNumber of causes present:")
        print((counts > 0).sum(axis=1))



    # Evaluate the already-trained baseline on DEV
    model = joblib.load("baseline_tfidf_svm.pkl")

    dev_df = preprocess_dataset(
        dev_df, text_cols=["open_response"], download=False
    )
    dev_df["prediction"] = model.predict(
        dev_df["open_response_clean"]
    )

    print("\nBASELINE PERFORMANCE BY SUBGROUP:")

    for col in candidates:
        results = []
        for group, part in dev_df.groupby(col):
            results.append({
                col: group,
                "n": len(part),
                "accuracy": accuracy_score(
                    part[label], part["prediction"]
                ),
                "macro_f1": f1_score(
                    part[label], part["prediction"],
                    average="macro", zero_division=0
                ),
                "n_classes": part[label].nunique()
            })

        print(f"\n{col}:")
        print(pd.DataFrame(results).round(3).to_string(index=False))


if __name__ == "__main__":
    explore_subgroups()
