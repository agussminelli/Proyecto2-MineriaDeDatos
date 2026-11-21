import joblib
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

from simple_load_dataset import simple_load_dataset
from divide_data import create_stratified_partitions
from preprocess import preprocess_dataset
from logistic_regression import load_and_preprocess


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

def exploration_for_controlled_loso(train_df, dev_df):

    research_df = pd.concat([train_df, dev_df], ignore_index=True)

    counts = pd.crosstab(
        research_df["gs_text34"],
        research_df["site"]
    )

    for min_samples in [1, 5, 10, 15, 20, 30]:
        eligible = counts[
            (counts >= min_samples).all(axis=1)
        ]

        print(
            f"Minimum {min_samples:2d}: "
            f"{len(eligible)} classes, "
            f"{len(eligible) * min_samples * 6} "
            f"balanced samples"
        )

    print("\nCommon classes:")
    print(counts[(counts > 0).all(axis=1)])


def check_duplicated_newIDs(research_df):

    print("Total rows:", len(research_df))
    print("Unique newid:", research_df["newid"].nunique())

    print("Duplicate newid values:",
        research_df["newid"].duplicated().sum())

    print("Duplicate site-newid pairs:",
        research_df.duplicated(subset=["site", "newid"]).sum())

    duplicates = research_df[
        research_df.duplicated(
            subset=["site", "newid"],keep=False)
    ].copy()

    print("Rows involved:", len(duplicates))
    print("Unique duplicated groups:",
        duplicates.groupby(["site", "newid"]).ngroups)

    print("\nGroups with different causes:",
        (duplicates.groupby(["site", "newid"])["gs_text34"]
            .nunique() > 1).sum())

    print("\nGroups with different narratives:",
        (duplicates.groupby(["site", "newid"])["open_response_clean"]
            .nunique(dropna=False) > 1).sum())

    # print("\nExamples:")
    # print(duplicates[
    #         ["site", "newid", "module", "gs_text34", "open_response_clean"]
    #     ].sort_values(["site", "newid"]).head(12).to_string(index=False)
    # )

    print("Duplicate site-module-newid combinations:",
        research_df.duplicated(
            subset=["site", "module", "newid"]).sum())
    print("Groups with multiple modules:",
        (research_df.groupby(["site", "newid"])["module"]
            .nunique() > 1).sum())


if __name__ == "__main__":

    train_df, dev_df,_= load_and_preprocess(return_df=True)
    research_df = pd.concat([train_df, dev_df], ignore_index=True)
    check_duplicated_newIDs(research_df)
    # explore_subgroups()
