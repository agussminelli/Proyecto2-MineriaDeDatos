from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score

from logistic_regression import load_and_preprocess, create_LR_pipeline
from run_experiment import apply_site_mi_weighting


FRACTION = 0.02
FEATURE_WEIGHT = 0.50
C = 1.0
BOOTSTRAP_REPS = 1000
BOOTSTRAP_SEED = 42


def macro_f1(y_true, y_pred, labels):
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def evaluate(y_true, y_pred, labels):
    return {
        "macro_f1": macro_f1(y_true, y_pred, labels),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def paired_bootstrap(predictions, labels, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    true = predictions["true_label"].to_numpy()
    e0 = predictions["prediction_E0"].to_numpy()
    e4 = predictions["prediction_E4"].to_numpy()
    indices_by_site = [np.flatnonzero(predictions["site"].to_numpy() == site)
                       for site in sorted(predictions["site"].unique())]
    rows = []

    for _ in range(reps):
        idx = np.concatenate([
            rng.choice(site_idx, size=len(site_idx), replace=True)
            for site_idx in indices_by_site
        ])
        rows.append({
            "delta_macro_f1": macro_f1(true[idx], e4[idx], labels)
                              - macro_f1(true[idx], e0[idx], labels),
            "delta_accuracy": accuracy_score(true[idx], e4[idx])
                              - accuracy_score(true[idx], e0[idx]),
        })

    sampled = pd.DataFrame(rows)
    return pd.DataFrame([
        {
            "metric": metric,
            "lower_95": sampled[metric].quantile(0.025),
            "upper_95": sampled[metric].quantile(0.975),
            "bootstrap_reps": reps,
            "resampling": "paired, stratified by site",
        }
        for metric in ["delta_macro_f1", "delta_accuracy"]
    ])


def run_final_seen_test(cfg: DictConfig):
    train_df, dev_df, test_df = load_and_preprocess(test=True, return_df=True)
    research_df = pd.concat([train_df, dev_df], ignore_index=True)

    if test_df is None or len(test_df) == 0:
        raise ValueError("The third return value must be the untouched test dataframe.")

    text_c = cfg.data.text_column
    target_c = cfg.data.target_column
    site_c = cfg.data.site_column
    labels = sorted(research_df[target_c].unique())

    if set(test_df[target_c].unique()) - set(labels):
        raise ValueError("Test contains target classes absent from Train + Dev.")
    if len(research_df) != 4688 or len(test_df) != 521:
        raise ValueError(
            f"Unexpected split sizes: Train+Dev={len(research_df)}, Test={len(test_df)}. "
            "Check preprocessing and data partitions before final evaluation."
        )

    # Compare composite record identifiers, since newid alone may repeat across sites.
    key_cols = [site_c, "module", "newid"]
    if any(col not in research_df.columns or col not in test_df.columns for col in key_cols):
        raise ValueError(f"Missing identifier columns: {key_cols}")
    train_ids = pd.MultiIndex.from_frame(research_df[key_cols])
    test_ids = pd.MultiIndex.from_frame(test_df[key_cols])
    if not train_ids.is_unique or not test_ids.is_unique:
        raise ValueError("Duplicate composite identifiers within a split.")
    if not train_ids.intersection(test_ids).empty:
        raise ValueError("Train + Dev overlaps with Test: possible data leakage.")

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = create_LR_pipeline(with_SVD=False, OAA=False)
    vectorizer = clone(pipeline.named_steps["tfidf"])
    classifier = clone(pipeline.named_steps["clf"])
    classifier.set_params(C=C)

    X_train = vectorizer.fit_transform(research_df[text_c])
    X_test = vectorizer.transform(test_df[text_c])
    y_train = research_df[target_c].to_numpy()
    y_test = test_df[target_c].to_numpy()

    print(f"Train+Dev={len(research_df)}, Test={len(test_df)}, "
          f"Features={X_train.shape[1]}, Classes={len(labels)}")

    # Baseline E0: original TF-IDF.
    e0_classifier = clone(classifier)
    e0_classifier.fit(X_train, y_train)
    e0_pred = e0_classifier.predict(X_test)

    # E4: MI is estimated only from training data; no test labels or sites are used.
    X_train_e4, X_test_e4, selected_idx, site_scores = apply_site_mi_weighting(
        X_train, X_test, research_df[site_c].to_numpy(),
        fraction=FRACTION, weight=FEATURE_WEIGHT
    )
    e4_classifier = clone(classifier)
    e4_classifier.fit(X_train_e4, y_train)
    e4_pred = e4_classifier.predict(X_test_e4)

    predictions = pd.DataFrame({
        "site": test_df[site_c].to_numpy(),
        "module": test_df["module"].to_numpy(),
        "newid": test_df["newid"].to_numpy(),
        "true_label": y_test,
        "prediction_E0": e0_pred,
        "prediction_E4": e4_pred,
        "correct_E0": (y_test == e0_pred),
        "correct_E4": (y_test == e4_pred),
    })
    predictions.to_csv(output_dir / "final_test_predictions.csv", index=False)

    rows = []
    for method, pred in [("E0", e0_pred), ("E4", e4_pred)]:
        rows.append({"method": method, "scope": "ALL", "n": len(y_test),
                     **evaluate(y_test, pred, labels)})
        for site in sorted(test_df[site_c].unique()):
            mask = test_df[site_c].to_numpy() == site
            rows.append({"method": method, "scope": str(site),
                         "n": int(mask.sum()),
                         **evaluate(y_test[mask], pred[mask], labels)})

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "final_test_results.csv", index=False)

    paired = results.pivot(index=["scope", "n"], columns="method",
                           values=["macro_f1", "accuracy"])
    paired.columns = [f"{metric}_{method}" for metric, method in paired.columns]
    paired = paired.reset_index()
    paired["delta_macro_f1"] = paired["macro_f1_E4"] - paired["macro_f1_E0"]
    paired["delta_accuracy"] = paired["accuracy_E4"] - paired["accuracy_E0"]
    paired.to_csv(output_dir / "final_test_comparison.csv", index=False)

    site_rows = results[results.scope != "ALL"]
    mean_sites = site_rows.groupby("method")[["macro_f1", "accuracy"]].mean()
    mean_sites.to_csv(output_dir / "final_test_mean_sites.csv")

    per_class = []
    for label in labels:
        mask = y_test == label
        for method, pred in [("E0", e0_pred), ("E4", e4_pred)]:
            per_class.append({
                "label": label,
                "method": method,
                "support": int(mask.sum()),
                "f1": float(f1_score(y_test, pred, labels=[label],
                                     average="macro", zero_division=0)),
                "recall": float(np.mean(pred[mask] == label)) if mask.any() else np.nan,
            })
    per_class = pd.DataFrame(per_class)
    per_class.to_csv(output_dir / "final_test_per_class.csv", index=False)

    selected_terms = pd.DataFrame({
        "feature": vectorizer.get_feature_names_out()[selected_idx],
        "site_mi": site_scores[selected_idx],
        "multiplier": FEATURE_WEIGHT,
    })
    selected_terms.to_csv(output_dir / "final_test_selected_features.csv", index=False)

    uncertainty = paired_bootstrap(predictions, labels)
    uncertainty.to_csv(output_dir / "final_test_bootstrap.csv", index=False)

    newly_correct = int(((y_test != e0_pred) & (y_test == e4_pred)).sum())
    newly_incorrect = int(((y_test == e0_pred) & (y_test != e4_pred)).sum())

    print("\nFINAL SEEN TEST (E0 vs E4; fixed C=1.0)")
    print(paired.round(5).to_string(index=False))
    print("\nMean across sites:\n", mean_sites.round(5).to_string())
    print("\nPaired bootstrap intervals:\n", uncertainty.round(5).to_string(index=False))
    print(f"\nE4 newly correct: {newly_correct}; newly incorrect: {newly_incorrect}")
    print(f"Results saved to: {output_dir}")
    return paired, predictions


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    run_final_seen_test(cfg)


if __name__ == "__main__":
    main()
