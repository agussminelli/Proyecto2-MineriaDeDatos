import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, recall_score


def compare_E2_E3(e2_path, e3_path, output_dir, n_examples=15):
    e2 = pd.read_csv(e2_path)
    e3 = pd.read_csv(e3_path)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    example_rows = []

    thresholds = sorted(set(e2["threshold"]) & set(e3["threshold"]))
    sites = sorted(set(e2["held_out_site"]) & set(e3["held_out_site"]))

    for site in sites:
        for threshold in thresholds:
            a = e2[(e2["held_out_site"] == site) & (e2["threshold"].round(4) == round(threshold, 4))]
            b = e3[(e3["held_out_site"] == site) & (e3["threshold"].round(4) == round(threshold, 4))]

            if a.empty or b.empty:
                continue

            e2_features = set(a["feature"])
            e3_features = set(b["feature"])
            overlap = e2_features & e3_features

            # Compare Site MI using each method's selected features
            summary_rows.append({
                "site": site,
                "threshold": threshold,
                "e2_removed": len(e2_features),
                "e3_removed": len(e3_features),
                "overlap": len(overlap),
                "overlap_pct": 100 * len(overlap) / len(e2_features),
                "e2_mean_site_mi": a["site_mi"].mean(),
                "e3_mean_site_mi": b["site_mi"].mean(),
                "e3_mean_cod_mi": b["cod_mi"].mean()
            })

            # Strong Site-MI features removed by E2 but retained by E3
            retained = a[~a["feature"].isin(e3_features)]
            retained = retained.sort_values("site_mi", ascending=False).head(n_examples)

            for _, row in retained.iterrows():
                example_rows.append({
                    "site": site,
                    "threshold": threshold,
                    "feature": row["feature"],
                    "site_mi": row["site_mi"],
                    "e2_rank": row["rank"],
                    "e2_document_frequency": row["document_frequency"]
                })

    summary = pd.DataFrame(summary_rows)
    examples = pd.DataFrame(example_rows)

    summary.to_csv(output_dir / "e2_e3a_comparison.csv", index=False)
    examples.to_csv(output_dir / "e2_features_retained_by_e3a.csv", index=False)

    print("\nE2 vs E3a — Feature selection comparison:")
    print(summary.round(4).to_string(index=False))

    print(f"\nSaved comparison to {output_dir}")

    return summary, examples


def E5_output_diagnostic(e5_path):
    df = pd.read_csv(e5_path)

    df["df_ratio"] = (
        np.maximum(df["e2_document_frequency"], df["random_document_frequency"])
        / np.minimum(df["e2_document_frequency"], df["random_document_frequency"])
    )

    summary = df.groupby("threshold").agg(
        mean_distance=("log_df_distance", "mean"),
        median_distance=("log_df_distance", "median"),
        p95_distance=("log_df_distance", lambda x: x.quantile(0.95)),
        mean_df_ratio=("df_ratio", "mean"),
        pct_within_20pct=("df_ratio", lambda x: 100 * (x <= 1.2).mean()),
        pct_within_50pct=("df_ratio", lambda x: 100 * (x <= 1.5).mean())
    )

    print(summary.round(3))


def run_robustness_analysis(e4_dir, output_dir,configs):
    """Post-hoc E4 robustness analysis; no model retraining required."""
    e4_dir, output_dir = Path(e4_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(e4_dir / "e4_results.csv")
    predictions_path = e4_dir / "e4_predictions.csv"
    required = {"mode", "eval_site", "experiment", "macro_f1"}
    if not required.issubset(results.columns):
        raise ValueError(f"Missing result columns: {required - set(results.columns)}")

    per_site_rows, sensitivity_rows, overview_rows = [], [], []

    for mode in ("Unseen", "Seen"):
        # Seen ALL is pooled and must NOT count as a seventh geographical site.
        rows = results[(results["mode"] == mode) & (results["eval_site"] != "ALL")]
        baseline = rows[rows["experiment"] == "E0"].set_index("eval_site")
        if baseline.empty or baseline.index.has_duplicates:
            raise ValueError(f"Missing or duplicate E0 site scores for {mode}")

        for experiment in configs:
            chosen = rows[rows["experiment"] == experiment].set_index("eval_site")
            if set(chosen.index) != set(baseline.index) or chosen.index.has_duplicates:
                raise ValueError(f"Site mismatch for {mode} / {experiment}")

            sites = sorted(baseline.index)
            delta = chosen.loc[sites, "macro_f1"] - baseline.loc[sites, "macro_f1"]
            overall_delta = float(delta.mean())

            for site in sites:
                per_site_rows.append({
                    "mode": mode, "experiment": experiment, "site": site,
                    "e0_f1": float(baseline.loc[site, "macro_f1"]),
                    "method_f1": float(chosen.loc[site, "macro_f1"]),
                    "delta_f1": float(delta.loc[site])
                })
                other_sites_mean = float(delta.drop(site).mean())
                sensitivity_rows.append({
                    "mode": mode, "experiment": experiment, "excluded_site": site,
                    "full_six_site_mean_delta": overall_delta,
                    "five_site_mean_delta": other_sites_mean,
                    "change_from_full_mean": other_sites_mean - overall_delta
                })

            excluded_means = [float(delta.drop(site).mean()) for site in sites]
            overview_rows.append({
                "mode": mode, "experiment": experiment,
                "mean_site_delta_f1": overall_delta,
                "sites_improved": int((delta > 0).sum()),
                "sites_degraded": int((delta < 0).sum()),
                "sites_unchanged": int((delta == 0).sum()),
                "minimum_site_delta": float(delta.min()),
                "maximum_site_delta": float(delta.max()),
                "minimum_five_site_delta": min(excluded_means),
                "maximum_five_site_delta": max(excluded_means),
                "positive_after_every_exclusion": all(x > 0 for x in excluded_means)
            })

    per_site = pd.DataFrame(per_site_rows)
    sensitivity = pd.DataFrame(sensitivity_rows)
    overview = pd.DataFrame(overview_rows)

    per_site.to_csv(output_dir / "robustness_per_site.csv", index=False)
    sensitivity.to_csv(output_dir / "robustness_leave_one_site_out.csv", index=False)
    overview.to_csv(output_dir / "robustness_overview.csv", index=False)

    print("\nROBUSTNESS OVERVIEW")
    print(overview.round(5).to_string(index=False))

    # Per-class analysis requires the individual predictions, not only site F1.
    if not predictions_path.is_file():
        print(f"\nNo {predictions_path}: skipping per-class analysis.")
        return overview, sensitivity, per_site, None

    predictions = pd.read_csv(predictions_path)
    keys = ["site", "module", "newid"]
    needed = set(keys + ["mode", "experiment", "true_label", "predicted_label"])
    if not needed.issubset(predictions.columns):
        raise ValueError(f"Missing prediction columns: {needed - set(predictions.columns)}")

    class_rows = []
    class_summaries = []

    for mode in ("Unseen", "Seen"):
        frame = predictions[predictions["mode"] == mode]
        e0 = frame[frame["experiment"] == "E0"][
            keys + ["true_label", "predicted_label"]
        ].rename(columns={"predicted_label": "baseline_prediction"})

        if e0.duplicated(keys).any():
            raise ValueError(f"E0 identifiers not unique in {mode}; inspect composite keys")

        for experiment in configs:
            chosen = frame[frame["experiment"] == experiment][
                keys + ["true_label", "predicted_label"]
            ].rename(columns={
                "true_label": "selected_true_label",
                "predicted_label": "selected_prediction"
            })

            matched = e0.merge(chosen, on=keys, validate="one_to_one", how="inner")
            if len(matched) != len(e0) or len(matched) != len(chosen):
                raise ValueError(f"Predictions do not match for {mode} / {experiment}")
            if not (matched["true_label"] == matched["selected_true_label"]).all():
                raise ValueError(f"True-label mismatch for {mode} / {experiment}")

            truth = matched["true_label"].to_numpy()
            baseline_pred = matched["baseline_prediction"].to_numpy()
            chosen_pred = matched["selected_prediction"].to_numpy()
            labels = sorted(frame["true_label"].dropna().unique())

            e0_f1 = f1_score(truth, baseline_pred, labels=labels, average=None, zero_division=0)
            method_f1 = f1_score(truth, chosen_pred, labels=labels, average=None, zero_division=0)
            e0_recall = recall_score(truth, baseline_pred, labels=labels, average=None, zero_division=0)
            method_recall = recall_score(truth, chosen_pred, labels=labels, average=None, zero_division=0)

            for i, label in enumerate(labels):
                class_mask = truth == label
                corrected = int(np.sum(class_mask & (baseline_pred != truth) & (chosen_pred == truth)))
                broken = int(np.sum(class_mask & (baseline_pred == truth) & (chosen_pred != truth)))

                class_rows.append({
                    "mode": mode, "experiment": experiment, "cause": label,
                    "n_true": int(class_mask.sum()),
                    "e0_f1": float(e0_f1[i]), "method_f1": float(method_f1[i]),
                    "delta_f1": float(method_f1[i] - e0_f1[i]),
                    "e0_recall": float(e0_recall[i]),
                    "method_recall": float(method_recall[i]),
                    "corrected_by_method": corrected,
                    "made_incorrect_by_method": broken
                })

            class_delta = method_f1 - e0_f1
            class_summaries.append({
                "mode": mode, "experiment": experiment,
                "n_classes_improved": int(np.sum(class_delta > 0)),
                "n_classes_degraded": int(np.sum(class_delta < 0)),
                "largest_class_gain": float(class_delta.max()),
                "largest_class_loss": float(class_delta.min())
            })

    per_class = pd.DataFrame(class_rows)
    per_class.to_csv(output_dir / "robustness_per_class.csv", index=False)
    pd.DataFrame(class_summaries).to_csv(
        output_dir / "robustness_class_summary.csv", index=False
    )

    print("\nPER-CLASS SUMMARY")
    print(pd.DataFrame(class_summaries).round(5).to_string(index=False))
    print(f"\nSaved to: {output_dir.resolve()}")
    return overview, sensitivity, per_site, per_class


if __name__ == "__main__":
    # compare_E2_E3(
    #     e2_path= "output/E2_autocut_top_site_features/2026-11-21_19-27-44/e2_removed_features.csv", #"outputs/YOUR_E2_RUN/e2_removed_features.csv",
    #     e3_path= "output/E3_autocut_site_cod_features/2026-11-21_20-38-19/e3_removed_features.csv", #outputs/YOUR_E3_RUN/e3_removed_features.csv",
    #     output_dir="output/feature_selection_analysis",
    #     n_examples=15
    # )

    # E5_output_diagnostic(
    #     e5_path="output/E5_frequency_matched_random/2026-11-22_15-39-24/e5b_random_removed_features.csv",#outputs/YOUR_E5_RUN/e5_random_removed_features.csv"
    # )

    run_robustness_analysis(
        e4_dir=Path(r"output/E4_feature_downweighting/2026-11-23_16-58-19"), #"outputs/YOUR_E4_RUN_DIR"
        output_dir=Path("output/E4_robustness_analysis"),
        configs=["E2_2pct_w0.50", "E2_5pct_w0.50", "E3b_5pct_w0.50"]
    )