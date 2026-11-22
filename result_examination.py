import pandas as pd
from pathlib import Path


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


if __name__ == "__main__":
    compare_E2_E3(
        e2_path= "output/E2_autocut_top_site_features/2026-11-21_19-27-44/e2_removed_features.csv", #"outputs/YOUR_E2_RUN/e2_removed_features.csv",
        e3_path= "output/E3_autocut_site_cod_features/2026-11-21_20-38-19/e3_removed_features.csv", #outputs/YOUR_E3_RUN/e3_removed_features.csv",
        output_dir="output/feature_selection_analysis",
        n_examples=15
    )