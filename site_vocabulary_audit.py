"""PHMRC-VAI site-associated vocabulary audit (exploratory, TRAIN only).

Run from your Proyecto2-MineriaDeDatos repository root:
    python site_vocabulary_audit.py

Alternatively:
    python site_vocabulary_audit.py --csv ../verbal_autopsy/cleaned_PHMRC_VAI_redacted_free_text.train.csv

Outputs CSV summaries under vocabulary_audit/. Uses document frequencies, not token counts.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def summarize(df, X, terms, masks, min_count=5):
    rows = []
    for site in sorted(df.site.dropna().unique()):
        yes = masks[site]
        no = ~yes
        ny, nn = int(yes.sum()), int(no.sum())
        if not ny or not nn:
            continue
        a = np.asarray(X[yes].sum(axis=0)).ravel()
        b = np.asarray(X[no].sum(axis=0)).ravel()
        py = a / ny
        pn = b / nn
        pooled = (a + b) / (ny + nn)
        z = (py - pn) / np.sqrt(np.maximum(pooled * (1 - pooled) * (1 / ny + 1 / nn), 1e-12))
        lift = (py + 1 / ny) / (pn + 1 / nn)
        keep = (a >= min_count) & (lift > 1.25) & (z > 0)
        ix = np.where(keep)[0]
        for j in ix:
            rows.append((site, terms[j], int(a[j]), ny, round(float(py[j]), 4),
                         round(float(pn[j]), 4), round(float(lift[j]), 3),
                         round(float(z[j]), 3)))
    return pd.DataFrame(rows, columns=["site", "term", "site_docs_with_term", "site_docs",
                                       "site_rate", "other_sites_rate", "lift", "z_score"])


def adjusted_audit(df, X, terms, min_count=5):
    """Within module+cause comparison against OTHER sites, aggregated by site.

    A site/stratum cell contributes only if it has >=3 narratives and the
    other sites collectively have >=10 in that same stratum.
    Scores are descriptive screening statistics, not significance tests.
    """
    rows = []
    group = df.groupby(["module", "gs_text34"], observed=True).indices
    for site in sorted(df.site.dropna().unique()):
        obs = np.zeros(X.shape[1], dtype=float)
        exp = np.zeros(X.shape[1], dtype=float)
        var = np.zeros(X.shape[1], dtype=float)
        n_eligible = 0
        strata = 0
        for indices in group.values():
            indices = np.asarray(indices, dtype=int)
            loc = (df.site.iloc[indices].to_numpy() == site)
            ia, ib = indices[loc], indices[~loc]
            if len(ia) < 3 or len(ib) < 10:
                continue
            a = np.asarray(X[ia].sum(axis=0)).ravel()
            b = np.asarray(X[ib].sum(axis=0)).ravel()
            # Laplace smoothing for rare terms, based on other sites only
            p = (b + 0.5) / (len(ib) + 1)
            obs += a
            exp += len(ia) * p
            var += len(ia) * p * (1-p)
            n_eligible += len(ia)
            strata += 1
        if not n_eligible:
            continue
        z = (obs - exp) / np.sqrt(np.maximum(var, 1e-9))
        lift = (obs + 1) / (exp + 1)
        keep = (obs >= min_count) & (lift > 1.3) & (z > 0)
        for j in np.where(keep)[0]:
            rows.append((site, terms[j], int(obs[j]), n_eligible, strata,
                         round(float(exp[j]), 2), round(float(lift[j]), 3),
                         round(float(z[j]), 3)))
    return pd.DataFrame(rows, columns=["site", "term", "observed_docs",
                                       "eligible_site_docs", "eligible_strata",
                                       "expected_docs_from_other_sites", "adjusted_lift", "adjusted_z"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("vocabulary_audit"))
    args = parser.parse_args()
    path = args.csv or (Path(__file__).resolve().parent.parent / "verbal_autopsy" /
                        "cleaned_PHMRC_VAI_redacted_free_text.train.csv")
    if not path.exists():
        raise SystemExit(f"CSV not found: {path}\nUse --csv to specify the train CSV path.")
    df = pd.read_csv(path)
    required = {"open_response", "site", "module", "gs_text34"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns: {sorted(missing)}. Found: {df.columns.tolist()}")
    df = df.dropna(subset=["site", "module", "gs_text34"]).copy().reset_index(drop=True)
    df["open_response"] = df["open_response"].fillna("").astype(str)
    # Use raw (lowercased) narratives, not aggressively normalized text:
    # spelling and style are part of what we're trying to detect.
    vec = CountVectorizer(binary=True, lowercase=True, stop_words=None,
                          ngram_range=(1, 2), min_df=8, max_df=0.95,
                          max_features=15000, token_pattern=r"(?u)\b[a-zA-Z]{2,}\b")
    X = vec.fit_transform(df.open_response)
    terms = vec.get_feature_names_out()
    masks = {s: (df.site.to_numpy() == s) for s in df.site.unique()}
    all_results = summarize(df, X, terms, masks)
    adult_df = df.loc[df.module.eq("Adult")].reset_index(drop=True)
    adult_X = X[df.module.eq("Adult").to_numpy()]
    adult_masks = {s: (adult_df.site.to_numpy() == s) for s in adult_df.site.unique()}
    adult_results = summarize(adult_df, adult_X, terms, adult_masks)
    adjusted = adjusted_audit(df, X, terms)
    args.out.mkdir(parents=True, exist_ok=True)
    files = [
        ("overall_site_terms.csv", all_results, "z_score"),
        ("adult_only_site_terms.csv", adult_results, "z_score"),
        ("module_cause_adjusted_terms.csv", adjusted, "adjusted_z"),
    ]
    print(f"Training narratives: {len(df)} | Vocabulary features: {len(terms)}")
    for name, results, score in files:
        if len(results):
            results = results.sort_values(["site", score], ascending=[True, False])
        results.to_csv(args.out / name, index=False)
        print(f"\n{name} ({len(results)} candidate site-term pairs)")
        if len(results):
            top = results.groupby("site", sort=True).head(8)
            print(top.to_string(index=False))
    print(f"\nSaved all CSVs to {args.out.resolve()}")
    print("Caution: scores flag associations; they do not establish causality or statistical significance.")


if __name__ == "__main__":
    main()
