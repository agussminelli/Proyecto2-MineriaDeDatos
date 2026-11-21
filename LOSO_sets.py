import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut


def create_loso_sets(cfg, research_df):

    target_c=cfg.data.target_column
    site_c=cfg.data.site_column

    logo = LeaveOneGroupOut()

    for train_idx, eval_idx in logo.split(
        research_df,
        y=research_df[target_c],
        groups=research_df[site_c]
    ):
        fold_train = research_df.iloc[train_idx]
        fold_eval = research_df.iloc[eval_idx]

        held_out_site = fold_eval[site_c].iloc[0]

        print(
            f"{held_out_site}: "
            f"Train={len(fold_train)}, "
            f"Eval={len(fold_eval)}"
        )

        yield held_out_site, fold_train, fold_eval


def create_controlled_site_sets(cfg, research_df):
    # Identifier NewID in AP only unique within Modules
    research_df = research_df.copy().reset_index(drop=True)
    research_df["_row_id"] = np.arange(len(research_df))

    target = cfg.data.target_column
    site_col = cfg.data.site_column
    # add to config
    min_count = int(cfg.controlled_loso.min_count)
    eval_fraction = float(cfg.controlled_loso.eval_fraction)
    seeds = list(cfg.controlled_loso.seeds)

    counts = pd.crosstab(
        research_df[target],
        research_df[site_col]
    )

    # Classes represented sufficiently at all sites
    eligible = counts[
        (counts >= min_count).all(axis=1)
    ]

    classes = eligible.index.tolist()
    sites = counts.columns.tolist()

    if not classes:
        raise ValueError("No eligible causes found.")

    # Per-cause quotas: same for every site
    quotas = eligible.min(axis=1).astype(int).to_dict()

    # Reserve the same number of evaluation
    # examples for each site and each cause
    eval_quotas = {
        cause: max(1, int(round(n * eval_fraction)))
        for cause, n in quotas.items()
    }

    train_quotas = {
        cause: quotas[cause] - eval_quotas[cause]
        for cause in classes
    }

    for seed in seeds:
        rng = np.random.default_rng(seed)

        train_parts = {}
        eval_parts = {}

        # Build identical site/cause distributions
        for site in sites:
            for cause in classes:
                candidates = research_df[
                    (research_df[site_col] == site)
                    & (research_df[target] == cause)
                ]

                selected = candidates.sample(
                    n=quotas[cause],
                    random_state=int(
                        rng.integers(0, 2**31 - 1)
                    ),
                )

                shuffled = selected.sample(frac=1,
                    random_state=int(rng.integers(0, 2**31 - 1)),)

                n_eval = eval_quotas[cause]

                eval_parts[(site, cause)] = (shuffled.iloc[:n_eval].copy())
                train_parts[(site, cause)] = (shuffled.iloc[n_eval:].copy())

        # Six paired seen/unseen evaluations
        for target_site in sites:
            source_sites = [s for s in sites if s != target_site]

            unseen_parts = []
            seen_parts = []

            for cause in classes:
                n_train = train_quotas[cause]

                # Unseen: full training quota 5/6
                for source_site in source_sites:
                    unseen_parts.append(train_parts[(source_site, cause)])

                # Seen: include target-site narratives,
                # replacing the same total number of
                # examples from the five source sites
                seen_parts.append(train_parts[(target_site, cause)])

                # Allocate 4 * n_train across 5
                # source sites as evenly as possible
                total_source = 4 * n_train
                base, remainder = divmod(total_source, len(source_sites))

                source_order = list(rng.permutation(source_sites))

                for i, source_site in enumerate(source_order):
                    n_take = base + (i < remainder)

                    source_pool = train_parts[(source_site, cause)]

                    seen_parts.append(
                        source_pool.sample(n=n_take,
                            random_state=int(rng.integers(0, 2**31 - 1)),)
                    )

            train_unseen = pd.concat(unseen_parts, ignore_index=True)
            train_seen = pd.concat(seen_parts, ignore_index=True)

            eval_df = pd.concat(
                [eval_parts[(target_site, cause)]for cause in classes],
                ignore_index=True,
            )

            assert len(train_seen) == len(train_unseen)
            eval_ids = set(eval_df["_row_id"])
            assert eval_ids.isdisjoint(train_seen["_row_id"])
            assert eval_ids.isdisjoint(train_unseen["_row_id"])

            yield {
                "seed": seed,
                "site": target_site,
                "classes": classes,
                "train_unseen": train_unseen,
                "train_seen": train_seen,
                "eval": eval_df,
            }


def create_site_prediction_sets(cfg, research_df):

    target = cfg.data.target_column       # gs_text34
    site_col = cfg.data.site_column       # site

    min_count = int(cfg.controlled_loso.min_count)
    eval_fraction = float(cfg.controlled_loso.eval_fraction)
    seeds = list(cfg.controlled_loso.seeds)

    research_df = research_df.copy().reset_index(drop=True)
    research_df["_row_id"] = np.arange(len(research_df))

    counts = pd.crosstab(
        research_df[target],
        research_df[site_col]
    )

    eligible = counts[
        (counts >= min_count).all(axis=1)
    ]

    classes = eligible.index.tolist()
    sites = counts.columns.tolist()

    if not classes:
        raise ValueError("No eligible causes found.")

    quotas = eligible.min(axis=1).astype(int).to_dict()

    eval_quotas = {
        cause: max(1, int(round(n * eval_fraction)))
        for cause, n in quotas.items()
    }

    for seed in seeds:
        rng = np.random.default_rng(seed)

        train_parts = []
        eval_parts = []

        for site in sites:
            for cause in classes:

                candidates = research_df[
                    (research_df[site_col] == site)
                    & (research_df[target] == cause)
                ]

                selected = candidates.sample(
                    n=quotas[cause],
                    random_state=int(
                        rng.integers(0, 2**31 - 1)
                    )
                )

                shuffled = selected.sample(
                    frac=1,
                    random_state=int(
                        rng.integers(0, 2**31 - 1)
                    )
                )

                n_eval = eval_quotas[cause]

                eval_parts.append(
                    shuffled.iloc[:n_eval].copy()
                )

                train_parts.append(
                    shuffled.iloc[n_eval:].copy()
                )

        train_df = pd.concat(
            train_parts, ignore_index=True
        )

        eval_df = pd.concat(
            eval_parts, ignore_index=True
        )

        # Validation
        assert set(train_df["_row_id"]).isdisjoint(
            eval_df["_row_id"]
        )

        assert train_df[site_col].nunique() == len(sites)
        assert eval_df[site_col].nunique() == len(sites)

        # Identical cause distributions across sites
        assert (
            pd.crosstab(train_df[target], train_df[site_col])
            .nunique(axis=1) == 1
        ).all()

        assert (
            pd.crosstab(eval_df[target], eval_df[site_col])
            .nunique(axis=1) == 1
        ).all()

        yield {
            "seed": seed,
            "train": train_df,
            "eval": eval_df,
            "sites": sites,
            "classes": classes
        }

            
if __name__ == "__main__":
    from sklearn.metrics import f1_score
    from logistic_regression import load_and_preprocess, create_LR_pipeline


    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

#Usage
    train_df, dev_df,_ = load_and_preprocess(return_df=True)   
    research_df = pd.concat(
        [train_df, dev_df],
        ignore_index=True
    ) 

    for site, fold_train, fold_eval in create_loso_sets(
        research_df, adult_only=False
    ):
        X_train = fold_train["open_response_clean"]
        y_train = fold_train["gs_text34"]

        X_eval = fold_eval["open_response_clean"]
        y_eval = fold_eval["gs_text34"]

        # Important: create a fresh model for each fold!
        model = create_LR_pipeline(with_SVD=True, OAA=True)

        model.fit(X_train, y_train)
        predictions = model.predict(X_eval)

        print(
            site,
            f1_score(
                y_eval, predictions,
                average="macro",
                zero_division=0
            )
        )
