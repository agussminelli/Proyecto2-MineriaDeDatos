import numpy as np
import pandas as pd
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.base import clone
from sklearn.feature_selection import mutual_info_classif

from loso_sets import create_loso_sets, create_controlled_site_sets, create_site_prediction_sets
from logistic_regression import load_and_preprocess, create_LR_pipeline
from train_baseline import create_SVM_pipeline
from data_exploration import exploration_for_controlled_loso


def run_loso_experiments(cfg: DictConfig):

    adult_only=cfg.experiment.adult_only
    filter_voc=cfg.experiment.filter_voc
    text_c=cfg.data.text_column
    target_c=cfg.data.target_column
    site_c=cfg.data.site_column

    train_df, dev_df,_ = load_and_preprocess(return_df=True)
    research_df = pd.concat([train_df, dev_df],ignore_index=True)
    if cfg.experiment.adult_only:
        research_df = research_df[research_df["module"] == "Adult"]
        print("ADULT ONLY")
    labels = sorted(research_df["gs_text34"].unique())

    exp_combinations = [
        ("LR", True, True),
        ("LR", True, False),
        ("LR", False, True),
        ("LR", False, False),
        ("SVM", True, False)
    ]

    results = []
    summary_results = []
    all_predictions = []

    output_dir = Path(HydraConfig.get().runtime.output_dir)

    for classifier_name, with_SVD, OAA in exp_combinations:
        print(f"\nClassifier: {classifier_name}, With SVD={with_SVD}, OAA={OAA}")
        site_results = []
        all_y_true = []
        all_y_pred = []


        for site, fold_train, fold_eval in create_loso_sets(cfg, research_df):
            # Important: create a fresh model for each fold!
            if classifier_name == "SVM":
                pipeline = create_SVM_pipeline(cfg)
                c_param = "clf__C"
            else:
                pipeline = create_LR_pipeline(with_SVD=with_SVD,OAA=OAA,filter_voc=filter_voc)
                c_param = ("clf__estimator__C" if OAA else "clf__C")

            search = GridSearchCV(
                estimator=pipeline,
                param_grid={c_param:list(cfg.classifier.c_values)},
                scoring=cfg.validation.scoring,
                cv=GroupKFold(n_splits=cfg.validation.n_splits),
                n_jobs=-1,
                refit=True
            )
            # Grouped validation: no site appears
            # in both inner training and validation
            search.fit(fold_train[text_c],fold_train[target_c],groups=fold_train[site_c])
            
            predictions = search.predict(fold_eval[text_c])
            prediction_df = pd.DataFrame({
                "newid": fold_eval["newid"].to_numpy(),
                "site": site,
                "true_label": fold_eval[target_c].to_numpy(),
                "predicted_label": predictions,
                "classifier": classifier_name,
                "with_SVD": with_SVD,
                "OAA": OAA
            })
            all_predictions.append(prediction_df)

            f1 = f1_score(
                fold_eval[target_c],
                predictions,
                labels=labels,
                average="macro",
                zero_division=0
            )
            site_results.append(f1)
            all_y_true.extend(fold_eval[target_c].tolist())
            all_y_pred.extend(predictions.tolist())

            result = {
                "site": site,
                "adult_only": adult_only,
                "classifier": classifier_name,
                "with_SVD": with_SVD,
                "OAA": OAA,
                "train_size": len(fold_train),
                "eval_size": len(fold_eval),
                "best_C": search.best_params_[c_param],
                "cv_macro_f1": search.best_score_,
                "eval_macro_f1": f1,
                "eval_accuracy": accuracy_score(fold_eval[target_c],predictions)
            }
            results.append(result)


            print(
                f"{site}: "
                f"C={result['best_C']}, "
                f"CV={result['cv_macro_f1']:.4f}, "
                f"LOSO F1={result['eval_macro_f1']:.4f}"
            )

            filename = ("loso_adult.csv" if adult_only else "loso_all.csv")

            pd.DataFrame(results).to_csv(output_dir / filename,index=False)
            pd.concat(all_predictions,ignore_index=True).to_csv(
                output_dir / "predictions.csv",
                index=False
            )

        mean_loso_f1 = np.mean(site_results)
        std_loso_f1 = np.std(site_results, ddof=1)
        pooled_loso_f1 = f1_score(
            all_y_true,
            all_y_pred,
            labels=labels,
            average="macro",
            zero_division=0
        )
        print(f"\n{classifier_name}, SVD={with_SVD}, OAA={OAA}")
        print(f"Mean LOSO Macro-F1: {mean_loso_f1:.4f}")
        print(f"Std LOSO Macro-F1: {std_loso_f1:.4f}")
        print(f"Pooled LOSO Macro-F1: {pooled_loso_f1:.4f}")

        summary_results.append({
            "classifier": classifier_name,
            "with_SVD": with_SVD,
            "OAA": OAA,
            "adult_only": cfg.experiment.adult_only,
            "mean_loso_f1": float(mean_loso_f1),
            "std_loso_f1": float(std_loso_f1),
            "pooled_loso_f1": float(pooled_loso_f1)
        })

    pd.DataFrame(summary_results).to_csv(
        output_dir / "loso_summary.csv",
        index=False
    )

    # Save selected hyperparameters as structured YAML
    OmegaConf.save(
        OmegaConf.create({
            f"{r['site']}_{'svd' if r['with_SVD'] else 'no_svd'}_"
            f"{'oaa' if r['OAA'] else 'multinomial'}": {
                "best_C": float(r["best_C"]),
                "cv_macro_f1": float(r["cv_macro_f1"])
            }
            for r in results
        }),
        output_dir / "best_parameters.yaml"
    )
    return pd.DataFrame(results)


def run_controlled_loso_experiment(cfg):

    train_df, dev_df,_ = load_and_preprocess(return_df=True)
    research_df = pd.concat([train_df, dev_df],ignore_index=True)
    output_dir =  Path(HydraConfig.get().runtime.output_dir)
    filter_voc=cfg.experiment.filter_voc

    results = []

    for fold in create_controlled_site_sets(cfg, research_df):

        for condition in ["seen", "unseen"]:
            train = fold[f"train_{condition}"]
            eval_df = fold["eval"]

            pipeline = create_LR_pipeline(with_SVD=False,OAA=False,filter_voc=filter_voc)

            pipeline.set_params(clf__C=1.0)

            pipeline.fit(
                train["open_response_clean"],
                train["gs_text34"]
            )

            pred = pipeline.predict(eval_df["open_response_clean"])

            f1 = f1_score(
                eval_df["gs_text34"],
                pred,
                labels=fold["classes"],
                average="macro",
                zero_division=0
            )

            results.append({
                "seed": fold["seed"],
                "site": fold["site"],
                "condition": condition,
                "macro_f1": float(f1),
                "train_size": len(train),
                "eval_size": len(eval_df)
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        output_dir / "controlled_site_results.csv",
        index=False
    )

    paired = results_df.pivot(
        index=["seed", "site"],
        columns="condition",
        values="macro_f1"
    ).reset_index()

    paired["gap"] = paired["seen"] - paired["unseen"]

    summary = (
        paired.groupby("site")
        .agg(
            seen_mean=("seen", "mean"),
            seen_median=("seen", "median"),
            unseen_mean=("unseen", "mean"),
            unseen_median=("unseen", "median"),
            gap_mean=("gap", "mean"),
            gap_median=("gap", "median"),
            gap_std=("gap", "std"),
            gap_q25=("gap", lambda x: x.quantile(0.25)),
            gap_q75=("gap", lambda x: x.quantile(0.75)),
            n_seeds=("seed", "nunique")
        )
        .reset_index()
    )

    summary["gap_iqr"] = summary["gap_q75"] - summary["gap_q25"]

    summary.to_csv(
        output_dir / "controlled_site_summary.csv",
        index=False
    )


def run_predict_site_exp(cfg):

    train_df, dev_df,_ = load_and_preprocess(return_df=True)
    research_df = pd.concat([train_df, dev_df],ignore_index=True)
    output_dir =  Path(HydraConfig.get().runtime.output_dir)

    text_col = cfg.data.text_column
    site_col = cfg.data.site_column

    results = []
    all_predictions = []
    all_top_features = []

    for fold in create_site_prediction_sets(cfg, research_df):

        seed = fold["seed"]
        train = fold["train"]
        eval_df = fold["eval"]

        print(f"\nSeed: {seed}")
        print(f"Train: {len(train)}, Eval: {len(eval_df)}")

        for filter_voc in [False, True]:

            pipeline = create_LR_pipeline(with_SVD=False,OAA=False,filter_voc=filter_voc)
            pipeline.set_params(clf__C=1.0)

            pipeline.fit(train[text_col],train[site_col])
            pred = pipeline.predict(eval_df[text_col])

            site_f1 = f1_score(
                eval_df[site_col],
                pred,
                labels=fold["sites"],
                average="macro",
                zero_division=0
            )
            site_accuracy = accuracy_score(eval_df[site_col],pred)

            print(
                f"Filtered={filter_voc}: "
                f"F1={site_f1:.4f}, "
                f"Accuracy={site_accuracy:.4f}"
            )

            # Save performance
            results.append({
                "seed": seed,
                "filter_documentation": filter_voc,
                "macro_f1": float(site_f1),
                "accuracy": float(site_accuracy),
                "train_size": len(train),
                "eval_size": len(eval_df)
            })

            # Save individual predictions
            prediction_df = pd.DataFrame({
                "seed": seed,
                "filter_documentation": filter_voc,
                "site": eval_df[site_col].to_numpy(),
                "module": eval_df["module"].to_numpy(),
                "newid": eval_df["newid"].to_numpy(),
                "row_id": eval_df["_row_id"].to_numpy(),
                "predicted_site": pred
            })

            all_predictions.append(prediction_df)

            # Inspect feature coefficients
            tfidf = pipeline.named_steps["tfidf"]
            clf = pipeline.named_steps["clf"]

            feature_names = tfidf.get_feature_names_out()

            for i, site in enumerate(clf.classes_):

                top_idx = np.argsort(
                    clf.coef_[i]
                )[-15:][::-1]

                for rank, idx in enumerate(top_idx, start=1):
                    all_top_features.append({
                        "seed": seed,
                        "filter_documentation": filter_voc,
                        "site": site,
                        "rank": rank,
                        "feature": feature_names[idx],
                        "coefficient": float(clf.coef_[i, idx])
                    })


        # Save incrementally after each seed
        pd.DataFrame(results).to_csv(
            output_dir / "site_prediction_results.csv",
            index=False
        )

        pd.concat(
            all_predictions, ignore_index=True
        ).to_csv(
            output_dir / "site_prediction_predictions.csv",
            index=False
        )

        pd.DataFrame(all_top_features).to_csv(
            output_dir / "site_prediction_features.csv",
            index=False
        )

    # Summary over all seeds
    results_df = pd.DataFrame(results)

    paired = results_df.pivot(
        index="seed",
        columns="filter_documentation",
        values=["macro_f1", "accuracy"]
    )

    paired["delta_f1"] = (
        paired[("macro_f1", True)]
        - paired[("macro_f1", False)]
    )

    paired["delta_accuracy"] = (
        paired[("accuracy", True)]
        - paired[("accuracy", False)]
    )
    
    print("\nFiltering effect:")
    print(paired[["delta_f1", "delta_accuracy"]].mean())
    print(paired[["delta_f1", "delta_accuracy"]].median())

    summary = (
        results_df.groupby("filter_documentation")
        .agg(
            mean_f1=("macro_f1", "mean"),
            median_f1=("macro_f1", "median"),
            std_f1=("macro_f1", "std"),
            mean_accuracy=("accuracy", "mean"),
            median_accuracy=("accuracy", "median"),
            std_accuracy=("accuracy", "std"),
            n_seeds=("seed", "nunique")
        )
        .reset_index()
    )

    summary.to_csv(output_dir / "site_prediction_summary.csv",index=False)

    print("\nOverall summary:")
    print(summary.to_string(index=False))

    return results_df

def run_module_only_site_pred(cfg):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.linear_model import LogisticRegression

    train_df, dev_df,_ = load_and_preprocess(return_df=True)
    research_df = pd.concat([train_df, dev_df],ignore_index=True)
    output_dir =  Path(HydraConfig.get().runtime.output_dir)
    max_iter=cfg.classifier.max_iter
    random_state = cfg.svd.random_state

    for fold in create_site_prediction_sets(cfg, research_df):

        train = fold["train"]
        eval_df = fold["eval"]

        module_model = make_pipeline(
            OneHotEncoder(handle_unknown="ignore"),
            LogisticRegression(max_iter=max_iter,random_state=random_state)
        )

        module_model.fit(train[["module"]],train["site"])
        pred = module_model.predict(eval_df[["module"]])
        accuracy = accuracy_score(eval_df["site"], pred)
        print(f"Seed {fold['seed']}: {accuracy:.4f}")



def E2_autocut_top_site_features(cfg):
    """
    E2: Automatic removal of site-associated TF-IDF features.
    Thresholds: 1%,2%,5% and reference (0%)
    Evaluation:
        All modules, 14-class Leave-One-Site-Out
        Multinomial Logistic Regression, no SVD
        Fixed C=1.0
    Feature selection:
        Mutual Information(feature presence, site)
        Calculated ONLY on the five training sites.
    """

    train_df, dev_df, _ = load_and_preprocess(return_df=True)

    research_df = pd.concat([train_df, dev_df],ignore_index=True)

    text_c = cfg.data.text_column
    target_c = cfg.data.target_column
    site_c = cfg.data.site_column

    labels = sorted(research_df[target_c].unique())

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = [0.0, 0.01, 0.02, 0.05]

    results = []
    all_predictions = []
    all_removed_features = []

    reference_pipeline = create_LR_pipeline(with_SVD=False,OAA=False)
    # same vectorizer and classifier settings C=1.0 fixed
    base_classifier = clone(reference_pipeline.named_steps["clf"])
    base_classifier.set_params(C=1.0)

    # Outer Loso-Loop
    for held_out_site, fold_train, fold_eval in (
        create_loso_sets(cfg, research_df)
    ):

        print(f"\n{'=' * 50}")
        print(f"HELD-OUT SITE: {held_out_site}")
        print(f"{'=' * 50}")
        print(f"Train={len(fold_train)}\tEval={len(fold_eval)}")

        # 3.1 Fit TF-IDF on training sites ONLY
        vectorizer = clone(reference_pipeline.named_steps["tfidf"])

        X_train = vectorizer.fit_transform(fold_train[text_c])
        X_eval = vectorizer.transform(fold_eval[text_c])

        y_train = fold_train[target_c].to_numpy()
        y_eval = fold_eval[target_c].to_numpy()

        n_features = X_train.shape[1]
        feature_names = (vectorizer.get_feature_names_out())
        print(f"TF-IDF features: {n_features}")

        # 3.2 Calculate Site Mutual Information

        # Convert TF-IDF to binary presence/absence.
        # MI should measure whether a term occurs,
        # not the magnitude of its TF-IDF weight.
        X_binary = X_train.copy().tocsr()
        X_binary.data[:] = 1.0

        site_scores = mutual_info_classif(
            X_binary,
            fold_train[site_c].to_numpy(),
            discrete_features=True,
            random_state=42
        )

        # Document frequency:
        # Number of training narratives containing
        # the respective TF-IDF feature.
        doc_frequency = np.asarray(X_binary.getnnz(axis=0)).ravel()

        # Sort by decreasing site MI.
        # Feature index breaks ties deterministically.
        ranking = np.lexsort((np.arange(n_features),-site_scores))


        # 3.3 E0 + E2 filtering configurations
        for threshold in thresholds:

            if threshold == 0.0:
                experiment = "E0"
                removed_idx = np.array([], dtype=int)
            else:
                experiment = (f"E2_{int(threshold * 100)}pct")
                n_remove = max(1,int(np.ceil(n_features * threshold)))
                removed_idx = ranking[:n_remove]

            n_removed = len(removed_idx)

            print(f"\n{experiment}: Removing {n_removed}/{n_features} features")


            # 3.4 feature-column masking
            weights = np.ones(n_features,dtype=np.float64)
            weights[removed_idx] = 0.0

            X_train_filtered = (X_train.multiply(weights).tocsr())
            X_eval_filtered = (X_eval.multiply(weights).tocsr())

            # 3.5 Train Logistic Regression
            classifier = clone(base_classifier)
            classifier.fit(X_train_filtered,y_train)
            predictions = classifier.predict(X_eval_filtered)

            # 3.6 Calculate evaluation metrics
            macro_f1 = f1_score(
                y_eval,
                predictions,
                labels=labels,
                average="macro",
                zero_division=0
            )
            accuracy = accuracy_score(y_eval,predictions)
            print(
                f"Macro-F1={macro_f1:.4f}, "
                f"Accuracy={accuracy:.4f}"
            )


            # 3.7 Save metrics for this site and indiv. predictions
            results.append({
                "experiment": experiment,
                "held_out_site": held_out_site,
                "threshold": float(threshold),
                "n_features": int(n_features),
                "n_removed": int(n_removed),
                "train_size": len(fold_train),
                "eval_size": len(fold_eval),
                "C": 1.0,
                "macro_f1": float(macro_f1),
                "accuracy": float(accuracy)
            })

            prediction_df = pd.DataFrame({
                "experiment": experiment,
                "held_out_site": held_out_site,
                "site": fold_eval[site_c].to_numpy(),
                "module": fold_eval["module"].to_numpy(),
                "newid": fold_eval["newid"].to_numpy(),
                "true_label": y_eval,
                "predicted_label": predictions
            })
            all_predictions.append(prediction_df)

            # 3.8 Document removed features
            for rank, idx in enumerate(removed_idx,start=1):
                all_removed_features.append({
                    "experiment": experiment,
                    "held_out_site": held_out_site,
                    "threshold": float(threshold),
                    "rank": rank,
                    "feature": feature_names[idx],
                    "site_mi": float(site_scores[idx]),
                    "document_frequency": int(doc_frequency[idx]),
                    "document_frequency_ratio": float(
                        doc_frequency[idx] / len(fold_train)
                    )
                })
        # 3.9 Save incrementally after each site
        pd.DataFrame(results).to_csv(output_dir / "e2_results.csv",index=False)
        pd.concat(all_predictions,ignore_index=True).to_csv(
            output_dir / "e2_predictions.csv",
            index=False
        )
        pd.DataFrame(all_removed_features).to_csv(
            output_dir / "e2_removed_features.csv",
            index=False
        )
        print(f"\nSaved results for {held_out_site}")

    # 4. Aggregate results across all six sites
    results_df = pd.DataFrame(results)
    summary_rows = []
    for experiment, group in results_df.groupby("experiment",sort=False):

        # Mean and standard deviation of six site F1s
        mean_f1 = group["macro_f1"].mean()
        std_f1 = group["macro_f1"].std(ddof=1)
        median_f1 = group["macro_f1"].median()

        mean_accuracy = group["accuracy"].mean()

        # Pooled Macro-F1:
        experiment_predictions = pd.concat(
            [
                df for df in all_predictions
                if df["experiment"].iloc[0] == experiment
            ],
            ignore_index=True
        )

        pooled_f1 = f1_score(
            experiment_predictions["true_label"],
            experiment_predictions["predicted_label"],
            labels=labels,
            average="macro",
            zero_division=0
        )

        summary_rows.append({
            "experiment": experiment,
            "threshold": float(
                group["threshold"].iloc[0]
            ),
            "mean_loso_macro_f1": float(mean_f1),
            "median_loso_macro_f1": float(median_f1),
            "std_loso_macro_f1": float(std_f1),
            "pooled_loso_macro_f1": float(pooled_f1),
            "mean_accuracy": float(mean_accuracy),
            "n_sites": int(len(group))
        })

    summary = pd.DataFrame(summary_rows)

    # 5. Compare every experiment against E0
    baseline = results_df[
        results_df["experiment"] == "E0"
    ][["held_out_site", "macro_f1", "accuracy"]].rename(
        columns={
            "macro_f1": "baseline_f1",
            "accuracy": "baseline_accuracy"
        }
    )

    results_df = results_df.merge(
        baseline,
        on="held_out_site",
        how="left",
        validate="many_to_one"
    )

    results_df["delta_f1_vs_E0"] = (
        results_df["macro_f1"]
        - results_df["baseline_f1"]
    )

    results_df["delta_accuracy_vs_E0"] = (
        results_df["accuracy"]
        - results_df["baseline_accuracy"]
    )

    # Update per-site results with paired differences
    results_df.to_csv(output_dir / "e2_results.csv",index=False)

    mean_deltas = (
        results_df.groupby("experiment")[
            "delta_f1_vs_E0"
        ].mean()
    )

    summary["delta_mean_f1_vs_E0"] = (
        summary["experiment"].map(mean_deltas)
    )

    pooled_baseline = summary.loc[
        summary["experiment"] == "E0",
        "pooled_loso_macro_f1"
    ].iloc[0]

    summary["delta_pooled_f1_vs_E0"] = (
        summary["pooled_loso_macro_f1"]
        - pooled_baseline
    )

    summary.to_csv(output_dir / "e2_summary.csv",index=False)

    print("\n" + "=" * 60)
    print("E2 — FINAL LOSO SUMMARY")
    print("=" * 60)
    print(summary.to_string(index=False))

    return results_df, summary



def E3_autocut_site_cod_features(cfg):
    """E3: Remove features with high Site MI and low CoD MI."""

    train_df, dev_df, _ = load_and_preprocess(return_df=True)
    research_df = pd.concat([train_df, dev_df], ignore_index=True)

    text_c = cfg.data.text_column
    target_c = cfg.data.target_column
    site_c = cfg.data.site_column
    ordering_method = cfg.feature_filter.ordering_method
    labels = sorted(research_df[target_c].unique())

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = [0.0, 0.01, 0.02, 0.05]
    candidate_fraction = 0.10

    results = []
    all_predictions = []
    all_removed_features = []

    reference_pipeline = create_LR_pipeline(with_SVD=False, OAA=False)
    base_classifier = clone(reference_pipeline.named_steps["clf"])
    base_classifier.set_params(C=1.0)

    for held_out_site, fold_train, fold_eval in create_loso_sets(cfg, research_df):
        print(f"\n{'=' * 50}\nHELD-OUT SITE: {held_out_site}\n{'=' * 50}")

        vectorizer = clone(reference_pipeline.named_steps["tfidf"])
        X_train = vectorizer.fit_transform(fold_train[text_c])
        X_eval = vectorizer.transform(fold_eval[text_c])

        y_train = fold_train[target_c].to_numpy()
        y_eval = fold_eval[target_c].to_numpy()
        sites_train = fold_train[site_c].to_numpy()

        n_features = X_train.shape[1]
        feature_names = vectorizer.get_feature_names_out()

        X_binary = X_train.copy().tocsr()
        X_binary.data[:] = 1.0

        site_scores = mutual_info_classif(
            X_binary, sites_train, discrete_features=True, random_state=42
        )

        cod_scores = mutual_info_classif(
            X_binary, y_train, discrete_features=True, random_state=42
        )

        doc_frequency = np.asarray(X_binary.getnnz(axis=0)).ravel()

        # Candidate pool: top 10% by Site MI
        site_ranking = np.lexsort((np.arange(n_features), -site_scores))
        n_candidates = max(1, int(np.ceil(n_features * candidate_fraction)))
        candidates = site_ranking[:n_candidates]

        if ordering_method == "cod_first":
            # E3a: prioritize lowest CoD MI
            print("E3a - CoD first")
            candidate_order = np.lexsort((
                candidates,
                -site_scores[candidates],
                cod_scores[candidates]
            ))
            ranking = candidates[candidate_order]

        elif ordering_method == "site_first_protected":
            print("E3b - Site first protected")
            # E3b: protect the top 20% by CoD MI
            protection_fraction = 0.20
            n_protected = int(np.ceil(n_candidates * protection_fraction))

            cod_ranking = np.lexsort((
                candidates,
                -cod_scores[candidates]
            ))

            protected_idx = candidates[cod_ranking[:n_protected]]
            eligible_idx = candidates[~np.isin(candidates, protected_idx)]

            # Among remaining features, prioritize highest Site MI
            eligible_order = np.lexsort((
                eligible_idx,
                -site_scores[eligible_idx]
            ))
            ranking = eligible_idx[eligible_order]

            max_remove = int(np.ceil(n_features * max(thresholds)))

            if len(ranking) < max_remove:
                raise ValueError(
                    f"Only {len(ranking)} features available, "
                    f"but {max_remove} are required."
                )

        else:
            raise ValueError(f"Unknown E3 ordering method: {ordering_method}")

        print(f"TF-IDF features: {n_features}, E3 candidates: {n_candidates}")

        for threshold in thresholds:
            experiment = "E0" if threshold == 0 else f"E3_{int(threshold * 100)}pct"
            n_remove = 0 if threshold == 0 else max(1, int(np.ceil(n_features * threshold)))
            removed_idx = ranking[:n_remove]

            weights = np.ones(n_features, dtype=np.float64)
            weights[removed_idx] = 0.0

            X_train_filtered = X_train.multiply(weights).tocsr()
            X_eval_filtered = X_eval.multiply(weights).tocsr()

            classifier = clone(base_classifier)
            classifier.fit(X_train_filtered, y_train)
            predictions = classifier.predict(X_eval_filtered)

            macro_f1 = f1_score(
                y_eval, predictions, labels=labels, average="macro", zero_division=0
            )
            accuracy = accuracy_score(y_eval, predictions)

            print(
                f"{experiment}: Removed={len(removed_idx)}, "
                f"Macro-F1={macro_f1:.4f}, Accuracy={accuracy:.4f}"
            )

            results.append({
                "experiment": experiment,
                "ordering_method": ordering_method,
                "held_out_site": held_out_site,
                "threshold": float(threshold),
                "candidate_fraction": candidate_fraction,
                "n_features": int(n_features),
                "n_candidates": int(n_candidates),
                "n_removed": int(len(removed_idx)),
                "train_size": len(fold_train),
                "eval_size": len(fold_eval),
                "C": 1.0,
                "macro_f1": float(macro_f1),
                "accuracy": float(accuracy)
            })

            all_predictions.append(pd.DataFrame({
                "experiment": experiment,
                "held_out_site": held_out_site,
                "site": fold_eval[site_c].to_numpy(),
                "module": fold_eval["module"].to_numpy(),
                "newid": fold_eval["newid"].to_numpy(),
                "true_label": y_eval,
                "predicted_label": predictions
            }))

            for rank, idx in enumerate(removed_idx, start=1):
                all_removed_features.append({
                    "experiment": experiment,
                    "ordering_method": ordering_method,
                    "held_out_site": held_out_site,
                    "threshold": float(threshold),
                    "rank": rank,
                    "feature": feature_names[idx],
                    "site_mi": float(site_scores[idx]),
                    "cod_mi": float(cod_scores[idx]),
                    "document_frequency": int(doc_frequency[idx]),
                    "document_frequency_ratio": float(doc_frequency[idx] / len(fold_train))
                })

        # Incremental CSV saving
        pd.DataFrame(results).to_csv(output_dir / "e3_results.csv", index=False)
        pd.concat(all_predictions, ignore_index=True).to_csv(
            output_dir / f"e3_{ordering_method}_predictions.csv", index=False
        )
        pd.DataFrame(all_removed_features).to_csv(
            output_dir / f"e3_{ordering_method}_removed_features.csv", index=False
        )

    # Aggregate LOSO results
    results_df = pd.DataFrame(results)
    summary_rows = []

    for experiment, group in results_df.groupby("experiment", sort=False):
        experiment_predictions = pd.concat(
            [df for df in all_predictions if df["experiment"].iloc[0] == experiment],
            ignore_index=True
        )

        pooled_f1 = f1_score(
            experiment_predictions["true_label"],
            experiment_predictions["predicted_label"],
            labels=labels, average="macro", zero_division=0
        )

        summary_rows.append({
            "experiment": experiment,
            "ordering_method": ordering_method,
            "threshold": float(group["threshold"].iloc[0]),
            "mean_loso_macro_f1": float(group["macro_f1"].mean()),
            "median_loso_macro_f1": float(group["macro_f1"].median()),
            "std_loso_macro_f1": float(group["macro_f1"].std(ddof=1)),
            "pooled_loso_macro_f1": float(pooled_f1),
            "mean_accuracy": float(group["accuracy"].mean()),
            "n_sites": int(len(group))
        })

    summary = pd.DataFrame(summary_rows)

    # Paired comparison against E0
    baseline = results_df[results_df["experiment"] == "E0"][
        ["held_out_site", "macro_f1", "accuracy"]
    ].rename(columns={
        "macro_f1": "baseline_f1",
        "accuracy": "baseline_accuracy"
    })

    results_df = results_df.merge(
        baseline, on="held_out_site", how="left", validate="many_to_one"
    )

    results_df["delta_f1_vs_E0"] = results_df["macro_f1"] - results_df["baseline_f1"]
    results_df["delta_accuracy_vs_E0"] = results_df["accuracy"] - results_df["baseline_accuracy"]

    mean_deltas = results_df.groupby("experiment")["delta_f1_vs_E0"].mean()
    summary["delta_mean_f1_vs_E0"] = summary["experiment"].map(mean_deltas)

    pooled_baseline = summary.loc[
        summary["experiment"] == "E0", "pooled_loso_macro_f1"
    ].iloc[0]
    summary["delta_pooled_f1_vs_E0"] = summary["pooled_loso_macro_f1"] - pooled_baseline

    results_df.to_csv(output_dir / f"e3_{ordering_method}_results.csv", index=False)
    summary.to_csv(output_dir / f"e3_{ordering_method}_summary.csv", index=False)

    print("\n" + "=" * 60)
    print("E3 — FINAL LOSO SUMMARY")
    print("=" * 60)
    print(summary.to_string(index=False))

    return results_df, summary


@hydra.main(version_base=None,config_path="conf",config_name="config")
def main(cfg: DictConfig):

    exp=cfg.experiment.name
    if exp=="loso_LR":
        run_loso_experiments(cfg)
    elif exp == "loso_LR_controlled":
        run_controlled_loso_experiment(cfg)
    elif exp == "site_prediction":
        run_predict_site_exp(cfg)
    elif exp == "module_influence_on_site":
        run_module_only_site_pred(cfg)
    elif exp == "E2_autocut_top_site_features":
        E2_autocut_top_site_features(cfg)
    elif exp == "E3_autocut_site_cod_features":
        E3_autocut_site_cod_features(cfg)
    else:
        print("Please check experiment name in config")

    # exploration_for_controlled_loso(train_df, dev_df)


if __name__ == "__main__":

    main()

    # # All modules
    # py loso_experiments.py experiment.adult_only=false

    # # Adult only
    # py loso_experiments.py experiment.adult_only=true

    # # Different regularization search
    # py loso_experiments.py classifier.c_values=[0.1,1,10]