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


def run_controlled_loso_experiment(cfg, apply_site_weighting=False):
    """Controlled 5-class LOSO, optionally comparing E0 with E4."""

    train_df, dev_df, _ = load_and_preprocess(return_df=True)
    research_df = pd.concat([train_df, dev_df], ignore_index=True)

    text_c = cfg.data.text_column
    target_c = cfg.data.target_column
    site_c = cfg.data.site_column

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fraction = 0.02
    feature_weight = 0.50

    results = []
    prediction_frames = []
    selected_feature_rows = []

    reference_pipeline = create_LR_pipeline(
        with_SVD=False, OAA=False, filter_voc=False
    )

    classifier_template = clone(reference_pipeline.named_steps["clf"])
    classifier_template.set_params(C=1.0)

    for fold in create_controlled_site_sets(cfg, research_df):
        seed = int(fold["seed"])
        site = fold["site"]
        eval_df = fold["eval"]

        print(f"\nSeed={seed}, Held-out site={site}")

        for condition in ["seen", "unseen"]:
            fold_train = fold[f"train_{condition}"]

            vectorizer = clone(reference_pipeline.named_steps["tfidf"])
            X_train = vectorizer.fit_transform(fold_train[text_c])
            X_eval = vectorizer.transform(eval_df[text_c])

            y_train = fold_train[target_c].to_numpy()
            y_eval = eval_df[target_c].to_numpy()

            configurations = {
                "E0": (X_train, X_eval)
            }

            if apply_site_weighting:
                X_train_e4, X_eval_e4, selected_idx, site_scores = (
                    apply_site_mi_weighting(
                        X_train, X_eval, fold_train[site_c].to_numpy(),
                        fraction=fraction, weight=feature_weight
                    )
                )

                configurations["E4"] = (X_train_e4, X_eval_e4)

                feature_names = vectorizer.get_feature_names_out()

                for rank, idx in enumerate(selected_idx, start=1):
                    selected_feature_rows.append({
                        "seed": seed,
                        "site": site,
                        "condition": condition,
                        "rank": rank,
                        "feature": feature_names[idx],
                        "site_mi": float(site_scores[idx])
                    })

            for method, (X_tr, X_ev) in configurations.items():
                classifier = clone(classifier_template)
                classifier.fit(X_tr, y_train)
                predicted = classifier.predict(X_ev)

                macro_f1 = f1_score(
                    y_eval, predicted, labels=fold["classes"],
                    average="macro", zero_division=0
                )
                accuracy = accuracy_score(y_eval, predicted)

                results.append({
                    "seed": seed,
                    "site": site,
                    "condition": condition,
                    "method": method,
                    "threshold": fraction if method == "E4" else 0.0,
                    "weight": feature_weight if method == "E4" else 1.0,
                    "macro_f1": float(macro_f1),
                    "accuracy": float(accuracy),
                    "train_size": len(fold_train),
                    "eval_size": len(eval_df),
                    "n_features": X_train.shape[1],
                    "n_selected": len(selected_idx) if method == "E4" else 0,
                    "C": 1.0
                })

                prediction_frames.append(pd.DataFrame({
                    "seed": seed,
                    "site": site,
                    "condition": condition,
                    "method": method,
                    "module": eval_df["module"].to_numpy(),
                    "newid": eval_df["newid"].to_numpy(),
                    "true_label": y_eval,
                    "predicted_label": predicted
                }))

                print(
                    f"  {condition:6s} | {method} | "
                    f"Macro-F1={macro_f1:.4f} | Accuracy={accuracy:.4f}"
                )

        # Save after each complete seed/site evaluation
        pd.DataFrame(results).to_csv(
            output_dir / "controlled_site_results.csv", index=False
        )

    results_df = pd.DataFrame(results)

    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(
            output_dir / "controlled_site_predictions.csv", index=False
        )

    if selected_feature_rows:
        pd.DataFrame(selected_feature_rows).to_csv(
            output_dir / "controlled_site_selected_features.csv", index=False
        )

    if not apply_site_weighting:
        return results_df

    # Paired differences: E4 minus E0
    pairs = results_df.pivot(
        index=["seed", "site", "condition"],
        columns="method",
        values=["macro_f1", "accuracy"]
    )

    pairs.columns = [f"{metric}_{method}" for metric, method in pairs.columns]
    pairs = pairs.reset_index()

    pairs["delta_f1"] = pairs["macro_f1_E4"] - pairs["macro_f1_E0"]
    pairs["delta_accuracy"] = pairs["accuracy_E4"] - pairs["accuracy_E0"]

    pairs.to_csv(output_dir / "controlled_site_deltas.csv", index=False)

    # Seen-unseen gap for each method on the same seed/site
    gaps = results_df.pivot(
        index=["seed", "site", "method"],
        columns="condition",
        values="macro_f1"
    ).reset_index()

    gaps["seen_unseen_gap"] = gaps["seen"] - gaps["unseen"]

    gap_comparison = gaps.pivot(
        index=["seed", "site"],
        columns="method",
        values="seen_unseen_gap"
    ).reset_index()

    gap_comparison["delta_gap"] = (
        gap_comparison["E4"] - gap_comparison["E0"]
    )

    gap_comparison.to_csv(
        output_dir / "controlled_site_gap_comparison.csv", index=False
    )

    # Aggregate performance per site and condition
    site_summary = pairs.groupby(
        ["site", "condition"], as_index=False
    ).agg(
        e0_mean_f1=("macro_f1_E0", "mean"),
        e4_mean_f1=("macro_f1_E4", "mean"),
        mean_delta_f1=("delta_f1", "mean"),
        std_delta_f1=("delta_f1", "std"),
        e0_mean_accuracy=("accuracy_E0", "mean"),
        e4_mean_accuracy=("accuracy_E4", "mean"),
        mean_delta_accuracy=("delta_accuracy", "mean"),
        n_seeds=("seed", "nunique")
    )

    site_summary.to_csv(
        output_dir / "controlled_site_summary.csv", index=False
    )

    # Overall results
    overall_summary = pairs.groupby(
        "condition", as_index=False
    ).agg(
        e0_mean_f1=("macro_f1_E0", "mean"),
        e4_mean_f1=("macro_f1_E4", "mean"),
        mean_delta_f1=("delta_f1", "mean"),
        std_delta_f1=("delta_f1", "std"),
        e0_mean_accuracy=("accuracy_E0", "mean"),
        e4_mean_accuracy=("accuracy_E4", "mean"),
        mean_delta_accuracy=("delta_accuracy", "mean")
    )

    mean_e0_gap = gap_comparison["E0"].mean()
    mean_e4_gap = gap_comparison["E4"].mean()
    mean_delta_gap = gap_comparison["delta_gap"].mean()

    overall_summary["mean_gap_E0"] = mean_e0_gap
    overall_summary["mean_gap_E4"] = mean_e4_gap
    overall_summary["delta_gap"] = mean_delta_gap

    overall_summary.to_csv(
        output_dir / "controlled_overall_summary.csv", index=False
    )

    print("\n" + "=" * 70)
    print("CONTROLLED LOSO — E0 VS E4")
    print("=" * 70)
    print(overall_summary.round(5).to_string(index=False))

    print(f"\nMean Seen-Unseen gap E0: {mean_e0_gap:.4f}")
    print(f"Mean Seen-Unseen gap E4: {mean_e4_gap:.4f}")
    print(f"Change in gap: {mean_delta_gap:+.4f}")

    return results_df, site_summary, overall_summary


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


def apply_site_mi_weighting(X_train, X_eval, train_sites, fraction=0.02, weight=0.50):
    """Downweight the top fraction of Site-MI features using training data only."""

    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1.")
    if not 0 <= weight <= 1:
        raise ValueError("weight must be between 0 and 1.")
    if len(train_sites) != X_train.shape[0]:
        raise ValueError("train_sites must match the number of training rows.")

    X_binary = X_train.copy().tocsr()
    X_binary.data[:] = 1.0

    site_scores = mutual_info_classif(
        X_binary, np.asarray(train_sites),
        discrete_features=True, random_state=42
    )

    n_features = X_train.shape[1]
    n_selected = int(np.ceil(n_features * fraction))

    ranking = np.lexsort((np.arange(n_features), -site_scores))
    selected_idx = ranking[:n_selected]

    feature_weights = np.ones(n_features, dtype=np.float64)
    feature_weights[selected_idx] = weight

    X_train_weighted = X_train.multiply(feature_weights).tocsr()
    X_eval_weighted = X_eval.multiply(feature_weights).tocsr()

    return X_train_weighted, X_eval_weighted, selected_idx, site_scores


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


def E5_frequency_matched_random(cfg, random_seed=42):
    """E5b: Globally frequency-matched alternative feature removal."""

    from scipy.optimize import linear_sum_assignment

    train_df, dev_df, _ = load_and_preprocess(return_df=True)
    research_df = pd.concat([train_df, dev_df], ignore_index=True)

    text_c = cfg.data.text_column
    target_c = cfg.data.target_column
    site_c = cfg.data.site_column
    labels = sorted(research_df[target_c].unique())

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    e2_features_path=Path(cfg.feature_filter.e2_removed_features_path)
    n_repeats=cfg.feature_filter.random_repeats

    e2_features = pd.read_csv(e2_features_path)
    thresholds = [0.01, 0.02, 0.05]
    matching_jitter = 0.01

    results = []
    removed_records = []
    pooled_predictions = {}

    reference_pipeline = create_LR_pipeline(with_SVD=False, OAA=False)
    base_classifier = clone(reference_pipeline.named_steps["clf"])
    base_classifier.set_params(C=1.0)

    def evaluate(X_train, X_eval, y_train, y_eval, removed_idx):
        weights = np.ones(X_train.shape[1], dtype=np.float64)
        weights[removed_idx] = 0.0

        X_tr = X_train.multiply(weights).tocsr()
        X_ev = X_eval.multiply(weights).tocsr()

        classifier = clone(base_classifier)
        classifier.fit(X_tr, y_train)
        predictions = classifier.predict(X_ev)

        macro_f1 = f1_score(
            y_eval, predictions, labels=labels, average="macro", zero_division=0
        )
        accuracy = accuracy_score(y_eval, predictions)

        return float(macro_f1), float(accuracy), predictions

    def globally_matched_indices(e2_idx, doc_frequency, rng):
        """Find a one-to-one minimum-cost matching, with randomized near-ties."""

        all_idx = np.arange(len(doc_frequency))
        available_idx = all_idx[~np.isin(all_idx, e2_idx)]

        if len(available_idx) < len(e2_idx):
            raise ValueError("Not enough alternative features for matching.")

        target_log_df = np.log1p(doc_frequency[e2_idx])
        available_log_df = np.log1p(doc_frequency[available_idx])

        # Rows: E2 features; columns: eligible alternative features
        true_cost = np.abs(
            target_log_df[:, None] - available_log_df[None, :]
        )

        # Random perturbation creates alternative near-optimal matches.
        noise = rng.uniform(
            -matching_jitter, matching_jitter, size=true_cost.shape
        )

        row_idx, col_idx = linear_sum_assignment(true_cost + noise)
        matched_idx = available_idx[col_idx]

        matches = pd.DataFrame({
            "e2_idx": e2_idx[row_idx],
            "random_idx": matched_idx,
            "log_df_distance": true_cost[row_idx, col_idx]
        })

        return matched_idx, matches
    def matching_statistics(matches, doc_frequency):
        e2_df = doc_frequency[matches["e2_idx"].to_numpy()]
        random_df = doc_frequency[matches["random_idx"].to_numpy()]

        ratios = np.maximum(e2_df, random_df) / np.minimum(e2_df, random_df)
        distances = matches["log_df_distance"].to_numpy()

        return {
            "mean_log_df_distance": float(distances.mean()),
            "median_log_df_distance": float(np.median(distances)),
            "p95_log_df_distance": float(np.quantile(distances, 0.95)),
            "mean_df_ratio": float(ratios.mean()),
            "pct_within_20pct": float(100 * np.mean(ratios <= 1.2)),
            "pct_within_50pct": float(100 * np.mean(ratios <= 1.5))
        }

    for site_index, (held_out_site, fold_train, fold_eval) in enumerate(
        create_loso_sets(cfg, research_df)
    ):
        print(f"\n{'=' * 55}")
        print(f"HELD-OUT SITE: {held_out_site}")
        print(f"{'=' * 55}")

        vectorizer = clone(reference_pipeline.named_steps["tfidf"])
        X_train = vectorizer.fit_transform(fold_train[text_c])
        X_eval = vectorizer.transform(fold_eval[text_c])

        y_train = fold_train[target_c].to_numpy()
        y_eval = fold_eval[target_c].to_numpy()

        feature_names = vectorizer.get_feature_names_out()
        feature_lookup = {name: idx for idx, name in enumerate(feature_names)}
        n_features = X_train.shape[1]

        X_binary = X_train.copy().tocsr()
        X_binary.data[:] = 1.0

        doc_frequency = np.asarray(X_binary.getnnz(axis=0)).ravel()

        site_scores = mutual_info_classif(
            X_binary, fold_train[site_c].to_numpy(),
            discrete_features=True, random_state=42
        )

        # E0 baseline
        f1, accuracy, predictions = evaluate(
            X_train, X_eval, y_train, y_eval, np.array([], dtype=int)
        )

        results.append({
            "method": "E0", "held_out_site": held_out_site,
            "threshold": 0.0, "repeat": -1, "n_removed": 0,
            "macro_f1": f1, "accuracy": accuracy,
            "mean_site_mi": np.nan,
            "mean_log_df_distance": np.nan
        })

        pooled_predictions[("E0", 0.0, -1, held_out_site)] = (y_eval, predictions)
        print(f"E0: Macro-F1={f1:.4f}")

        for threshold in thresholds:
            e2_rows = e2_features[
                (e2_features["held_out_site"] == held_out_site)
                & np.isclose(e2_features["threshold"], threshold)
            ].sort_values("rank")

            n_remove = int(np.ceil(n_features * threshold))

            if len(e2_rows) != n_remove:
                raise ValueError(
                    f"{held_out_site}, {threshold}: Expected {n_remove} "
                    f"E2 features, found {len(e2_rows)}."
                )

            missing = set(e2_rows["feature"]) - set(feature_lookup)
            if missing:
                raise ValueError(
                    f"Vocabulary mismatch for {held_out_site}: {sorted(missing)[:10]}"
                )

            e2_idx = np.array(
                [feature_lookup[name] for name in e2_rows["feature"]], dtype=int
            )

            if not np.array_equal(
                doc_frequency[e2_idx], e2_rows["document_frequency"].to_numpy()
            ):
                raise ValueError(
                    f"Document-frequency mismatch for {held_out_site}. "
                    "Check preprocessing, splits and TF-IDF settings."
                )

            # Verify that E2's Site MI scores are reproducible
            if not np.allclose(
                site_scores[e2_idx], e2_rows["site_mi"].to_numpy(),
                atol=1e-7, rtol=1e-5
            ):
                raise ValueError(f"Site MI mismatch for {held_out_site}.")

            # E2 reference
            f1, accuracy, predictions = evaluate(
                X_train, X_eval, y_train, y_eval, e2_idx
            )

            results.append({
                "method": "E2", "held_out_site": held_out_site,
                "threshold": threshold, "repeat": -1,
                "n_removed": n_remove, "macro_f1": f1,
                "accuracy": accuracy,
                "mean_site_mi": float(site_scores[e2_idx].mean()),
                "mean_log_df_distance": 0.0
            })

            pooled_predictions[("E2", threshold, -1, held_out_site)] = (
                y_eval, predictions
            )

            print(f"E2 {threshold:.0%}: Macro-F1={f1:.4f}")

            random_scores = []
            match_distances = []

            for repeat in range(n_repeats):
                seed = (
                    random_seed + site_index * 100000
                    + int(round(threshold * 1000)) * 1000 + repeat
                )
                rng = np.random.default_rng(seed)

                random_idx, matches = globally_matched_indices(
                    e2_idx, doc_frequency, rng
                )

                assert len(np.unique(random_idx)) == n_remove
                assert not np.isin(random_idx, e2_idx).any()

                match_stats = matching_statistics(matches, doc_frequency)

                f1, accuracy, predictions = evaluate(
                    X_train, X_eval, y_train, y_eval, random_idx
                )

                results.append({
                    "method": "E5b", "held_out_site": held_out_site,
                    "threshold": threshold, "repeat": repeat,
                    "n_removed": n_remove, "macro_f1": f1,
                    "accuracy": accuracy,
                    "mean_site_mi": float(site_scores[random_idx].mean()),
                    **match_stats
                })

                pooled_predictions[("E5b", threshold, repeat, held_out_site)] = (
                    y_eval, predictions
                )

                random_scores.append(f1)
                match_distances.append(match_stats["mean_log_df_distance"])

                for row in matches.itertuples(index=False):
                    removed_records.append({
                        "held_out_site": held_out_site,
                        "threshold": threshold,
                        "repeat": repeat,
                        "e2_feature": feature_names[row.e2_idx],
                        "random_feature": feature_names[row.random_idx],
                        "e2_document_frequency": int(doc_frequency[row.e2_idx]),
                        "random_document_frequency": int(doc_frequency[row.random_idx]),
                        "e2_site_mi": float(site_scores[row.e2_idx]),
                        "random_site_mi": float(site_scores[row.random_idx]),
                        "log_df_distance": float(row.log_df_distance)
                    })

            print(
                f"E5b {threshold:.0%}: Mean F1={np.mean(random_scores):.4f}, "
                f"Std={np.std(random_scores, ddof=1):.4f}, "
                f"Mean match distance={np.mean(match_distances):.4f}"
            )

        pd.DataFrame(results).to_csv(
            output_dir / "e5b_results.csv", index=False
        )
        pd.DataFrame(removed_records).to_csv(
            output_dir / "e5b_random_removed_features.csv", index=False
        )

    results_df = pd.DataFrame(results)
    site_rows = []
    summary_rows = []

    sites = sorted(results_df["held_out_site"].unique())
    baseline_mean = results_df[
        results_df["method"] == "E0"
    ]["macro_f1"].mean()

    for site in sites:
        baseline = results_df[
            (results_df["held_out_site"] == site)
            & (results_df["method"] == "E0")
        ]["macro_f1"].iloc[0]

        for threshold in thresholds:
            subset = results_df[
                (results_df["held_out_site"] == site)
                & (results_df["threshold"] == threshold)
            ]

            e2_row = subset[subset["method"] == "E2"].iloc[0]
            random_rows = subset[subset["method"] == "E5b"]

            site_rows.append({
                "held_out_site": site,
                "threshold": threshold,
                "baseline_f1": baseline,
                "e2_f1": e2_row["macro_f1"],
                "e2_delta_vs_E0": e2_row["macro_f1"] - baseline,
                "random_mean_f1": random_rows["macro_f1"].mean(),
                "random_std_f1": random_rows["macro_f1"].std(ddof=1),
                "random_delta_vs_E0": random_rows["macro_f1"].mean() - baseline,
                "e2_minus_random": e2_row["macro_f1"] - random_rows["macro_f1"].mean(),
                "e2_mean_site_mi": e2_row["mean_site_mi"],
                "random_mean_site_mi": random_rows["mean_site_mi"].mean(),
                "mean_log_df_distance": random_rows["mean_log_df_distance"].mean(),
                "pct_within_20pct": random_rows["pct_within_20pct"].mean(),
                "pct_within_50pct": random_rows["pct_within_50pct"].mean()
            })

    for threshold in thresholds:
        subset = results_df[results_df["threshold"] == threshold]
        e2_rows = subset[subset["method"] == "E2"]
        random_rows = subset[subset["method"] == "E5b"]

        repeat_means = random_rows.groupby("repeat")["macro_f1"].mean()
        e2_mean = e2_rows["macro_f1"].mean()

        e2_pairs = [
            pooled_predictions[("E2", threshold, -1, site)] for site in sites
        ]
        e2_pooled = f1_score(
            np.concatenate([pair[0] for pair in e2_pairs]),
            np.concatenate([pair[1] for pair in e2_pairs]),
            labels=labels, average="macro", zero_division=0
        )

        random_pooled = []
        for repeat in range(n_repeats):
            pairs = [
                pooled_predictions[("E5b", threshold, repeat, site)]
                for site in sites
            ]
            random_pooled.append(f1_score(
                np.concatenate([pair[0] for pair in pairs]),
                np.concatenate([pair[1] for pair in pairs]),
                labels=labels, average="macro", zero_division=0
            ))

        summary_rows.append({
            "threshold": threshold,
            "n_repeats": n_repeats,
            "e0_mean_loso_f1": baseline_mean,
            "e2_mean_loso_f1": e2_mean,
            "random_mean_loso_f1": repeat_means.mean(),
            "random_std_loso_f1": repeat_means.std(ddof=1),
            "e2_minus_random_mean_f1": e2_mean - repeat_means.mean(),
            "e2_pooled_f1": e2_pooled,
            "random_mean_pooled_f1": np.mean(random_pooled),
            "e2_minus_random_pooled_f1": e2_pooled - np.mean(random_pooled),
            "random_fraction_at_least_E2": float((repeat_means >= e2_mean).mean()),
            "e2_mean_site_mi": e2_rows["mean_site_mi"].mean(),
            "random_mean_site_mi": random_rows["mean_site_mi"].mean(),
            "mean_log_df_distance": random_rows["mean_log_df_distance"].mean(),
            "median_log_df_distance": random_rows["median_log_df_distance"].mean(),
            "p95_log_df_distance": random_rows["p95_log_df_distance"].mean(),
            "pct_within_20pct": random_rows["pct_within_20pct"].mean(),
            "pct_within_50pct": random_rows["pct_within_50pct"].mean()
        })

    site_comparison = pd.DataFrame(site_rows)
    summary = pd.DataFrame(summary_rows)

    site_comparison.to_csv(
        output_dir / "e5b_site_comparison.csv", index=False
    )
    summary.to_csv(output_dir / "e5b_summary.csv", index=False)

    print("\n" + "=" * 60)
    print("E5b — GLOBAL FREQUENCY-MATCHED RANDOM CONTROL")
    print("=" * 60)
    print(summary.round(5).to_string(index=False))

    return results_df, site_comparison, summary


def E_seen_feature_comparison(cfg):
    """Seen-site evaluation: E0, E2, E3a and E3b on train/dev."""

    train_df, dev_df, _ = load_and_preprocess(return_df=True)

    text_c = cfg.data.text_column
    target_c = cfg.data.target_column
    site_c = cfg.data.site_column

    labels = sorted(pd.concat([train_df[target_c], dev_df[target_c]]).unique())
    sites = sorted(dev_df[site_c].unique())

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = [0.01, 0.02, 0.05]
    candidate_fraction = 0.10
    protection_fraction = 0.20

    results = []
    predictions_records = []
    removed_records = []

    reference_pipeline = create_LR_pipeline(with_SVD=False, OAA=False)

    vectorizer = clone(reference_pipeline.named_steps["tfidf"])
    classifier_template = clone(reference_pipeline.named_steps["clf"])
    classifier_template.set_params(C=1.0)

    X_train = vectorizer.fit_transform(train_df[text_c])
    X_dev = vectorizer.transform(dev_df[text_c])

    y_train = train_df[target_c].to_numpy()
    y_dev = dev_df[target_c].to_numpy()
    train_sites = train_df[site_c].to_numpy()
    dev_sites = dev_df[site_c].to_numpy()

    n_features = X_train.shape[1]
    feature_names = vectorizer.get_feature_names_out()

    print(f"\nSeen comparison: Train={len(train_df)}, Dev={len(dev_df)}")
    print(f"TF-IDF features: {n_features}")

    # Feature scores use ONLY the original training partition
    X_binary = X_train.copy().tocsr()
    X_binary.data[:] = 1.0

    site_scores = mutual_info_classif(
        X_binary, train_sites, discrete_features=True, random_state=42
    )

    cod_scores = mutual_info_classif(
        X_binary, y_train, discrete_features=True, random_state=42
    )

    doc_frequency = np.asarray(X_binary.getnnz(axis=0)).ravel()

    # E2: Descending Site MI
    site_ranking = np.lexsort((np.arange(n_features), -site_scores))

    # E3a: CoD-first ranking within top 10% Site MI candidates
    n_candidates = max(1, int(np.ceil(n_features * candidate_fraction)))
    candidates = site_ranking[:n_candidates]

    cod_first_order = np.lexsort((
        candidates,
        -site_scores[candidates],
        cod_scores[candidates]
    ))
    e3a_ranking = candidates[cod_first_order]

    # E3b: Protect top 20% CoD-MI candidates, then rank by Site MI
    n_protected = int(np.ceil(n_candidates * protection_fraction))

    cod_ranking = np.lexsort((
        candidates,
        -cod_scores[candidates]
    ))
    protected_idx = candidates[cod_ranking[:n_protected]]
    eligible_idx = candidates[~np.isin(candidates, protected_idx)]

    eligible_order = np.lexsort((
        eligible_idx,
        -site_scores[eligible_idx]
    ))
    e3b_ranking = eligible_idx[eligible_order]

    rankings = {
        "E2": site_ranking,
        "E3a": e3a_ranking,
        "E3b": e3b_ranking
    }

    max_remove = int(np.ceil(n_features * max(thresholds)))
    if any(len(ranking) < max_remove for ranking in rankings.values()):
        raise ValueError("Insufficient candidate features for removal budgets.")

    def evaluate(experiment, threshold, removed_idx):
        weights = np.ones(n_features, dtype=np.float64)
        weights[removed_idx] = 0.0

        X_train_filtered = X_train.multiply(weights).tocsr()
        X_dev_filtered = X_dev.multiply(weights).tocsr()

        classifier = clone(classifier_template)
        classifier.fit(X_train_filtered, y_train)
        predicted = classifier.predict(X_dev_filtered)

        overall_f1 = f1_score(
            y_dev, predicted, labels=labels, average="macro", zero_division=0
        )
        overall_accuracy = accuracy_score(y_dev, predicted)

        print(
            f"{experiment}: Removed={len(removed_idx)}, "
            f"Macro-F1={overall_f1:.4f}, Accuracy={overall_accuracy:.4f}"
        )

        # Overall results
        results.append({
            "experiment": experiment,
            "site": "ALL",
            "threshold": threshold,
            "n_features": n_features,
            "n_removed": len(removed_idx),
            "train_size": len(train_df),
            "eval_size": len(dev_df),
            "C": 1.0,
            "macro_f1": float(overall_f1),
            "accuracy": float(overall_accuracy)
        })

        # Per-site results: one global model evaluated on each familiar site
        for site in sites:
            mask = dev_sites == site

            site_f1 = f1_score(
                y_dev[mask], predicted[mask],
                labels=labels, average="macro", zero_division=0
            )

            site_accuracy = accuracy_score(y_dev[mask], predicted[mask])

            results.append({
                "experiment": experiment,
                "site": site,
                "threshold": threshold,
                "n_features": n_features,
                "n_removed": len(removed_idx),
                "train_size": len(train_df),
                "eval_size": int(mask.sum()),
                "C": 1.0,
                "macro_f1": float(site_f1),
                "accuracy": float(site_accuracy)
            })

        predictions_records.append(pd.DataFrame({
            "experiment": experiment,
            "threshold": threshold,
            "site": dev_sites,
            "module": dev_df["module"].to_numpy(),
            "newid": dev_df["newid"].to_numpy(),
            "true_label": y_dev,
            "predicted_label": predicted
        }))

        # Removed vocabulary and feature information
        for rank, idx in enumerate(removed_idx, start=1):
            removed_records.append({
                "experiment": experiment,
                "threshold": threshold,
                "rank": rank,
                "feature": feature_names[idx],
                "site_mi": float(site_scores[idx]),
                "cod_mi": float(cod_scores[idx]),
                "document_frequency": int(doc_frequency[idx]),
                "document_frequency_ratio": float(
                    doc_frequency[idx] / len(train_df)
                )
            })

    # E0 baseline
    evaluate("E0", 0.0, np.array([], dtype=int))

    # E2, E3a and E3b at equal removal budgets
    for method, ranking in rankings.items():
        print(f"\n{'=' * 50}\n{method}\n{'=' * 50}")

        for threshold in thresholds:
            n_remove = max(1, int(np.ceil(n_features * threshold)))
            experiment = f"{method}_{int(round(threshold * 100))}pct"
            removed_idx = ranking[:n_remove]

            evaluate(experiment, threshold, removed_idx)

        # Incremental saving
        pd.DataFrame(results).to_csv(
            output_dir / "seen_results.csv", index=False
        )
        pd.concat(predictions_records, ignore_index=True).to_csv(
            output_dir / "seen_predictions.csv", index=False
        )
        pd.DataFrame(removed_records).to_csv(
            output_dir / "seen_removed_features.csv", index=False
        )

    # Paired differences versus E0, separately for each site
    results_df = pd.DataFrame(results)

    baseline = results_df[results_df["experiment"] == "E0"][
        ["site", "macro_f1", "accuracy"]
    ].rename(columns={
        "macro_f1": "baseline_f1",
        "accuracy": "baseline_accuracy"
    })

    results_df = results_df.merge(
        baseline, on="site", how="left", validate="many_to_one"
    )

    results_df["delta_f1_vs_E0"] = (
        results_df["macro_f1"] - results_df["baseline_f1"]
    )
    results_df["delta_accuracy_vs_E0"] = (
        results_df["accuracy"] - results_df["baseline_accuracy"]
    )

    # Summary: global dev score and equal-weight mean across six sites
    summary_rows = []

    for experiment, group in results_df.groupby("experiment", sort=False):
        overall = group[group["site"] == "ALL"].iloc[0]
        per_site = group[group["site"] != "ALL"]

        summary_rows.append({
            "experiment": experiment,
            "threshold": float(overall["threshold"]),
            "overall_macro_f1": float(overall["macro_f1"]),
            "overall_accuracy": float(overall["accuracy"]),
            "delta_overall_f1_vs_E0": float(overall["delta_f1_vs_E0"]),
            "mean_site_macro_f1": float(per_site["macro_f1"].mean()),
            "median_site_macro_f1": float(per_site["macro_f1"].median()),
            "std_site_macro_f1": float(per_site["macro_f1"].std(ddof=1)),
            "delta_mean_site_f1_vs_E0": float(per_site["delta_f1_vs_E0"].mean()),
            "n_sites_improved": int((per_site["delta_f1_vs_E0"] > 0).sum()),
            "n_sites_degraded": int((per_site["delta_f1_vs_E0"] < 0).sum())
        })

    summary = pd.DataFrame(summary_rows)

    results_df.to_csv(output_dir / "seen_results.csv", index=False)
    summary.to_csv(output_dir / "seen_summary.csv", index=False)

    print("\n" + "=" * 60)
    print("SEEN — FINAL SUMMARY")
    print("=" * 60)
    print(summary.round(5).to_string(index=False))

    return results_df, summary


def E4_feature_downweighting(cfg):
    """E4: Compare feature removal and downweighting for Seen and Unseen sites."""

    train_df, dev_df, _ = load_and_preprocess(return_df=True)
    research_df = pd.concat([train_df, dev_df], ignore_index=True)

    text_c = cfg.data.text_column
    target_c = cfg.data.target_column
    site_c = cfg.data.site_column

    labels = sorted(research_df[target_c].unique())
    thresholds = [0.01, 0.02, 0.05]
    weights_to_test = [0.0, 0.25, 0.50]

    candidate_fraction = 0.10
    protection_fraction = 0.20

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    prediction_frames = []
    feature_records = []

    reference_pipeline = create_LR_pipeline(with_SVD=False, OAA=False)

    classifier_template = clone(reference_pipeline.named_steps["clf"])
    classifier_template.set_params(C=1.0)

    def evaluate_fold(mode, held_out_site, fold_train, fold_eval):
        print(f"\n{'=' * 60}")
        print(f"E4 — {mode}: {held_out_site}")
        print(f"{'=' * 60}")

        vectorizer = clone(reference_pipeline.named_steps["tfidf"])
        X_train = vectorizer.fit_transform(fold_train[text_c])
        X_eval = vectorizer.transform(fold_eval[text_c])

        y_train = fold_train[target_c].to_numpy()
        y_eval = fold_eval[target_c].to_numpy()
        sites_train = fold_train[site_c].to_numpy()
        sites_eval = fold_eval[site_c].to_numpy()

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

        # E2: highest Site MI first
        site_ranking = np.lexsort((np.arange(n_features), -site_scores))

        # E3b: top 10% Site MI, protect top 20% CoD MI
        n_candidates = max(1, int(np.ceil(n_features * candidate_fraction)))
        candidates = site_ranking[:n_candidates]

        n_protected = int(np.ceil(n_candidates * protection_fraction))
        cod_order = np.lexsort((candidates, -cod_scores[candidates]))
        protected_idx = candidates[cod_order[:n_protected]]

        eligible_idx = candidates[~np.isin(candidates, protected_idx)]
        eligible_order = np.lexsort((eligible_idx, -site_scores[eligible_idx]))
        e3b_ranking = eligible_idx[eligible_order]

        rankings = {"E2": site_ranking, "E3b": e3b_ranking}

        max_remove = int(np.ceil(n_features * max(thresholds)))
        if any(len(ranking) < max_remove for ranking in rankings.values()):
            raise ValueError("Not enough eligible features for E4 removal budgets.")

        print(
            f"Train={len(fold_train)}, Eval={len(fold_eval)}, "
            f"Features={n_features}"
        )

        def fit_and_record(method, threshold, weight, selected_idx):
            feature_weights = np.ones(n_features, dtype=np.float64)
            feature_weights[selected_idx] = weight

            X_tr = X_train.multiply(feature_weights).tocsr()
            X_ev = X_eval.multiply(feature_weights).tocsr()

            classifier = clone(classifier_template)
            classifier.fit(X_tr, y_train)
            predicted = classifier.predict(X_ev)

            experiment = (
                "E0" if method == "E0"
                else f"{method}_{int(round(threshold * 100))}pct_w{weight:.2f}"
            )

            prediction_frames.append(pd.DataFrame({
                "mode": mode,
                "held_out_site": held_out_site,
                "experiment": experiment,
                "method": method,
                "threshold": threshold,
                "weight": weight,
                "site": sites_eval,
                "module": fold_eval["module"].to_numpy(),
                "newid": fold_eval["newid"].to_numpy(),
                "true_label": y_eval,
                "predicted_label": predicted
            }))

            # For Seen: ALL plus a separate score for each familiar site
            # For Unseen: evaluate the single held-out geographical site
            evaluation_groups = (
                [("ALL", np.ones(len(y_eval), dtype=bool))]
                + [(site, sites_eval == site) for site in sorted(np.unique(sites_eval))]
                if mode == "Seen"
                else [(held_out_site, np.ones(len(y_eval), dtype=bool))]
            )

            for eval_site, mask in evaluation_groups:
                macro_f1 = f1_score(
                    y_eval[mask], predicted[mask],
                    labels=labels, average="macro", zero_division=0
                )
                accuracy = accuracy_score(y_eval[mask], predicted[mask])

                results.append({
                    "mode": mode,
                    "held_out_site": held_out_site,
                    "eval_site": eval_site,
                    "experiment": experiment,
                    "method": method,
                    "threshold": threshold,
                    "weight": weight,
                    "n_features": n_features,
                    "n_selected": len(selected_idx),
                    "train_size": len(fold_train),
                    "eval_size": int(mask.sum()),
                    "C": 1.0,
                    "macro_f1": float(macro_f1),
                    "accuracy": float(accuracy)
                })

                if eval_site == "ALL" or mode == "Unseen":
                    print(
                        f"{experiment}: Macro-F1={macro_f1:.4f}, "
                        f"Accuracy={accuracy:.4f}"
                    )

        # Baseline E0
        fit_and_record("E0", 0.0, 1.0, np.array([], dtype=int))

        # E2 and E3b, with identical feature sets across weights
        for method, ranking in rankings.items():
            for threshold in thresholds:
                n_select = max(1, int(np.ceil(n_features * threshold)))
                selected_idx = ranking[:n_select]

                for rank, idx in enumerate(selected_idx, start=1):
                    feature_records.append({
                        "mode": mode,
                        "held_out_site": held_out_site,
                        "method": method,
                        "threshold": threshold,
                        "rank": rank,
                        "feature": feature_names[idx],
                        "site_mi": float(site_scores[idx]),
                        "cod_mi": float(cod_scores[idx]),
                        "document_frequency": int(doc_frequency[idx])
                    })

                for weight in weights_to_test:
                    fit_and_record(method, threshold, weight, selected_idx)

        # Incremental saving after each complete fold
        pd.DataFrame(results).to_csv(
            output_dir / "e4_results.csv", index=False
        )
        pd.concat(prediction_frames, ignore_index=True).to_csv(
            output_dir / "e4_predictions.csv", index=False
        )
        pd.DataFrame(feature_records).to_csv(
            output_dir / "e4_selected_features.csv", index=False
        )

    # Six Unseen folds
    for held_out_site, fold_train, fold_eval in create_loso_sets(cfg, research_df):
        evaluate_fold("Unseen", held_out_site, fold_train, fold_eval)

    # One Seen evaluation: original train/dev partition
    evaluate_fold("Seen", "ALL", train_df, dev_df)

    results_df = pd.DataFrame(results)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)

    # Paired differences against E0 within each mode and evaluation site
    baseline = results_df[results_df["method"] == "E0"][
        ["mode", "eval_site", "macro_f1", "accuracy"]
    ].rename(columns={
        "macro_f1": "baseline_f1",
        "accuracy": "baseline_accuracy"
    })

    results_df = results_df.merge(
        baseline, on=["mode", "eval_site"], how="left", validate="many_to_one"
    )

    results_df["delta_f1_vs_E0"] = results_df["macro_f1"] - results_df["baseline_f1"]
    results_df["delta_accuracy_vs_E0"] = (
        results_df["accuracy"] - results_df["baseline_accuracy"]
    )

    summary_rows = []

    for mode in ["Unseen", "Seen"]:
        mode_results = results_df[results_df["mode"] == mode]
        mode_predictions = predictions_df[predictions_df["mode"] == mode]

        for experiment, group in mode_results.groupby("experiment", sort=False):
            if mode == "Seen":
                overall = group[group["eval_site"] == "ALL"].iloc[0]
                site_group = group[group["eval_site"] != "ALL"]
                overall_f1 = float(overall["macro_f1"])
                overall_accuracy = float(overall["accuracy"])
            else:
                site_group = group
                experiment_predictions = mode_predictions[
                    mode_predictions["experiment"] == experiment
                ]

                overall_f1 = float(f1_score(
                    experiment_predictions["true_label"],
                    experiment_predictions["predicted_label"],
                    labels=labels, average="macro", zero_division=0
                ))
                overall_accuracy = float(accuracy_score(
                    experiment_predictions["true_label"],
                    experiment_predictions["predicted_label"]
                ))

            summary_rows.append({
                "mode": mode,
                "experiment": experiment,
                "method": group["method"].iloc[0],
                "threshold": float(group["threshold"].iloc[0]),
                "weight": float(group["weight"].iloc[0]),
                "mean_site_macro_f1": float(site_group["macro_f1"].mean()),
                "median_site_macro_f1": float(site_group["macro_f1"].median()),
                "std_site_macro_f1": float(site_group["macro_f1"].std(ddof=1)),
                "pooled_macro_f1": overall_f1,
                "pooled_accuracy": overall_accuracy,
                "mean_site_accuracy": float(site_group["accuracy"].mean()),
                "n_sites_improved": int((site_group["delta_f1_vs_E0"] > 0).sum()),
                "n_sites_degraded": int((site_group["delta_f1_vs_E0"] < 0).sum()),
                "n_sites": len(site_group)
            })

    summary = pd.DataFrame(summary_rows)

    # Calculate baseline differences separately for Seen and Unseen
    for mode in ["Seen", "Unseen"]:
        mask = summary["mode"] == mode
        baseline_row = summary[mask & (summary["method"] == "E0")].iloc[0]

        summary.loc[mask, "delta_mean_site_f1_vs_E0"] = (
            summary.loc[mask, "mean_site_macro_f1"]
            - baseline_row["mean_site_macro_f1"]
        )

        summary.loc[mask, "delta_pooled_f1_vs_E0"] = (
            summary.loc[mask, "pooled_macro_f1"]
            - baseline_row["pooled_macro_f1"]
        )

        summary.loc[mask, "delta_pooled_accuracy_vs_E0"] = (
            summary.loc[mask, "pooled_accuracy"]
            - baseline_row["pooled_accuracy"]
        )

    # Compare downweighting directly against complete removal
    removal_baselines = summary[
        (summary["method"] != "E0") & (summary["weight"] == 0.0)
    ][[
        "mode", "method", "threshold",
        "mean_site_macro_f1", "pooled_macro_f1"
    ]].rename(columns={
        "mean_site_macro_f1": "removal_mean_site_f1",
        "pooled_macro_f1": "removal_pooled_f1"
    })

    summary = summary.merge(
        removal_baselines, on=["mode", "method", "threshold"],
        how="left", validate="many_to_one"
    )

    summary["delta_mean_site_f1_vs_removal"] = (
        summary["mean_site_macro_f1"] - summary["removal_mean_site_f1"]
    )
    summary["delta_pooled_f1_vs_removal"] = (
        summary["pooled_macro_f1"] - summary["removal_pooled_f1"]
    )

    # Save final CSV files
    results_df.to_csv(output_dir / "e4_results.csv", index=False)
    summary.to_csv(output_dir / "e4_summary.csv", index=False)

    print("\n" + "=" * 75)
    print("E4 — FINAL SEEN / UNSEEN SUMMARY")
    print("=" * 75)

    display_columns = [
        "mode", "experiment", "mean_site_macro_f1",
        "delta_mean_site_f1_vs_E0", "pooled_macro_f1",
        "delta_pooled_f1_vs_E0", "delta_pooled_f1_vs_removal"
    ]

    print(summary[display_columns].round(5).to_string(index=False))

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
    elif exp == "E5_frequency_matched_random":
        E5_frequency_matched_random(cfg)
    elif exp == "E_seen_feature_comparison":
        E_seen_feature_comparison(cfg)
    elif exp == "E4_feature_downweighting":
        E4_feature_downweighting(cfg)
    elif exp == "loso_LR_controlled_E4":
        run_controlled_loso_experiment(cfg, apply_site_weighting=True)
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