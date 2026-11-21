import numpy as np
import pandas as pd
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import f1_score, accuracy_score

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