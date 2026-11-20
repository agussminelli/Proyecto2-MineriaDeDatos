import numpy as np
import pandas as pd
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import f1_score, accuracy_score

from LOSO_sets import create_loso_sets
from logistic_regression import load_and_preprocess, create_LR_pipeline
from train_baseline import create_SVM_pipeline



@hydra.main(version_base=None,config_path="conf",config_name="config")
def run_loso_experiments(cfg: DictConfig):

    adult_only=cfg.experiment.adult_only
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
                pipeline = create_LR_pipeline(with_SVD=with_SVD,OAA=OAA)
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

if __name__ == "__main__":
    run_loso_experiments()

    
    # # All modules
    # py loso_experiments.py experiment.adult_only=false

    # # Adult only
    # py loso_experiments.py experiment.adult_only=true

    # # Different regularization search
    # py loso_experiments.py classifier.c_values=[0.1,1,10]