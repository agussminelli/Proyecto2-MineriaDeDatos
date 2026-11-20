from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, GroupKFold

import pandas as pd
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from simple_load_dataset import simple_load_dataset
from divide_data import create_stratified_partitions
from preprocess import preprocess_dataset
from LOSO_sets import create_loso_sets


def load_and_preprocess(test=False, return_df=False):

    train_data, _, headers, _, _ = simple_load_dataset()

    train_df, dev_df, test_df = create_stratified_partitions(
        raw_data=train_data, headers=headers, target_col="gs_text34"
    )

    # 2. Preprocess
    train_df = preprocess_dataset(train_df, text_cols=["open_response"], download=False)
    dev_df = preprocess_dataset(dev_df, text_cols=["open_response"], download=False)

    X_train, y_train = train_df["open_response_clean"], train_df["gs_text34"]
    X_dev, y_dev = dev_df["open_response_clean"], dev_df["gs_text34"]

    if test:
        test_df = preprocess_dataset(test_df, text_cols=["open_response"], download=False)
        X_test, y_test = test_df["open_response_clean"], test_df["gs_text34"]
        if return_df: return train_df, dev_df, test_df
        return X_train, y_train, X_dev, y_dev, X_test, y_test
    if return_df:
        return train_df, dev_df, None

    return X_train, y_train, X_dev, y_dev, None, None


def main():
    X_train, y_train, X_dev, y_dev,_,_ = load_and_preprocess()

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 3), max_df=0.85)),
        ('svd', TruncatedSVD(n_components=300, random_state=42)),
        ("clf", LogisticRegression(C=1.0,solver="lbfgs",max_iter=2000,class_weight="balanced")),
    ])

    pipeline_oaa = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 3), max_df=0.85)),
        ('svd', TruncatedSVD(n_components=300, random_state=42)),
        ("clf", OneVsRestClassifier(LogisticRegression(C=1.0,solver="lbfgs",max_iter=2000,class_weight="balanced"))),
    ])


    for name, model in [
        ("Multinomial LR", pipeline),
        ("OAA LR", pipeline_oaa)
    ]:
        model.fit(X_train,y_train)
        predictions = model.predict(X_dev)

        print(f"\n{name}")
        print("Macro-F1:", f1_score(
            y_dev, predictions,
            average="macro",
            zero_division=0
        ))
        print(classification_report(
            y_dev, predictions,
            zero_division=0
        ))

def search_params(LR=True, OAA=True, verbose=True):
    X_train, y_train, X_dev, y_dev,_,_ = load_and_preprocess()

    cv = StratifiedKFold(n_splits=3,shuffle=True,random_state=42)
    if LR:
        param_grid = {
            "clf__C":[0.01, 0.1, 1, 10, 100]
        }
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 3), max_df=0.85)),
            ('svd', TruncatedSVD(n_components=300, random_state=42)),
            ("clf", LogisticRegression(C=1.0,solver="lbfgs",max_iter=2000,class_weight="balanced")),
        ])
        search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1
        )
        search.fit(X_train, y_train)
        if verbose:
            print("LR")
            print("Best C:", search.best_params_)
            print("CV Macro-F1:", search.best_score_)
            print(
                "Dev Macro-F1:",
                f1_score(
                    y_dev,
                    search.predict(X_dev),
                    average="macro",
                    zero_division=0
                )
            )
    if OAA:
        param_grid_oaa = {
            "clf__estimator__C": [0.01, 0.1, 1, 10, 100]
        }
        pipeline_oaa = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 3), max_df=0.85)),
            ('svd', TruncatedSVD(n_components=300, random_state=42)),
            ("clf", OneVsRestClassifier(LogisticRegression(C=1.0,solver="lbfgs",max_iter=2000,class_weight="balanced"))),
        ])
        search_oaa = GridSearchCV(
            pipeline_oaa,
            param_grid=param_grid_oaa,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1
        )
        search_oaa.fit(X_train, y_train)

        if verbose:
            print("OAA")
            print("Best C:", search_oaa.best_params_)
            print("CV Macro-F1:", search_oaa.best_score_)
            print(
                "Dev Macro-F1:",
                f1_score(
                    y_dev,
                    search_oaa.predict(X_dev),
                    average="macro",
                    zero_division=0
                )
            )

    if LR and OAA:
        return search, search_oaa
    

def compare_performance_on_diff_sites():
    import pandas as pd
    from sklearn.metrics import f1_score, accuracy_score

    train_df, dev_df,_ = load_and_preprocess(return_df=True)
    X_train, y_train = train_df["open_response_clean"], train_df["gs_text34"]
    X_dev, y_dev = dev_df["open_response_clean"], dev_df["gs_text34"]

    search_lr, search_oaa = search_params()

    models = {
        "Multinomial LR": search_lr.best_estimator_,
        "OAA LR": search_oaa.best_estimator_
    }

    results = []

    for name, model in models.items():
        predictions = model.predict(X_dev)

        for site in dev_df["site"].unique():
            mask = dev_df["site"].to_numpy() == site

            results.append({
                "model": name,
                "site": site,
                "n": mask.sum(),
                "accuracy": accuracy_score(
                    y_dev[mask], predictions[mask]
                ),
                "macro_f1": f1_score(
                    y_dev[mask], predictions[mask],
                    average="macro",
                    zero_division=0
                )
            })

    results_df = pd.DataFrame(results)

    print(results_df.pivot(
        index="site",
        columns="model",
        values="macro_f1"
    ).round(3))


def experiment_without_svd():
    X_train, y_train, X_dev, y_dev,_,_ = load_and_preprocess()
    models = {
        "Multinomial LR": (
            LogisticRegression(solver="lbfgs",max_iter=2000,class_weight="balanced"),
            "clf__C"
        ),
        "OAA LR": (
            OneVsRestClassifier(
                LogisticRegression(solver="lbfgs",max_iter=2000,class_weight="balanced")
            ),
            "clf__estimator__C"
        )
    }

    cv = StratifiedKFold(n_splits=3,shuffle=True,random_state=42)

    for name, (classifier, c_param) in models.items():

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000,ngram_range=(1, 3))),
            ("clf", classifier)
        ])

        search = GridSearchCV(
            pipeline,
            param_grid={c_param: [0.01, 0.1, 1, 10, 100]},
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1
        )

        search.fit(X_train, y_train)

        predictions = search.predict(X_dev)

        print(f"\n{name} — WITHOUT SVD")
        print("Best parameters:", search.best_params_)
        print("CV Macro-F1:", search.best_score_)
        print(
            "Dev Macro-F1:",
            f1_score(
                y_dev,
                predictions,
                average="macro",
                zero_division=0
            )
        )


def create_LR_pipeline(cfg=None, with_SVD=True, OAA=True):

    if cfg is not None:
        max_features=cfg.vectorizer.max_features
        ngram_range=cfg.vectorizer.ngram_range
        n_components=cfg.svd.n_components
        random_state=cfg.svd.random_state
        max_iter=cfg.classifier.max_iter
        class_weight=cfg.classifier.class_weight
        C=1.0 #Default, Grid search will overwrite 
    else:
        max_features=5000
        ngram_range=(1,3)
        n_components=300
        random_state=42
        max_iter=2000
        class_weight="balanced"
        # Previously selected hyperparameters
        C = 1 if OAA and with_SVD else 10

    classifier = LogisticRegression(C=C,max_iter=max_iter,class_weight=class_weight, solver="lbfgs")
    if OAA:
        classifier = OneVsRestClassifier(classifier)

    steps = [("tfidf", TfidfVectorizer(max_features=max_features,ngram_range=ngram_range))]
    if with_SVD:
        steps.append(("svd",TruncatedSVD(n_components=n_components,random_state=random_state)))
    steps.append(("clf", classifier))

    return Pipeline(steps)


@hydra.main(version_base=None,config_path="conf",config_name="config")
def run_loso_experiments(cfg: DictConfig):
    train_df, dev_df,_ = load_and_preprocess(return_df=True)    
    exp_combination=[(True,True),
                     (True,False),
                     (False,True),
                     (False,False)]
    adult_only=cfg.experiment.adult_only
    results = []
    all_predictions = []

    output_dir = Path(HydraConfig.get().runtime.output_dir)

    for with_SVD, OAA in exp_combination:
        print(f"\nWith SVD={with_SVD}, OAA={OAA}")
        for site, fold_train, fold_eval in create_loso_sets(
            train_df, dev_df, adult_only
        ):
            text_c=cfg.data.text_column
            target_c=cfg.data.target_column
            site_c=cfg.data.site_column

            # Important: create a fresh model for each fold!
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
                "true_label": fold_eval["gs_text34"].to_numpy(),
                "predicted_label": predictions,
                "with_SVD": with_SVD,
                "OAA": OAA
            })
            all_predictions.append(prediction_df)

            f1 = f1_score(fold_eval[target_c],predictions,average="macro",zero_division=0)

            result = {
                "site": site,
                "adult_only": adult_only,
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
    # main()
    # search_params()
    # compare_performance_on_diff_sites()
    # experiment_without_svd()

    results = run_loso_experiments()

# # All modules
# python logistic_regression.py experiment.adult_only=false

# # Adult only
# python loso_experiments.py experiment.adult_only=true

# # Different regularization search
# python loso_experiments.py classifier.c_values=[0.1,1,10]