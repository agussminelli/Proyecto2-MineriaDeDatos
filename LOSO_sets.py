import pandas as pd
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
