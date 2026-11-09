
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from loading_dataset import load_dataset
from preprocess import preprocess_dataset
from random_forest import RandomForest
from simple_load_dataset import simple_load_dataset


def main():

    # 1. Load dataset
    train, _, train_headers, _, _ = simple_load_dataset()
    df = pd.DataFrame(train, columns=train_headers)

    print("Columnas del dataset:")
    print(df.columns.tolist())

    df = pd.DataFrame(train, columns=train_headers)

    # 2. Preprocess text
    df = preprocess_dataset(
        df=df,
        text_cols=["open_response"],
        download=False
    )

    # 3. Select text and labels
   # 3. Select text and labels
    df = df[
        (df["open_response_clean"].notna()) &
        (df["open_response_clean"].str.strip() != "") &
        (df["gs_text34"].notna()) &
        (df["gs_text34"].str.strip() != "")
    ].copy()

    X = df["open_response_clean"]
    y = df["gs_text34"]
    # 4. Split train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 5. TF-IDF vectorization USE A RANDOM ONE, WE SHOULD CHANGE IT LATER
    vectorizer = TfidfVectorizer(
        max_features=5000
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    # 6. Train Random Forest
    model = RandomForest()
    model.fit(X_train_vec, y_train)

    # 7. Evaluate
    y_pred = model.predict(X_val_vec)

    print("Accuracy:", accuracy_score(y_val, y_pred))
    print("\nClassification report:")
    print(classification_report(y_val, y_pred, zero_division=0))


if __name__ == "__main__":
    main()
