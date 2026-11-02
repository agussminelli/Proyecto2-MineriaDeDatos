
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from loading_dataset import load_dataset
from preprocess import preprocess_dataset
from randomForest import random_forest


# 1. Load dataset
train, _, train_headers, _, _ = load_dataset()

train_df = pd.DataFrame(train, columns=train_headers)


# 2. Preprocess text
processed_df = preprocess_dataset(
    df=train_df,
    text_cols=["open_response"],
    download=False
)


# 3. Select text and labels
processed_df = processed_df[
    (processed_df["open_response_clean"].str.strip() != "") &
    (processed_df["gs_grouped"].str.strip() != "")
]

X = processed_df["open_response_clean"]
y = processed_df["gs_grouped"]


# 4. Temporary train/dev split
X_train, X_dev, y_train, y_dev = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 5. Vectorization THIS SHOULD BE OURS, JUST TO TEST
vectorizer = TfidfVectorizer(max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_dev_vec = vectorizer.transform(X_dev)


# 6. Random Forest
rf = randomForest.RandomForest(n_estimators=100, max_depth=None, random_state=42)
rf.fit(X_train_vec, y_train)



# 7. Predictions and evaluation
y_pred = rf.predict(X_dev_vec)

print(classification_report(
    y_dev,
    y_pred,
    zero_division=0
))
