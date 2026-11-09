import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

from random_forest import RandomForest
from simple_load_dataset import simple_load_dataset
from divide_data import create_stratified_partitions
from preprocess import preprocess_dataset

def run_tfidf_svm_baseline():
    # 1. Load raw data
    train, test, train_headers, test_headers, _ = simple_load_dataset(
        dataset_dir="/Users/frejaschon/Downloads/DATA_MDD"
    ) 
    
    # 2. Get the stratified 70/20/10 split as DataFrames
    train_df, dev_df, test_df = create_stratified_partitions(
        raw_data=train,
        headers=train_headers,
        target_col="gs_text34"
    )

    # 3. Apply heavy NLP preprocessing for TF-IDF
    print("\nApplying preprocessing for TF-IDF...")
    train_df = preprocess_dataset(train_df, text_cols=["open_response"], download=False)
    dev_df   = preprocess_dataset(dev_df, text_cols=["open_response"], download=False)
    test_df  = preprocess_dataset(test_df, text_cols=["open_response"], download=False)

    # 4. TF-IDF Vectorization
    print("Fitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    
    X_train_vec = vectorizer.fit_transform(train_df["open_response_clean"])
    X_dev_vec   = vectorizer.transform(dev_df["open_response_clean"])
    X_test_vec  = vectorizer.transform(test_df["open_response_clean"])

    # 5. Train Linear SVM with Balanced Class Weights
    print("Training Linear SVM Baseline...")
    #svm_model = LinearSVC(class_weight="balanced", random_state=42, max_iter=2000)
    svm_model = LinearSVC(random_state=42, max_iter=2000)
    svm_model.fit(X_train_vec, train_df["gs_text34"])

    # 6. Evaluate on Test Set
    print("\n--- TF-IDF + SVM Baseline Results on TEST Set ---")
    y_pred_test = svm_model.predict(X_test_vec)
    
    print(f"Accuracy: {accuracy_score(test_df['gs_text34'], y_pred_test):.4f}\n")
    print(classification_report(test_df["gs_text34"], y_pred_test, zero_division=0))

    model = RandomForest()
    y_train = train_df["gs_text34"]
    y_val = dev_df["gs_text34"]
    model.fit(X_train_vec, y_train)

    # 7. Evaluate
    y_pred = model.predict(X_dev_vec)

    print("Accuracy:", accuracy_score(y_val, y_pred))
    print("\nClassification report:")
    print(classification_report(y_val, y_pred, zero_division=0))

if __name__ == "__main__":
    run_tfidf_svm_baseline()