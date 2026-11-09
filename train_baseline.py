import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
from simple_load_dataset import simple_load_dataset
from divide_data import create_stratified_partitions
from preprocess import preprocess_dataset

def train_baseline():
    print("--- Training Baseline: TF-IDF + SVM ---")
    
    # 1. Load and Partition Data 
    train_data, _, headers, _, _ = simple_load_dataset(
        dataset_dir="/Users/frejaschon/Downloads/DATA_MDD"
    )
    
    train_df, dev_df, test_df = create_stratified_partitions(
        raw_data=train_data, headers=headers, target_col="gs_text34"
    )

    # 2. Preprocess
    train_df = preprocess_dataset(train_df, text_cols=["open_response"], download=False)
    dev_df = preprocess_dataset(dev_df, text_cols=["open_response"], download=False)

    X_train, y_train = train_df["open_response_clean"], train_df["gs_text34"]
    X_dev, y_dev = dev_df["open_response_clean"], dev_df["gs_text34"]

    # 3. Build and Train Pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 3), max_df=0.85)),
        ('svd', TruncatedSVD(n_components=300, random_state=42)),
        ('svm', LinearSVC(class_weight="balanced", max_iter=2000, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # 4. Evaluate on Dev Partition
    print("\n--- Dev Set Evaluation ---")
    y_pred = pipeline.predict(X_dev)
    print(f"Accuracy: {accuracy_score(y_dev, y_pred):.4f}")
    print(f"Macro F1: {f1_score(y_dev, y_pred, average='macro'):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_dev, y_pred, zero_division=0))

    # 5. Save
    save_path = "baseline_tfidf_svm.pkl"
    joblib.dump(pipeline, save_path)
    print(f"\nModel saved successfully to {save_path}")

if __name__ == "__main__":
    train_baseline()