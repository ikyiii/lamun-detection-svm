import os
import joblib
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from extract_features import extract_features

def build_dataset():
    """Build dataset from train/valid/test folders"""
    splits = ['train', 'valid', 'test']
    dfs = []
    
    for split in splits:
        csv_path = f"dataset/{split}/_annotations.csv"
        if not os.path.exists(csv_path):
            print(f"Missing {split} annotations file")
            return None, None, None
        dfs.append(pd.read_csv(csv_path))
    
    df_combined = pd.concat(dfs)
    features = []
    labels = []
    
    for _, row in df_combined.iterrows():
        img_path = None
        for split in splits:
            possible_path = f"dataset/{split}/images/{row['filename']}"
            if os.path.exists(possible_path):
                img_path = possible_path
                break
                
        if img_path is None:
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        feat = extract_features(img)
        if feat is not None:
            features.append(feat)
            labels.append(row['class'])
    
    if not features:
        return None, None, None
        
    scaler = StandardScaler()
    return scaler.fit_transform(features), np.array(labels), scaler

def save_training_log(model, X_test, y_test, X_train, y_train, class_dist):
    """Save training log"""
    import datetime
    import pandas as pd
    from sklearn.metrics import classification_report
    
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    log_data = {
        "timestamp": timestamp,
        "model_type": "SVM",
        "best_params": str(model.best_params_),
        "accuracy": report['accuracy'],
        "weighted_precision": report['weighted avg']['precision'],
        "weighted_recall": report['weighted avg']['recall'],
        "weighted_f1": report['weighted avg']['f1-score'],
        "total_samples": len(y_test) + len(y_train),
        **{f"class_{k}": v for k,v in class_dist.items()}
    }
    
    log_df = pd.DataFrame([log_data])
    log_file = "logs/training_logs.csv"
    
    if os.path.exists(log_file):
        existing_logs = pd.read_csv(log_file)
        log_df = pd.concat([existing_logs, log_df], ignore_index=True)
    
    log_df.to_csv(log_file, index=False)
    
    print("\n=== Training Log ===")
    print(f"Timestamp: {timestamp}")
    print(f"Accuracy: {report['accuracy']:.2f}")
    print(f"Precision: {report['weighted avg']['precision']:.2f}")
    print(f"Recall: {report['weighted avg']['recall']:.2f}")
    print(f"F1-Score: {report['weighted avg']['f1-score']:.2f}")
    print("Class Distribution:", class_dist)

def main():
    # Setup paths
    os.makedirs("model", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    print("[1/5] Building dataset...")
    X, y, scaler = build_dataset()
    if X is None:
        print("Failed to build dataset")
        return
    
    class_dist = pd.Series(y).value_counts().to_dict()
    print(f"[2/5] Data ready: {X.shape[0]} samples, {len(class_dist)} classes")
    
    print("[3/5] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)  # 70% train, 30% temp

    # Split temp into valid and test (20% and 10% of original)
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_test, y_test, test_size=1/3, random_state=42, stratify=y_test)
    
    print("[4/5] Handling class imbalance...")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    print("[5/5] Training model...")
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf', 'linear']
    }
    
    model = GridSearchCV(
        SVC(probability=True, class_weight='balanced'),
        param_grid, cv=3, n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(report)
    
    # Save model and log
    joblib.dump(model.best_estimator_, "model/svm_model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")
    
    with open("reports/classification_report.txt", 'w') as f:
        f.write(report)
    
    save_training_log(model, X_test, y_test, X_train, y_train, class_dist)
    print("\nTraining complete!")

if __name__ == "__main__":
    main()