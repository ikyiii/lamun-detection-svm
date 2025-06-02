import os
import joblib
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from extract_features import extract_features
import streamlit as st

def build_dataset():
    """Build dataset for retraining from all splits"""
    splits = ['train', 'valid', 'test']
    dfs = []
    
    for split in splits:
        csv_path = f"dataset/{split}/_annotations.csv"
        if not os.path.exists(csv_path):
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
    return scaler.fit_transform(features), labels, scaler

def save_retraining_log(model, X_test, y_test, X_train, y_train, class_dist, samples_added):
    """Save retraining log"""
    import datetime
    import pandas as pd
    from sklearn.metrics import classification_report
    
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    log_data = {
        "timestamp": timestamp,
        "model_type": "SVM (Retrain)",
        "parameters": "C=1.0, kernel=rbf, gamma=scale",
        "accuracy": report['accuracy'],
        "weighted_precision": report['weighted avg']['precision'],
        "weighted_recall": report['weighted avg']['recall'],
        "weighted_f1": report['weighted avg']['f1-score'],
        "samples_added": samples_added,
        "total_samples": len(y_test) + len(y_train),
        **{f"class_{k}": v for k,v in class_dist.items()}
    }
    
    log_df = pd.DataFrame([log_data])
    log_file = "logs/training_logs.csv"
    
    if os.path.exists(log_file):
        existing_logs = pd.read_csv(log_file)
        log_df = pd.concat([existing_logs, log_df], ignore_index=True)
    
    log_df.to_csv(log_file, index=False)
    
    print("\n=== Retraining Log ===")
    print(f"Timestamp: {timestamp}")
    print(f"New samples: {samples_added}")
    print(f"Accuracy: {report['accuracy']:.2f}")
    print(f"Precision: {report['weighted avg']['precision']:.2f}")
    print(f"Recall: {report['weighted avg']['recall']:.2f}")
    print(f"F1-Score: {report['weighted avg']['f1-score']:.2f}")
    print("Class Distribution:", class_dist)

def retrain_model():
    """Main retraining function"""
    print("\nStarting retraining...")
    
    # 1. Prepare dataset
    X, y, scaler = build_dataset()
    if X is None:
        return False, "No valid data for training"
    
    # 2. Handle class imbalance
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    # 3. Split data (70% train, 20% valid, 10% test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_res, y_res, test_size=0.3, random_state=42, stratify=y_res)
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=1/3, random_state=42, stratify=y_temp)
    
    # 4. Training
    model = SVC(
        C=1.0, 
        kernel='rbf', 
        gamma='scale',
        probability=True,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    # 5. Save model and log
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/svm_model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")
    
    class_dist = pd.Series(y).value_counts().to_dict()
    samples_added = st.session_state.new_samples_added if 'new_samples_added' in st.session_state else 0
    save_retraining_log(model, X_test, y_test, X_train, y_train, class_dist, samples_added)
    
    return True, "Model updated successfully"

if __name__ == "__main__":
    retrain_model()