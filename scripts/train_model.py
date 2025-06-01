import os
import joblib
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from extract_features import extract_features

def build_dataset():
    """Membangun dataset dari folder gambar dengan pembagian train/valid/test"""
    # Path ke folder dataset
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../dataset')
    train_dir = os.path.join(base_dir, 'train')
    valid_dir = os.path.join(base_dir, 'valid')
    test_dir = os.path.join(base_dir, 'test')
    
    # Fungsi untuk memuat data dari satu folder
    def load_from_folder(folder_path):
        csv_path = os.path.join(folder_path, '_annotations.csv')
        if not os.path.exists(csv_path):
            return [], []
            
        df = pd.read_csv(csv_path)
        features = []
        labels = []
        
        for _, row in df.iterrows():
            img_path = os.path.join(folder_path, row['filename'])
            if not os.path.exists(img_path):
                continue
                
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            feat = extract_features(img)
            if feat is not None:
                features.append(feat)
                labels.append(row['class'])
        
        return features, labels
    
    # Gabungkan semua data untuk training
    X_train, y_train = load_from_folder(train_dir)
    X_valid, y_valid = load_from_folder(valid_dir)
    X_test, y_test = load_from_folder(test_dir)
    
    # Gabungkan train dan valid untuk training model
    X_train.extend(X_valid)
    y_train.extend(y_valid)
    
    if not X_train:
        return None, None, None, None, None
        
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test) if X_test else None
    
    return X_train, np.array(y_train), X_test, np.array(y_test) if y_test else None, scaler

def main():
    # Path configuration
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, '../model')
    report_dir = os.path.join(current_dir, '../reports')
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    
    print("[1/5] Membangun dataset...")
    X_train, y_train, X_test, y_test, scaler = build_dataset()
    if X_train is None:
        print("Gagal membangun dataset")
        return
    
    print(f"[2/5] Data siap: {len(X_train)} sampel training")
    print(f"Distribusi kelas: {pd.Series(y_train).value_counts().to_dict()}")

    print("[3/5] Menangani class imbalance dengan SMOTE...")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    print("[4/5] Training model SVM...")
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

    # Evaluasi jika ada test set
    if X_test is not None:
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        print("\nClassification Report:")
        print(report)
        
        with open(os.path.join(report_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)
    else:
        print("\nTidak ada data testing untuk evaluasi")

    # Simpan model
    joblib.dump(model.best_estimator_, os.path.join(model_dir, 'svm_model.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    
    print(f"\nModel disimpan di: {model_dir}")

if __name__ == "__main__":
    main()