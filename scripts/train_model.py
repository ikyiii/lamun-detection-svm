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

def build_dataset(image_folder):
    """Membangun dataset dari folder gambar"""
    csv_path = os.path.join(image_folder, "_annotations.csv")
    
    if not os.path.exists(csv_path):
        print(f"File CSV tidak ditemukan di {csv_path}")
        return None, None, None
    
    df = pd.read_csv(csv_path)
    features = []
    labels = []
    
    for _, row in df.iterrows():
        img_path = os.path.join(image_folder, row['filename'])
        if not os.path.exists(img_path):
            print(f"Gambar tidak ditemukan: {img_path}")
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            print(f"Gagal membaca gambar: {img_path}")
            continue
            
        feat = extract_features(img)
        if feat is not None:
            features.append(feat)
            labels.append(row['class'])
    
    if not features:
        print("Tidak ada fitur yang berhasil diekstraksi")
        return None, None, None
        
    scaler = StandardScaler()
    return scaler.fit_transform(features), np.array(labels), scaler

def main():
    # Path configuration
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_folder = os.path.join(current_dir, '../dataset/images')
    model_dir = os.path.join(current_dir, '../model')
    report_dir = os.path.join(current_dir, '../reports')
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    
    print("[1/5] Membangun dataset...")
    X, y, scaler = build_dataset(image_folder)
    if X is None:
        print("Gagal membangun dataset")
        return
    
    print(f"[2/5] Data siap: {X.shape[0]} sampel, {X.shape[1]} fitur")
    print(f"Distribusi kelas: {pd.Series(y).value_counts().to_dict()}")

    print("[3/5] Membagi data training-testing...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print("[4/5] Menangani class imbalance dengan SMOTE...")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    print("[5/5] Training model SVM...")
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

    # Evaluasi
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(report)

    # Simpan model dan laporan
    joblib.dump(model.best_estimator_, os.path.join(model_dir, 'svm_model.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    
    with open(os.path.join(report_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    print(f"\nModel disimpan di: {model_dir}")

if __name__ == "__main__":
    main()