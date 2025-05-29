import os
import cv2
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from extract_features import extract_features

def build_dataset():
    """Membangun dataset dari file CSV"""
    df = pd.read_csv("dataset/images/_annotations.csv")
    
    features = []
    labels = []
    
    for _, row in df.iterrows():
        img_path = f"dataset/images/{row['filename']}"
        if not os.path.exists(img_path):
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

def retrain_model():
    """Fungsi utama untuk retraining"""
    print("\nMemulai proses retraining...")
    
    # 1. Siapkan dataset
    X, y, scaler = build_dataset()
    if X is None:
        return False, "Tidak ada data valid untuk training"
    
    # 2. Handle class imbalance
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    # 3. Training dengan parameter tetap (lebih cepat)
    model = SVC(
        C=1.0, 
        kernel='rbf', 
        gamma='scale',
        probability=True,
        class_weight='balanced'
    )
    model.fit(X_res, y_res)
    
    # 4. Simpan model
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/svm_model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")
    
    return True, "Model berhasil diperbarui"

if __name__ == "__main__":
    retrain_model()