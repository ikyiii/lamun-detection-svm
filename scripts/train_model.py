import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

from extract_features import build_dataset

current_dir = os.path.dirname(os.path.abspath(__file__))
image_folder = os.path.join(os.path.dirname(current_dir), 'dataset/images')
model_output_path = os.path.join(os.path.dirname(current_dir), 'model/svm_model.csv')
scaler_output_path = os.path.join(os.path.dirname(current_dir), 'model/scaler.csv')
report_output_path = os.path.join(os.path.dirname(current_dir), 'reports/classification_report.txt')

def main():
    print("[INFO] Ekstraksi fitur...")
    X, y, scaler = build_dataset(image_folder)
    print(f"[INFO] Jumlah data: {X.shape[0]}, Jumlah fitur: {X.shape[1]}")
    print(f"[INFO] Distribusi kelas:\n{pd.Series(y).value_counts()}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("[INFO] Menangani ketidakseimbangan kelas dengan SMOTE...")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    print("[INFO] Training SVM dengan Grid Search dan class_weight...")
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1],
        'kernel': ['rbf', 'linear', 'poly'],
        'degree': [2, 3, 4] 
    }

    model = GridSearchCV(SVC(probability=True, class_weight='balanced'), param_grid, cv=5, n_jobs=-1)
    model.fit(X_train, y_train)
    print(f"[INFO] Parameter terbaik: {model.best_params_}")

    print("[INFO] Evaluasi model:")
    y_pred = model.predict(X_test)

    os.makedirs(os.path.dirname(report_output_path), exist_ok=True)
    with open(report_output_path, 'w') as f:
        f.write(classification_report(y_test, y_pred))
        f.write("\n\nConfusion Matrix:\n")
        f.write(np.array2string(confusion_matrix(y_test, y_pred)))

    print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model.best_estimator_, model_output_path)
    joblib.dump(scaler, scaler_output_path)
    print(f"[INFO] Model disimpan di: {model_output_path}")
    print(f"[INFO] Scaler disimpan di: {scaler_output_path}")

if __name__ == "__main__":
    main()