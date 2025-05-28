import joblib
import pandas as pd
import os

# Path file model dan scaler
model_path = os.path.join("model", "svm_model.pkl")
scaler_path = os.path.join("model", "scaler.pkl")

# Cek apakah file ada
if not os.path.exists(model_path):
    print(f"❌ File tidak ditemukan: {model_path}")
    exit()
if not os.path.exists(scaler_path):
    print(f"❌ File tidak ditemukan: {scaler_path}")
    exit()

# Load model dan scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# === Buat folder summary jika belum ada
summary_dir = "summary"
os.makedirs(summary_dir, exist_ok=True)

# === Simpan parameter model
params = model.get_params()
params['classes'] = ', '.join(model.classes_)

df_model = pd.DataFrame(list(params.items()), columns=["Parameter", "Value"])
df_model.to_csv(os.path.join(summary_dir, "model_summary.csv"), index=False)

# === Simpan ringkasan scaler
df_scaler = pd.DataFrame({
    "Feature": [f"F{i+1}" for i in range(len(scaler.mean_))],
    "Mean": scaler.mean_,
    "Std Dev": scaler.scale_
})
df_scaler.to_csv(os.path.join(summary_dir, "scaler_summary.csv"), index=False)

print("✅ CSV berhasil disimpan ke folder 'summary'")
