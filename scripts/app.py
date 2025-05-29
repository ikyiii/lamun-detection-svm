import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
import os
import pandas as pd
import uuid
from train_model_autoretrain import retrain_model
from extract_features import extract_features

# Konfigurasi
CONFIG = {
    'min_samples': 5,          # Minimal sampel untuk training
    'min_classes': 2,          # Minimal kelas berbeda
    'retrain_threshold': 3     # Minimal sampel baru untuk retrain
}

# Session state untuk tracking
if 'new_samples_added' not in st.session_state:
    st.session_state.new_samples_added = 0
if 'last_trained' not in st.session_state:
    st.session_state.last_trained = 0

@st.cache_resource
def load_model():
    """Load model dengan error handling"""
    try:
        model = joblib.load("model/svm_model.pkl")
        scaler = joblib.load("model/scaler.pkl")
        return model, scaler, None
    except Exception as e:
        return None, None, str(e)

def save_to_dataset(image, label):
    """Simpan data baru ke dataset"""
    try:
        # 1. Simpan gambar
        filename = f"{uuid.uuid4().hex}.jpg"
        cv2.imwrite(f"dataset/images/{filename}", image)
        
        # 2. Update CSV
        new_row = pd.DataFrame([[
            filename, 640, 480, label, 0, 0, 128, 128
        ]], columns=["filename","width","height","class","xmin","ymin","xmax","ymax"])
        
        if os.path.exists("dataset/images/_annotations.csv"):
            df = pd.read_csv("dataset/images/_annotations.csv")
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            df = new_row
        df.to_csv("dataset/images/_annotations.csv", index=False)
        
        # 3. Update counter
        st.session_state.new_samples_added += 1
        return True, "Data saved successfully"
    except Exception as e:
        return False, str(e)

def check_retrain_conditions():
    """Cek apakah perlu dilakukan retraining"""
    # 1. Cukup sampel baru?
    if st.session_state.new_samples_added < CONFIG['retrain_threshold']:
        return False, f"Need {CONFIG['retrain_threshold']} new samples (current: {st.session_state.new_samples_added})"
    
    # 2. Cek dataset lengkap
    if not os.path.exists("dataset/images/_annotations.csv"):
        return False, "Dataset not found"
    
    df = pd.read_csv("dataset/images/_annotations.csv")
    if len(df['class'].unique()) < CONFIG['min_classes']:
        return False, f"Need {CONFIG['min_classes']} different classes"
    
    return True, "Conditions met"

def main():
    st.set_page_config(
        page_title="Lamun Classifier - Klasifikasi Jenis Lamun dengan SVM",
        page_icon="🌿",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    st.title("🔍🌿 Lamun Classifier - Klasifikasi Jenis Lamun dengan SVM")
    st.write("Upload gambar lamun untuk diklasifikasi")
    st.write("🌿Jenis Lamun yang dapat dikenali : ")
    st.write("•- Thalassia Hemprichii | Cymodocea Rotundata | Enhalus Acoraides -•")
    
    # Load model awal
    model, scaler, model_error = load_model()
    if model_error:
        st.error(f"Error loading model: {model_error}")
    
    # Upload gambar
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg","jpeg","png"])
    
    if uploaded_file:
        # Proses gambar
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Konversi ke BGR
        if img_array.ndim == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        else:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Tampilkan preview
        st.image(cv2.resize(img_array, (128,128)), caption="Gambar yang diupload")
        
        # Lakukan prediksi jika model ada
        if model and scaler:
            try:
                features = extract_features(img_array)
                features = scaler.transform([features])
                prob = model.predict_proba(features)[0]
                pred_class = model.classes_[np.argmax(prob)]
                
                st.success(f"Hasil Prediksi: {pred_class}")
                st.bar_chart({k:v for k,v in zip(model.classes_, prob)})
                
                # Validasi user
                st.subheader("Validasi Hasil")
                true_label = st.selectbox("Label sebenarnya:", model.classes_)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Prediksi Benar"):
                        save_to_dataset(img_array, pred_class)
                        st.success("Data disimpan dengan label prediksi")
                with col2:
                    if st.button("❌ Prediksi Salah"):
                        save_to_dataset(img_array, true_label)
                        st.success("Data disimpan dengan label koreksi")
                
                # Cek retraining
                ready, msg = check_retrain_conditions()
                if ready:
                    if st.button("🔄 Latih Ulang Model"):
                        with st.spinner("Melatih model..."):
                            success, message = retrain_model()
                            if success:
                                st.success("Model berhasil diperbarui!")
                                st.session_state.new_samples_added = 0
                                st.cache_resource.clear()
                            else:
                                st.error(f"Gagal: {message}")
                else:
                    st.info(f"Info: {msg}")
                        
            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #7f8c8d;'>"
        "© 2025 Lamun Classifier | Senggarang Selatan"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()