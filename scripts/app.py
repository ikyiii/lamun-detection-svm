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
    'min_samples': 10,          # Minimal sampel untuk training
    'min_classes': 3,          # Minimal kelas berbeda
    'retrain_threshold': 10,    # Minimal sampel baru untuk retrain
    'dataset_path': "dataset/train"  # Path untuk menyimpan data baru
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
    """Simpan data baru ke folder train"""
    try:
        # 1. Pastikan folder dataset ada
        os.makedirs(CONFIG['dataset_path'], exist_ok=True)
        
        # 2. Simpan gambar
        filename = f"{uuid.uuid4().hex}.jpg"
        img_path = os.path.join(CONFIG['dataset_path'], filename)
        cv2.imwrite(img_path, image)
        
        # 3. Update CSV annotations
        new_row = pd.DataFrame([[
            filename, 640, 480, label, 0, 0, 640, 480
        ]], columns=["filename","width","height","class","xmin","ymin","xmax","ymax"])
        
        csv_path = os.path.join(CONFIG['dataset_path'], "_annotations.csv")
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            df = new_row
            
        df.to_csv(csv_path, index=False)
        
        # 4. Update counter
        st.session_state.new_samples_added += 1
        return True, "Data berhasil disimpan ke dataset training"
    except Exception as e:
        return False, f"Gagal menyimpan data: {str(e)}"

def check_retrain_conditions():
    """Cek apakah perlu dilakukan retraining"""
    # 1. Cukup sampel baru?
    if st.session_state.new_samples_added < CONFIG['retrain_threshold']:
        return False, f"Dibutuhkan {CONFIG['retrain_threshold']} sampel baru (saat ini: {st.session_state.new_samples_added})"
    
    # 2. Cek dataset lengkap
    csv_path = os.path.join(CONFIG['dataset_path'], "_annotations.csv")
    if not os.path.exists(csv_path):
        return False, "File annotations tidak ditemukan"
    
    try:
        df = pd.read_csv(csv_path)
        unique_classes = df['class'].unique()
        
        if len(unique_classes) < CONFIG['min_classes']:
            return False, f"Dibutuhkan minimal {CONFIG['min_classes']} kelas berbeda"
            
        if len(df) < CONFIG['min_samples']:
            return False, f"Dibutuhkan minimal {CONFIG['min_samples']} sampel"
            
        return True, "Kondisi retraining terpenuhi"
    except Exception as e:
        return False, f"Error membaca dataset: {str(e)}"

def preprocess_image(image):
    """Preprocessing gambar untuk konsistensi"""
    # Konversi ke numpy array jika belum
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Konversi ke BGR (OpenCV format)
    if image.ndim == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:  # RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image

def display_prediction_results(model, scaler, image):
    """Tampilkan hasil prediksi dan opsi feedback"""
    try:
        # Ekstrak fitur dan prediksi
        features = extract_features(image)
        if features is None:
            raise ValueError("Gagal mengekstrak fitur dari gambar")
            
        features_scaled = scaler.transform([features])
        prob = model.predict_proba(features_scaled)[0]
        
        # Pastikan classes adalah string
        classes = [str(cls) for cls in model.classes_]
        pred_class = str(model.classes_[np.argmax(prob)])
        
        # Tampilkan hasil
        st.success(f"Hasil Prediksi: {pred_class}")
        
        # Visualisasi probabilitas
        prob_data = {str(cls): prob[i] for i, cls in enumerate(model.classes_)}
        st.bar_chart(prob_data)
        
        # Bagian feedback user
        st.subheader("Validasi Hasil")
        
        # Pastikan pred_class ada dalam daftar classes
        if pred_class not in classes:
            classes.insert(0, pred_class)
            
        true_label = st.selectbox(
            "Label sebenarnya:", 
            classes,
            index=classes.index(pred_class)
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Prediksi Benar"):
                success, message = save_to_dataset(image, pred_class)
                if success:
                    st.success(message)
                else:
                    st.error(message)
        with col2:
            if st.button("❌ Prediksi Salah"):
                success, message = save_to_dataset(image, true_label)
                if success:
                    st.success(message)
                else:
                    st.error(message)
        
        # Cek kondisi retraining
        ready, msg = check_retrain_conditions()
        if ready:
            if st.button("🔄 Latih Ulang Model dengan Data Baru"):
                with st.spinner("Melatih model dengan data terbaru..."):
                    success, message = retrain_model()
                    if success:
                        st.success("Model berhasil diperbarui!")
                        st.session_state.new_samples_added = 0
                        st.cache_resource.clear()  # Clear cache untuk load model baru
                    else:
                        st.error(f"Gagal: {message}")
        else:
            st.info(f"Info: {msg}")
            
    except Exception as e:
        st.error(f"Error dalam prediksi: {str(e)}")


def main():
    # Konfigurasi halaman
    st.set_page_config(
        page_title="Lamun Classifier - Klasifikasi Jenis Lamun dengan SVM",
        page_icon="🌿",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    # Header aplikasi
    st.title("🔍🌿 Lamun Classifier")
    st.markdown("""
    <style>
    .title {
        color: #2e86ab;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.write("Upload gambar lamun untuk diklasifikasi menggunakan model SVM")
    
    # Informasi kelas yang dikenali
    with st.expander("🌿 Jenis Lamun yang Dikenali"):
        st.write("""
        - Thalassia Hemprichii
        - Cymodocea Rotundata  
        - Enhalus Acoraides
        """)
    
    # Load model
    model, scaler, model_error = load_model()
    if model_error:
        st.error(f"Error memuat model: {model_error}")
        st.warning("Aplikasi tetap berjalan tetapi fitur prediksi tidak tersedia")
    
    # Upload gambar
    uploaded_file = st.file_uploader(
        "Pilih gambar lamun...", 
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False
    )
    
    if uploaded_file:
        try:
            # Baca dan preprocess gambar
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar yang diupload", width=300)
            
            img_array = preprocess_image(image)
            
            # Lakukan prediksi jika model tersedia
            if model and scaler:
                display_prediction_results(model, scaler, img_array)
            else:
                st.warning("Model tidak tersedia, tidak dapat melakukan prediksi")
                
        except Exception as e:
            st.error(f"Error memproses gambar: {str(e)}")
    
    # Sidebar informasi
    st.sidebar.title("Informasi Aplikasi")
    st.sidebar.markdown("""
    Aplikasi ini menggunakan model SVM untuk mengklasifikasikan jenis lamun:
    - **Thalassia Hemprichii**
    - **Cymodocea Rotundata**
    - **Enhalus Acoraides**
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Statistik Data")
    if model:
        st.sidebar.write(f"Jumlah kelas: {len(model.classes_)}")
    
    if os.path.exists(os.path.join(CONFIG['dataset_path'], "_annotations.csv")):
        try:
            df = pd.read_csv(os.path.join(CONFIG['dataset_path'], "_annotations.csv"))
            st.sidebar.write(f"Total sampel training: {len(df)}")
            st.sidebar.write(f"Sampel baru ditambahkan: {st.session_state.new_samples_added}")
        except:
            pass
    
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