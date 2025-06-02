import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
import os
import pandas as pd
import uuid
import base64
from io import BytesIO
from train_model_autoretrain import retrain_model
from extract_features import extract_features

# Set page configuration
st.set_page_config(
    page_title="Lamun Classifier", 
    page_icon="🌿",
    layout="centered")

# Configuration
CONFIG = {
    'min_samples': 8,
    'min_classes': 3,
    'retrain_threshold': 10,
    'compression_quality': 90,
    'image_size': (640, 480),        # Original size for dataset storage
    'display_size': (150, 150),      # Small size for preview
    'split_ratios': {'train': 0.7, 'valid': 0.2, 'test': 0.1}
}

# Initialize session state  
if 'new_samples_added' not in st.session_state:
    st.session_state.new_samples_added = 0

# Helper function to convert image to base64
def image_to_base64(img_array):
    _, buffer = cv2.imencode('.jpg', img_array)
    return base64.b64encode(buffer).decode()

def get_split_folder():
    rand = np.random.random()
    if rand < CONFIG['split_ratios']['train']:
        return 'train'
    elif rand < CONFIG['split_ratios']['train'] + CONFIG['split_ratios']['valid']:
        return 'valid'
    else:
        return 'test'

@st.cache_resource
def load_model():
    try:
        model = joblib.load("model/svm_model.pkl")
        scaler = joblib.load("model/scaler.pkl")
        return model, scaler, None
    except Exception as e:
        return None, None, str(e)

def compress_image(image):
    try:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), CONFIG['compression_quality']]
        _, encoded_img = cv2.imencode('.jpg', image, encode_param)
        return cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    except Exception as e:
        st.error(f"Compression error: {str(e)}")
        return image

def save_to_dataset(image, label):
    try:
        # Use original size for dataset storage
        compressed_img = compress_image(image)
        resized_img = cv2.resize(compressed_img, CONFIG['image_size'])
        
        split = get_split_folder()
        filename = f"{uuid.uuid4().hex}.jpg"
        img_path = f"dataset/{split}/images/{filename}"
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        cv2.imwrite(img_path, resized_img)
        
        new_row = pd.DataFrame([[
            filename, *CONFIG['image_size'], label, 0, 0, *CONFIG['image_size']
        ]], columns=["filename","width","height","class","xmin","ymin","xmax","ymax"])
        
        csv_path = f"dataset/{split}/_annotations.csv"
        df = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(csv_path, index=False)
        
        st.session_state.new_samples_added += 1
        return True, f"Data saved to {split} set"
    except Exception as e:
        return False, str(e)

def check_retrain_conditions():
    if st.session_state.new_samples_added < CONFIG['retrain_threshold']:
        return False, f"Need {CONFIG['retrain_threshold']} new samples (current: {st.session_state.new_samples_added})"
    
    required_files = [
        "dataset/train/_annotations.csv",
        "dataset/valid/_annotations.csv",
        "dataset/test/_annotations.csv"
    ]
    
    if not all(os.path.exists(f) for f in required_files):
        return False, "Dataset files missing"
    
    dfs = []
    for split in ['train', 'valid', 'test']:
        dfs.append(pd.read_csv(f"dataset/{split}/_annotations.csv"))
    df_combined = pd.concat(dfs)
    
    if len(df_combined['class'].unique()) < CONFIG['min_classes']:
        return False, f"Need {CONFIG['min_classes']} different classes"
    
    return True, "Retraining conditions met"

# UI Styling
st.markdown("""
    <style>
    .main {
        background-color: #f5fff5;
        font-family: 'Segoe UI', sans-serif;
    }
    h1 {
        color: #2e7d32;
        text-align: center;
        font-weight: bold;
    }
    h2, h3 {
        color: #1b5e20;
    }
    .stButton button {
        background-color: #4caf50;
        color: white;
        border-radius: 8px;
        width: 100%;
        padding: 10px;
        font-size: 16px;
    }
    .stButton button:hover {
        background-color: #388e3c;
    }
    .stTextInput input {
        border: 2px solid #4caf50;
        border-radius: 5px;
    }
    .stSelectbox select {
        border: 2px solid #4caf50;
        border-radius: 5px;
    }
    .footer {
        margin-top: 30px;
        text-align: center;
        color: #aaa;
        font-size: 14px;
    }
    .preview-container {
        display: flex;
        justify-content: center;
        margin: 15px 0;
    }
    .preview-image {
        width: 150px;
        height: auto;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

def main():
    st.title("🔍🌿 Lamun Classifier - Klasifikasi Jenis Lamun dengan SVM")
    st.markdown("<p style='text-align:center; font-size:18px;'>Upload gambar untuk klasifikasi jenis lamun</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'><strong>Thalassia Hemprichii | Cymodocea Rotundata | Enhalus Acoraides</strong></p>", unsafe_allow_html=True)

    model, scaler, model_error = load_model()
    if model_error:
        st.warning(f"⚠️ Model tidak ditemukan: {model_error}")

    uploaded_file = st.file_uploader("📷 Pilih gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Process uploaded image
        image = Image.open(uploaded_file)
        img_array = np.array(image)

        # Convert image format if needed
        if img_array.ndim == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        else:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Create small preview (150x150)
        display_img = cv2.resize(img_array, CONFIG['display_size'])
        
        # Display small preview with custom styling
        st.markdown(
            f"""
            <div class="preview-container">
                <img src="data:image/jpeg;base64,{image_to_base64(display_img)}" 
                     class="preview-image"
                     alt="Preview Gambar">
            </div>
            <p style='text-align:center; font-size:14px; color:#666;'>
                Preview Gambar ({CONFIG['display_size'][0]}x{CONFIG['display_size'][1]})
            </p>
            """,
            unsafe_allow_html=True
        )

        if model and scaler:
            try:
                # Feature extraction uses original image (not the small preview)
                features = extract_features(img_array)
                features = scaler.transform([features])
                prob = model.predict_proba(features)[0]
                pred_class = model.classes_[np.argmax(prob)]

                st.success(f"🎯 Prediksi: **{pred_class}** ({max(prob):.2%})")
                st.bar_chart({k:v for k,v in zip(model.classes_, prob)})

                st.subheader("✅ Validasi Hasil")
                true_label = st.selectbox("Pilih label yang benar:", model.classes_)

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Simpan sebagai Benar"):
                        save_to_dataset(img_array, pred_class)
                        st.success("Gambar disimpan sesuai prediksi")
                with col2:
                    if st.button("❌ Simpan sebagai Salah"):
                        save_to_dataset(img_array, true_label)
                        st.success("Gambar disimpan dengan label baru")

                ready, msg = check_retrain_conditions()
                if ready:
                    if st.button("🔄 Retrain Model Baru"):
                        with st.spinner("Sedang melatih ulang model..."):
                            success, message = retrain_model()
                            if success:
                                st.success("✅ Model berhasil diperbarui!")
                                st.session_state.new_samples_added = 0
                                st.cache_resource.clear()
                            else:
                                st.error(f"❌ Gagal memperbarui model: {message}")
                else:
                    st.info(f"ℹ️ Persyaratan belum tercapai: {msg}")

            except Exception as e:
                st.error(f"🚨 Kesalahan: {str(e)}")

    st.markdown("---")
    st.markdown("<div class='footer'>© 2025 Lamun Classifier | Senggarang Selatan</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()