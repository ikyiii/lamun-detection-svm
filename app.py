import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
import os
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops

# Konfigurasi
model_path = "model/svm_model.pkl"
scaler_path = "model/scaler.pkl"

# Load model dan scaler
@st.cache_resource
def load_model():
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_model()

# Fungsi ekstraksi fitur (GLCM + Warna + Bentuk)
def extract_features(image):
    image = cv2.resize(image, (128, 128))

    # Fitur GLCM
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    distances = [1, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    properties = ['contrast', 'homogeneity', 'energy', 'correlation']

    glcm_features = []
    for d in distances:
        for angle in angles:
            glcm = graycomatrix(gray, distances=[d], angles=[angle],
                                levels=256, symmetric=True, normed=True)
            for prop in properties:
                glcm_features.append(graycoprops(glcm, prop)[0, 0])

    # Fitur warna
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_features = []
    for i in range(3):
        color_features.append(np.mean(hsv[:, :, i]))
        color_features.append(np.std(hsv[:, :, i]))

    # Fitur bentuk
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        shape_features = [0, 0, 0, 0]
    else:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        hull = cv2.convexHull(cnt)
        solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
        shape_features = [area, perimeter, circularity, solidity]

    return glcm_features + color_features + shape_features

# Antarmuka Streamlit
st.title("🔍🌿 Lamun Classifier - Klasifikasi Jenis Lamun dengan SVM")
st.write("Upload gambar daun lamun yang ingin dikenali.")
st.write("Jenis Lamun yang dapat dikenali : ")
st.write(" • Thalassia Hemprichii ")
st.write(" • Cymodocea Rotundata ")
st.write(" • Enhalus Acoraides.")

uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    if img_array.ndim == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    elif img_array.shape[2] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    st.image(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB), 
         caption='Gambar yang Diupload', 
         use_container_width=True)

    features = extract_features(img_array)
    features_scaled = scaler.transform([features])

    prob = model.predict_proba(features_scaled)[0]
    class_index = np.argmax(prob)
    pred = model.classes_[class_index]

    st.success(f"✅ Jenis lamun: **{pred}**")

    st.write("🔢 Probabilitas Klasifikasi:")
    fig, ax = plt.subplots()
    bars = ax.bar(model.classes_, prob)
    ax.set_xlabel('Kelas')
    ax.set_ylabel('Probabilitas')
    ax.set_ylim([0, 1])

    # Tambahkan angka di atas bar
    for bar, p in zip(bars, prob):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f"{p:.2f}",
                ha='center', va='bottom', fontsize=10)

    st.pyplot(fig)

    if hasattr(model, 'coef_'):
        st.write("📊 Fitur Penting untuk Klasifikasi:")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.bar(range(len(model.coef_[0])), model.coef_[0])
        ax2.set_xlabel('Indeks Fitur')
        ax2.set_ylabel('Koefisien SVM')
        st.pyplot(fig2)
