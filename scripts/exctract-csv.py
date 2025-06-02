import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

# Path konfigurasi
image_dir = "dataset/images"
csv_path = "dataset/images/_annotations.csv"
output_dir = "summary"
os.makedirs(output_dir, exist_ok=True)
output_csv = os.path.join(output_dir, "extract.csv")

# Fitur yang dihitung
distances = [1, 3]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
properties = ["contrast", "homogeneity", "energy", "correlation"]
glcm_features = [f"glcm_{p}_d{d}_a{int(np.degrees(a))}" for d in distances for a in angles for p in properties]
color_features = ["hue_mean", "hue_std", "saturation_mean", "saturation_std", "value_mean", "value_std"]
shape_features = ["area", "perimeter", "circularity", "solidity"]
all_columns = ["filename"] + glcm_features + color_features + shape_features + ["class"]

# Fungsi ekstraksi fitur
def extract_all_features(image):
    image = cv2.resize(image, (128, 128))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # GLCM
    glcm = []
    for d in distances:
        for angle in angles:
            mat = graycomatrix(gray, [d], [angle], levels=256, symmetric=True, normed=True)
            for prop in properties:
                glcm.append(graycoprops(mat, prop)[0, 0])

    # Warna
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color = [np.mean(hsv[:,:,i]) for i in range(3)] + [np.std(hsv[:,:,i]) for i in range(3)]

    # Bentuk
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        shape = [0, 0, 0, 0]
    else:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        hull = cv2.convexHull(cnt)
        solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
        shape = [area, perimeter, circularity, solidity]

    return glcm + color + shape

# Ekstraksi semua data
if not os.path.exists(csv_path):
    print(f"[ERROR] File anotasi tidak ditemukan: {csv_path}")
else:
    df = pd.read_csv(csv_path)
    rows = []
    for _, row in df.iterrows():
        img_path = os.path.join(image_dir, row["filename"])
        if not os.path.exists(img_path):
            print(f"[WARNING] Gambar tidak ditemukan: {img_path}")
            continue
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARNING] Gagal membaca gambar: {img_path}")
            continue
        features = extract_all_features(img)
        rows.append([row["filename"]] + features + [row["class"]])

    df_out = pd.DataFrame(rows, columns=all_columns)
    df_out.to_csv(output_csv, index=False)
    print(f"[INFO] Fitur berhasil diekstrak ke: {output_csv}")
