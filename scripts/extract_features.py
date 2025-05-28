import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler

def extract_shape_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return [0]*4

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    circularity = 4*np.pi*area/(perimeter**2) if perimeter > 0 else 0
    hull = cv2.convexHull(cnt)
    solidity = area/cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0

    return [area, perimeter, circularity, solidity]

def extract_glcm_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    distances = [1, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    properties = ['contrast', 'homogeneity', 'energy', 'correlation']

    features = []
    for d in distances:
        for angle in angles:
            glcm = graycomatrix(gray, distances=[d], angles=[angle], 
                              levels=256, symmetric=True, normed=True)
            for prop in properties:
                features.append(graycoprops(glcm, prop)[0, 0])
    return features

def extract_color_features(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    features = []
    for i in range(3):
        features.append(np.mean(hsv[:,:,i]))
        features.append(np.std(hsv[:,:,i]))
    return features

def extract_features_from_bbox(image_path, bbox):
    img = cv2.imread(image_path)
    if img is None:
        return None

    x_min, y_min, x_max, y_max = bbox
    crop = img[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return None

    crop = cv2.resize(crop, (128, 128))
    glcm_features = extract_glcm_features(crop)
    color_features = extract_color_features(crop)
    shape_features = extract_shape_features(crop)
    return glcm_features + color_features + shape_features

def build_dataset(image_folder):
    csv_path = os.path.join(image_folder, '_annotations.csv')
    df = pd.read_csv(csv_path)
    features_list = []
    labels_list = []

    for _, row in df.iterrows():
        img_path = os.path.join(image_folder, row['filename'])
        bbox = (int(row['xmin']), int(row['ymin']), 
                int(row['xmax']), int(row['ymax']))
        features = extract_features_from_bbox(img_path, bbox)
        if features is not None:
            features_list.append(features)
            labels_list.append(row['class'])

    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features_list)
    return features_normalized, np.array(labels_list), scaler