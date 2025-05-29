import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def extract_features(image):
    """Ekstraksi fitur dari gambar dengan error handling"""
    try:
        # Pastikan image adalah numpy array dan memiliki 3 channel
        if not isinstance(image, np.ndarray):
            raise ValueError("Input bukan numpy array")
            
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Format gambar harus 3 channel BGR")
        
        # Resize dan konversi ke grayscale
        image = cv2.resize(image, (128, 128))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. GLCM Features
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

        # 2. Color Features (HSV)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        color_features = [
            *np.mean(hsv, axis=(0,1)), 
            *np.std(hsv, axis=(0,1))
        ]

        # 3. Shape Features
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            shape_features = [0, 0, 0, 0]
        else:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            circularity = 0 if perimeter == 0 else (4*np.pi*area)/(perimeter**2)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = 0 if hull_area == 0 else area/hull_area
            shape_features = [area, perimeter, circularity, solidity]

        return np.concatenate([glcm_features, color_features, shape_features])
        
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        return None