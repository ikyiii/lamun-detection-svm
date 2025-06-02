import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def extract_features(image):
    """Ekstraksi fitur GLCM, warna, dan bentuk"""
    try:
        # 1. Konversi ke grayscale
        image = cv2.resize(image, (640,480))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 2. GLCM Features
        distances = [1, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        properties = ['contrast', 'homogeneity', 'energy', 'correlation']
        
        glcm_features = []
        for d in distances:
            for angle in angles:
                glcm = graycomatrix(gray, [d], [angle], levels=256, symmetric=True, normed=True)
                for prop in properties:
                    glcm_features.append(graycoprops(glcm, prop)[0, 0])
        
        # 3. Color Features (HSV)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        color_features = [
            *np.mean(hsv, axis=(0,1)),  # mean H, S, V
            *np.std(hsv, axis=(0,1))    # std H, S, V
        ]
        
        # 4. Shape Features
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            shape_features = [0, 0, 0, 0]
        else:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            circularity = 0 if perimeter == 0 else (4 * np.pi * area) / (perimeter ** 2)
            hull = cv2.convexHull(cnt)
            solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
            shape_features = [area, perimeter, circularity, solidity]
        
        return np.concatenate([glcm_features, color_features, shape_features])
    
    except Exception as e:
        print(f"Error ekstraksi fitur: {str(e)}")
        return None