import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt
from pathlib import Path

class GLCMExtractor:
    def __init__(self, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
        """
        Initialize GLCM extractor
        
        Args:
            distances: list of pixel distances (default: [1])
            angles: list of angles in radians (default: [0, 45, 90, 135 degrees])
        """
        self.distances = distances
        self.angles = angles
        self.properties = ['contrast', 'homogeneity', 'energy', 'correlation']
    
    def extract_glcm_features(self, image_path):
        """
        Extract GLCM features from an image
        
        Args:
            image_path: path to the image file
            
        Returns:
            dict: dictionary containing GLCM features
        """
        # Load and convert image to grayscale
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate GLCM
        glcm = graycomatrix(
            gray, 
            distances=self.distances, 
            angles=self.angles,
            levels=256,
            symmetric=True,
            normed=True
        )
        
        # Extract features
        results = {}
        
        for prop in self.properties:
            # Calculate feature for each distance and angle combination
            feature_values = graycoprops(glcm, prop)
            
            # Store results for each distance-angle combination
            for i, distance in enumerate(self.distances):
                for j, angle in enumerate(self.angles):
                    angle_deg = int(np.degrees(angle))
                    key = f"{prop}_d{distance}_a{angle_deg}"
                    results[key] = feature_values[i, j]
            
            # Calculate mean across all directions for each distance
            for i, distance in enumerate(self.distances):
                mean_key = f"{prop}_d{distance}_mean"
                results[mean_key] = np.mean(feature_values[i, :])
        
        # Calculate overall mean across all distances and angles
        for prop in self.properties:
            overall_mean = np.mean([results[k] for k in results.keys() if k.startswith(prop) and k.endswith('mean')])
            results[f"{prop}_overall"] = overall_mean
        
        return results
    
    def print_features(self, features, title="GLCM Features"):
        """
        Print GLCM features in a formatted way
        
        Args:
            features: dictionary of GLCM features
            title: title for the output
        """
        print(f"\n{'='*50}")
        print(f"{title:^50}")
        print(f"{'='*50}")
        
        # Print overall features (main results)
        print("\n📊 OVERALL FEATURES:")
        print("-" * 30)
        for prop in self.properties:
            value = features[f"{prop}_overall"]
            print(f"{prop.upper():>15}: {value:.6f}")
        
        # Print detailed results by distance and angle
        print(f"\n📋 DETAILED RESULTS:")
        print("-" * 30)
        
        for distance in self.distances:
            print(f"\nDistance {distance}:")
            
            # Mean values for this distance
            print("  Mean across all angles:")
            for prop in self.properties:
                value = features[f"{prop}_d{distance}_mean"]
                print(f"    {prop:>12}: {value:.6f}")
            
            # Individual angle values
            print("  Individual angles:")
            for angle in self.angles:
                angle_deg = int(np.degrees(angle))
                print(f"    Angle {angle_deg:>3}°:")
                for prop in self.properties:
                    value = features[f"{prop}_d{distance}_a{angle_deg}"]
                    print(f"      {prop:>10}: {value:.6f}")
    
    def save_results(self, features, output_file):
        """
        Save GLCM features to a text file
        
        Args:
            features: dictionary of GLCM features
            output_file: path to output file
        """
        with open(output_file, 'w') as f:
            f.write("GLCM Feature Extraction Results\n")
            f.write("=" * 40 + "\n\n")
            
            # Overall features
            f.write("Overall Features:\n")
            f.write("-" * 20 + "\n")
            for prop in self.properties:
                value = features[f"{prop}_overall"]
                f.write(f"{prop}: {value:.6f}\n")
            
            # Detailed features
            f.write(f"\nDetailed Features:\n")
            f.write("-" * 20 + "\n")
            for key, value in features.items():
                f.write(f"{key}: {value:.6f}\n")
        
        print(f"\n💾 Results saved to: {output_file}")

def main():
    """
    Main function to demonstrate GLCM feature extraction
    """
    print("🔍 GLCM Feature Extractor")
    print("=" * 50)
    
    # Get image path from user
    image_path = input("Enter image path (or press Enter for demo): ").strip()
    
    if not image_path:
        print("📝 Demo mode: Please ensure you have an image file to test")
        print("   Example usage: python glcm_extractor.py")
        return
    
    try:
        # Initialize extractor
        extractor = GLCMExtractor(
            distances=[1, 2],  # You can modify these values
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]
        )
        
        # Extract features
        print(f"\n🔄 Processing image: {image_path}")
        features = extractor.extract_glcm_features(image_path)
        
        # Display results
        extractor.print_features(features, f"GLCM Analysis: {Path(image_path).name}")
        
        # Ask if user wants to save results
        save_option = input("\n💾 Save results to file? (y/n): ").strip().lower()
        if save_option == 'y':
            output_file = Path(image_path).stem + "_glcm_results.txt"
            extractor.save_results(features, output_file)
        
        # Show simple summary
        print(f"\n📈 QUICK SUMMARY:")
        print("-" * 20)
        for prop in extractor.properties:
            value = features[f"{prop}_overall"]
            print(f"{prop.capitalize():>12}: {value:.4f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Please check if the image file exists and is valid.")

def extract_single_image(image_path, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """
    Simple function to extract GLCM features from a single image
    
    Args:
        image_path: path to image
        distances: list of pixel distances
        angles: list of angles in radians
        
    Returns:
        dict: GLCM features
    """
    extractor = GLCMExtractor(distances=distances, angles=angles)
    return extractor.extract_glcm_features(image_path)

# Example usage functions
def example_usage():
    """
    Example of how to use the GLCM extractor programmatically
    """
    # Example 1: Basic usage
    print("Example 1: Basic GLCM extraction")
    try:
        features = extract_single_image("your_image.jpg")
        print("Contrast:", features['contrast_overall'])
        print("Homogeneity:", features['homogeneity_overall'])
        print("Energy:", features['energy_overall'])
        print("Correlation:", features['correlation_overall'])
    except:
        print("Please provide a valid image path")
    
    # Example 2: Custom parameters
    print("\nExample 2: Custom distances and angles")
    try:
        features = extract_single_image(
            "your_image.jpg",
            distances=[1, 3, 5],
            angles=[0, np.pi/2]  # Only 0° and 90°
        )
        # Access specific distance-angle combinations
        print("Contrast at distance 1, angle 0°:", features['contrast_d1_a0'])
        print("Energy at distance 3, angle 90°:", features['energy_d3_a90'])
    except:
        print("Please provide a valid image path")

if __name__ == "__main__":
    main()