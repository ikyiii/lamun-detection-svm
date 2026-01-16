# 🌊 Lamun Detection using Support Vector Machines (SVM)

<div align="center">

<!-- TODO: Add project logo (e.g., an icon representing sea cucumber or detection) -->

[![GitHub stars](https://img.shields.io/github/stars/ikyiii/lamun-detection-svm?style=for-the-badge)](https://github.com/ikyiii/lamun-detection-svm/stargazers)

[![GitHub forks](https://img.shields.io/github/forks/ikyiii/lamun-detection-svm?style=for-the-badge)](https://github.com/ikyiii/lamun-detection-svm/network)

[![GitHub issues](https://img.shields.io/github/issues/ikyiii/lamun-detection-svm?style=for-the-badge)](https://github.com/ikyiii/lamun-detection-svm/issues)

[![GitHub license](https://img.shields.io/badge/license-Unlicensed-blue.svg?style=for-the-badge)](LICENSE)

**A Python-based project for automatically detecting sea cucumbers (Lamun) in images using Histogram of Oriented Gradients (HOG) features and a Support Vector Machine (SVM) classifier.**

</div>

## 📖 Overview

This repository presents a machine learning solution for the detection of sea cucumbers, locally known as Lamun, within image datasets. The project leverages classical computer vision techniques, specifically **Histogram of Oriented Gradients (HOG)** for feature extraction, combined with a **Support Vector Machine (SVM)** for classification. It includes scripts for data preparation, model training, evaluation, and prediction, aiming to provide an accessible framework for Lamun monitoring and research.

## ✨ Features

-   **HOG Feature Extraction:** Efficiently extracts discriminative HOG features from images for robust object representation.
-   **Support Vector Machine (SVM) Classifier:** Implements and trains an SVM model for accurate binary classification (Lamun/Non-Lamun).
-   **Dataset Preprocessing:** Handles image loading, resizing, and preparation for model training and evaluation.
-   **Imbalanced Dataset Handling:** Utilizes `imbalanced-learn` to apply techniques like SMOTE, addressing class imbalance issues common in real-world datasets and improving model generalization.
-   **Model Persistence:** Saves trained HOG descriptors and SVM models to disk using `joblib` for reusability without retraining.
-   **Comprehensive Evaluation:** Generates detailed classification reports, confusion matrices, and performance metrics to assess model effectiveness.
-   **Prediction Module:** Provides functionality to use the trained model for detecting Lamun in new, unseen images.
-   **Visual Studio Code Dev Container Support:** Offers a reproducible development environment with all necessary dependencies pre-configured.

## 🖥️ Screenshots

<!-- TODO: Add actual screenshots of:
- Example images with detected Lamun bounding boxes or highlights
- Classification report visualization
- Confusion matrix plot
- Feature visualization (e.g., HOG features overlayed on an image)
-->

## 🛠️ Tech Stack

**Programming Language:**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

**Machine Learning & Data Science:**

![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

![NumPy](https://img.shields.io/badge/numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)

![Pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

![Imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-green?style=for-the-badge)

**Image Processing:**

![OpenCV](https://img.shields.io/badge/OpenCV-2962FF?style=for-the-badge&logo=opencv&logoColor=white)

![Pillow](https://img.shields.io/badge/Pillow-4F7C8F?style=for-the-badge)

![Scikit-image](https://img.shields.io/badge/scikit--image-003366?style=for-the-badge)

**Data Visualization:**

![Matplotlib](https://img.shields.io/badge/Matplotlib-B33B3B?style=for-the-badge&logo=matplotlib&logoColor=white)

![Seaborn](https://img.shields.io/badge/Seaborn-4D88AE?style=for-the-badge&logo=seaborn&logoColor=white)

**Utilities:**

![Joblib](https://img.shields.io/badge/joblib-FFBB00?style=for-the-badge)

![tqdm](https://img.shields.io/badge/tqdm-blue?style=for-the-badge)

**Development Environment:**

![VS Code](https://img.shields.io/badge/VS%20Code-007ACC?style=for-the-badge&logo=visual-studio-code&logoColor=white)

![Dev Containers](https://img.shields.io/badge/Dev%20Containers-blue?style=for-the-badge&logo=visual-studio-code&logoColor=white)

## 🚀 Quick Start

Follow these steps to get the project up and running on your local machine.

### Prerequisites
-   **Python 3.x** (version 3.8 or higher is generally recommended for modern ML libraries)
-   It is highly recommended to use a virtual environment to manage dependencies.

### 1. Clone the repository
```bash
git clone https://github.com/ikyiii/lamun-detection-svm.git
cd lamun-detection-svm
```

### 2. Set up a Python Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate # On Windows use `venv\Scripts\activate`
```

### 3. Install dependencies
Install all required Python packages using `pip`:
```bash
pip install -r requirements.txt
```

### 4. Prepare your dataset
Place your image dataset into the `dataset/` directory. The `run.py` script expects the dataset to be organized into subdirectories representing different classes (e.g., `lamun/` for positive samples and `non_lamun/` for negative samples).

```
dataset/
├── lamun/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
└── non_lamun/
    ├── image_a.jpg
    ├── image_b.png
    └── ...
```

### 5. Run the detection process
Execute the main script to perform feature extraction, model training, evaluation, and potentially prediction.
```bash
python run.py
```
The script will output progress and logs to the console and save detailed logs to the `logs/` directory. Trained models (HOG descriptor and SVM classifier) will be saved to `model/`. Evaluation reports, such as classification reports and confusion matrices, will be saved in the `reports/` directory.

## 📁 Project Structure

```
lamun-detection-svm/
├── .devcontainer/       # Configuration files for VS Code Dev Containers, enabling a reproducible development environment.
├── .vscode/             # Visual Studio Code specific settings and configurations for the workspace.
├── dataset/             # Stores the raw or preprocessed image datasets, categorized by class.
├── logs/                # Contains log files generated during the execution of scripts (training, evaluation).
├── model/               # Directory for saving trained machine learning models (e.g., serialized HOG descriptor, SVM classifier).
├── reports/             # Stores generated evaluation outputs such as classification reports, confusion matrices, and plots.
├── scripts/             # Collection of utility Python scripts for tasks like data preparation, feature engineering, or helper functions.
├── summary/             # Intended for high-level summaries or aggregated results, potentially from multiple runs.
├── requirements.txt     # Lists all Python package dependencies required for the project.
├── run.py               # The main entry point script to orchestrate the entire Lamun detection pipeline.
└── README.md            # Project documentation, providing an overview and usage instructions.
```

## ⚙️ Development

### Using Dev Containers
This project includes a `.devcontainer` configuration, allowing you to open it directly in a consistent and pre-configured development environment using Visual Studio Code and Docker.

1.  **Install Docker** on your system.
2.  Install the **Remote - Containers** extension for Visual Studio Code.
3.  Open the `lamun-detection-svm` project folder in VS Code.
4.  VS Code will prompt you to "Reopen in Container". Click this option.
5.  The development container will be built (if not already cached), and all dependencies from `requirements.txt` will be automatically installed within the container, providing a ready-to-use environment.

### Available Scripts
The primary script for executing the entire machine learning pipeline is `run.py`.

*   **`python run.py`**: Executes the main workflow, which typically involves:
    1.  Loading and preprocessing images from `dataset/`.
    2.  Extracting HOG features from the preprocessed images.
    3.  Training an SVM model on the extracted features.
    4.  Evaluating the trained model on a test set and saving various reports (classification report, confusion matrix) to `reports/`.
    5.  Saving the trained HOG descriptor and SVM model to `model/` for future use.

Refer to the `run.py` script's source code for potential command-line arguments or configuration options to customize specific behaviors (e.g., specifying model parameters, adjusting dataset paths, or toggling between training and prediction modes).

## 🧪 Results & Reporting

After successfully running `python run.py`, the following key outputs will be generated:

*   **`model/`**: This directory will contain the serialized machine learning artifacts. Typically, you will find files like `hog_descriptor.pkl` (the trained HOG feature extractor) and `svm_model.pkl` (the trained Support Vector Machine classifier). These files allow you to load and use the model without needing to retrain it.
*   **`reports/`**: This directory will house visual and textual reports crucial for understanding the model's performance. Expected contents include:
    *   **Classification reports:** Detailed metrics such as precision, recall, F1-score, and support for each class (Lamun and Non-Lamun), along with overall accuracy.
    *   **Confusion matrices:** Visual representations that show the number of correct and incorrect predictions made by the classification model, broken down by class.
    *   (Potentially) other plots or analysis specific to the model's performance, such as ROC curves if implemented.
*   **`logs/`**: This directory stores detailed execution logs from the various stages of the pipeline. These logs are invaluable for debugging, monitoring training progress, and understanding the flow of operations.

## 🤝 Contributing

Contributions are welcome! If you'd like to improve this project, please consider:

1.  Forking the repository.
2.  Creating a new branch (`git checkout -b feature/your-feature-name`).
3.  Making your changes and ensuring they adhere to the project's coding style.
4.  Committing your changes (`git commit -m 'Add new feature'`).
5.  Pushing to the branch (`git push origin feature/your-feature-name`).
6.  Opening a Pull Request with a clear description of your changes.

Please ensure your code includes appropriate comments and documentation where necessary.

## 📄 License

This project is currently **Unlicensed**. Users are advised to contact the repository owner for licensing details regarding usage and distribution.
<!-- TODO: If a specific license is intended, please add a LICENSE file (e.g., MIT, Apache 2.0) to the repository root and update this section to reference it. -->

## 🙏 Acknowledgments

-   **scikit-learn:** For providing robust and efficient machine learning tools.
-   **OpenCV:** For powerful and comprehensive image processing capabilities.
-   **NumPy & Pandas:** For foundational support in numerical operations and data manipulation.
-   **Matplotlib & Seaborn:** For their extensive data visualization functionalities.
-   **imbalanced-learn:** For specialized tools to handle imbalanced datasets effectively.
-   **joblib & tqdm:** For utility functionalities such as model persistence and progress indicators.
-   **scikit-image:** For additional image processing algorithms.
-   The maintainers and contributors of all open-source Python libraries used in this project.

## 📞 Support & Contact

-   🐛 Issues: [GitHub Issues](https://github.com/ikyiii/lamun-detection-svm/issues)

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

Made with ❤️ by [ikyiii](https://github.com/ikyiii)

</div>

