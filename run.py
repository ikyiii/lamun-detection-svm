import subprocess
import os
import sys

def main():
    print("1. Train model")
    print("2. Run Streamlit app")
    choice = input("Pilih opsi (1/2): ")
    
    if choice == '1':
        print("\nTraining model...")
        # Menjalankan train_model.py yang ada di folder scripts
        subprocess.run([sys.executable, os.path.join("scripts", "train_model.py")])
    elif choice == '2':
        print("\nRunning Streamlit app...")
        subprocess.run(["streamlit", "run", "app.py"])
    else:
        print("Pilihan tidak valid")

if __name__ == "__main__":
    main()