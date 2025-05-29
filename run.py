import subprocess
import os
import sys
import streamlit as st

def main():
    st.set_page_config(
        page_title="Lamun Classifier - Klasifikasi Jenis Lamun dengan SVM",
        page_icon="🌿",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    # Custom CSS untuk styling
    st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 10px 24px;
            font-weight: bold;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .title {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 20px;
        }
        .option-container {
            display: flex;
            justify-content: center;
            margin: 30px 0;
        }
        .option-card {
            border-radius: 10px;
            padding: 20px;
            background-color: white;
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
            width: fit-content;
            margin: 0 auto;
        }
        .option-card:hover {
            box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
        }
        .stRadio>div {
            display: flex;
            justify-content: center;
        }
        .button-container {
            display: flex;
            justify-content: center;
            margin: 20px 0;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header dengan gambar dan judul
    st.markdown("<h1 class='title'>Lamun Classifier - Klasifikasi Jenis Lamun dengan SVM</h1>", unsafe_allow_html=True)
    st.markdown("<h5 class='title'>Pilih opsi di bawah untuk memulai: </h5>", unsafe_allow_html=True)

    # Container untuk option card di tengah
    with st.container():
        st.markdown('<div class="option-container">', unsafe_allow_html=True)
        
        # Opsi dalam bentuk card
        option = st.radio(
            "Pilih mode:",
            options=["Train Model", "Run Aplikasi"],
            format_func=lambda x: " " + x,
            horizontal=True,
            label_visibility="hidden"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Tombol di tengah dengan container khusus
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    if st.button(f"Jalankan {option}", key="run_button"):
        if option == "Train Model":
            with st.spinner("Sedang melatih model, harap tunggu..."):
                try:
                    subprocess.run([sys.executable, os.path.join("scripts", "train_model.py")])
                    st.success("Pelatihan model selesai!")
                except Exception as e:
                    st.error(f"Terjadi error: {str(e)}")
        
        elif option == "Run Aplikasi":
            st.info("Membuka aplikasi Streamlit di tab baru...")
            try:
                subprocess.Popen(["streamlit", "run", os.path.join("scripts", "app.py")])
            except Exception as e:
                st.error(f"Terjadi error: {str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)

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