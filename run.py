import subprocess
import os
import sys
import time
import streamlit as st

def main():
    st.set_page_config(
        page_title="Lamun Classifier - Klasifikasi Jenis Lamun dengan SVM",
        page_icon="🌿",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    # CSS Styling
    st.markdown("""
    <style>
        .main {
            background-color: #f9fbfc;
        }
        .title {
            text-align: center;
            color: #2c3e50;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 40px;
        }
        .card-container {
            display: flex;
            justify-content: center;
            gap: 30px;
            flex-wrap: wrap;
            margin-top: 30px;
        }
        .card {
            background-color: white;
            border-radius: 12px;
            padding: 25px 20px;
            width: 260px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s, box-shadow 0.2s;
            cursor: pointer;
            text-align: center;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15);
        }
        .card-title {
            font-size: 1.3em;
            font-weight: bold;
            margin-bottom: 10px;
            color: #2c3e50;
        }
        .card-desc {
            font-size: 0.9em;
            color: #555;
        }
        .notification-area {
            margin-top: 30px;
            display: flex;
            justify-content: center;
        }
        .footer {
            text-align: center;
            color: #aaa;
            margin-top: 50px;
            font-size: 0.9em;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("<h1 class='title'>🌿 Lamun Classifier</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Klasifikasi Jenis Lamun Menggunakan Algoritma Support Vector Machine (SVM)</p>", unsafe_allow_html=True)

    # Area notifikasi global di tengah
    notification_area = st.empty()

    def show_notification(message, type="info", duration=10):
        """Menampilkan notifikasi selama beberapa detik lalu menghilangkan."""
        with notification_area.container():
            if type == "success":
                notif = st.success(message)
            elif type == "error":
                notif = st.error(message)
            else:
                notif = st.info(message)
            time.sleep(duration)
            notification_area.empty()

    # Card container
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("🧪 Train Model", use_container_width=True):
            with st.spinner("Melatih model SVM... Mohon tunggu sejenak."):
                try:
                    subprocess.run([sys.executable, os.path.join("scripts", "train_model.py")])
                    show_notification("✅ Pelatihan model berhasil diselesaikan!", "success")
                except Exception as e:
                    show_notification(f"❌ Terjadi kesalahan saat melatih model: {str(e)}", "error")

    with col2:
        if st.button("🚀 Run Aplikasi", use_container_width=True):
            with st.spinner("Membuka aplikasi Streamlit..."):
                try:
                    subprocess.Popen(["streamlit", "run", os.path.join("scripts", "app.py")])
                    show_notification("🔄 Membuka aplikasi Streamlit di tab baru...", "info")
                except Exception as e:
                    show_notification(f"❌ Gagal membuka aplikasi: {str(e)}", "error")

    # Footer
    st.markdown("---")
    st.markdown("<div class='footer'>© 2025 Lamun Classifier | Senggarang Selatan</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()