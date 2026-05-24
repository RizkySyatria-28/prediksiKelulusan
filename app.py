# =========================
# IMPORT LIBRARY
# =========================

from altair import X
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# =========================
# KONFIGURASI HALAMAN
# =========================

st.set_page_config(
    page_title="Prediksi Kelulusan Mahasiswa",
    page_icon="🎓",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.stMetric {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.1);
}

h1 {
    color: #1f3c88;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL & DATASET
# =========================

model = joblib.load("model_kelulusan.pkl")

df = pd.read_excel(
    "dataset_kelulusan_final.xlsx"
)

# =========================
# SIDEBAR
# =========================

st.sidebar.title("Login Sebagai")

role = st.sidebar.selectbox(
    "Pilih User",
    [
        "Pimpinan",
        "Mahasiswa"
    ]
)

# =========================
# MENU
# =========================

if role == "Pimpinan":

    menu = st.sidebar.selectbox(
        "Menu",
        [
            "Dashboard",
            "Early Warning",
            "Prediksi Data Real"
        ]
    )

else:

    menu = st.sidebar.selectbox(
        "Menu",
        [
            "Prediksi Kelulusan"
        ]
    )

# =========================
# DASHBOARD PIMPINAN
# =========================

if role == "Pimpinan" and menu == "Dashboard":

    st.title("🎓 Dashboard Monitoring Mahasiswa")

    # =========================
    # METRIC
    # =========================

    total_mhs = len(df)

    total_ontime = len(
        df[df['Ontime'] == 1]
    )

    total_tidak = len(
        df[df['Ontime'] == 0]
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🎓 Total Mahasiswa",
            total_mhs
        )

    with col2:
        st.metric(
            "✅ Ontime",
            total_ontime
        )

    with col3:
        st.metric(
            "⚠️ Tidak Ontime",
            total_tidak
        )

    # =========================
    # GRAFIK DISTRIBUSI
    # =========================

    st.subheader("Distribusi Kelulusan")

    fig, ax = plt.subplots()

    sns.countplot(
        data=df,
        x='Ontime',
        ax=ax
    )

    ax.set_xticklabels([
        'Tidak Ontime',
        'Ontime'
    ])

    st.pyplot(fig)

    # =========================
    # PIE CHART
    # =========================

    st.subheader("Persentase Kelulusan")

    fig3, ax3 = plt.subplots()

    labels = [
        'Ontime',
        'Tidak Ontime'
    ]

    sizes = [
        total_ontime,
        total_tidak
    ]

    ax3.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%'
    )

    st.pyplot(fig3)

    # =========================
    # GRAFIK STATUS SKRIPSI
    # =========================

    st.subheader("Distribusi Status Skripsi")

    fig2, ax2 = plt.subplots()

    sns.countplot(
        data=df,
        x='Status_Skripsi',
        ax=ax2
    )

    st.pyplot(fig2)

# =========================
# EARLY WARNING SYSTEM
# =========================

elif role == "Pimpinan" and menu == "Early Warning":

    st.title("⚠️ Early Warning System")

    risiko = df[
        (
            (df['IPK'] < 3.00)
            |
            (df['MK_Gagal'] > 2)
            |
            (df['Jumlah_Bimbingan'] < 10)
            |
            (df['Status_Skripsi'] < 2)
        )
    ]

    st.subheader(
        "Daftar Mahasiswa Berisiko"
    )

    st.dataframe(risiko)

    st.warning("""
    Rekomendasi Tindakan:
    - Tingkatkan monitoring akademik
    - Tambahkan intensitas bimbingan
    - Evaluasi progres skripsi
    - Konsultasi dengan dosen wali
    """)

# =========================
# PREDIKSI DATA REAL
# =========================

elif role == "Pimpinan" and menu == "Prediksi Data Real":

    st.title("📊 Prediksi Data Real Angkatan 2023")

    uploaded_file = st.file_uploader(
        "Upload File CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        # =========================
        # LOAD DATA
        # =========================

        data_real = pd.read_csv(
            uploaded_file
        )

        st.subheader("Dataset Real")

        st.dataframe(data_real)

        # =========================
        # PREPROCESSING
        # =========================

        # Encoding Status Skripsi
        status_map = {
            "Belum": 0,
            "Judul ACC": 1,
            "Sempro": 2,
            "Semhas": 3,
            "Skripsi": 4
        }

        data_real['Status_Skripsi'] = (
            data_real['Status_Skripsi']
            .map(status_map)
        )

        # =========================
        # FEATURE ENGINEERING
        # =========================

        data_real['Rata_Rata_IP'] = (
            data_real['IPS_S1'] +
            data_real['IPS_S2'] +
            data_real['IPS_S3'] +
            data_real['IPS_S4']
        ) / 4

        data_real['Progress_SKS'] = (
            data_real['SKS_Lulus'] / 144
        )

        data_real['Tren_IP'] = (
            data_real['IPS_S4'] -
            data_real['IPS_S1']
        )

        # =========================
        # PREDIKSI
        # =========================

        hasil_prediksi = model.predict(
            data_real
        )

        # =========================
        # HASIL
        # =========================

        data_real['Prediksi'] = hasil_prediksi

        # Konversi label
        data_real['Prediksi'] = (
            data_real['Prediksi']
            .map({
                1: 'Ontime',
                0: 'Tidak Ontime'
            })
        )

        st.subheader("Hasil Prediksi")

        st.dataframe(data_real)

        # =========================
        # MAHASISWA RISIKO
        # =========================

        risiko = data_real[
            data_real['Prediksi']
            == 'Tidak Ontime'
        ]

        st.subheader(
            "⚠️ Mahasiswa Berisiko"
        )

        st.dataframe(risiko)

        # =========================
        # GRAFIK
        # =========================

        st.subheader(
            "Grafik Hasil Prediksi"
        )

        fig, ax = plt.subplots()

        sns.countplot(
            data=data_real,
            x='Prediksi',
            ax=ax
        )

        st.pyplot(fig)

        # =========================
        # REKOMENDASI
        # =========================

        st.warning("""
        Rekomendasi:
        - Tingkatkan monitoring akademik
        - Tambahkan intensitas bimbingan
        - Evaluasi progres skripsi
        - Konsultasi dengan dosen wali
        """)

# =========================
# MENU MAHASISWA
# =========================

elif role == "Mahasiswa":

    st.title("🤖 Prediksi Kelulusan Mahasiswa")

    # =========================
    # INPUT
    # =========================

    tahun_masuk = st.number_input(
        "Tahun Masuk",
        2017,
        2025,
        2021
    )

    ips1 = st.slider(
        "IPS Semester 1",
        0.0,
        4.0,
        3.0
    )

    ips2 = st.slider(
        "IPS Semester 2",
        0.0,
        4.0,
        3.0
    )

    ips3 = st.slider(
        "IPS Semester 3",
        0.0,
        4.0,
        3.0
    )

    ips4 = st.slider(
        "IPS Semester 4",
        0.0,
        4.0,
        3.0
    )

    ipk = st.slider(
        "IPK",
        0.0,
        4.0,
        3.0
    )

    mk_gagal = st.number_input(
        "Jumlah MK Gagal",
        0,
        20,
        1
    )

    sks_lulus = st.number_input(
        "SKS Lulus",
        0,
        160,
        140
    )

    status_skripsi = st.selectbox(
        "Status Skripsi",
        [
            "Belum",
            "Judul ACC",
            "Sempro",
            "Semhas",
            "Skripsi"
        ]
    )

    jumlah_bimbingan = st.number_input(
        "Jumlah Bimbingan",
        0,
        50,
        5
    )

    # =========================
    # ENCODING MANUAL
    # =========================

    status_map = {
        "Belum": 0,
        "Judul ACC": 1,
        "Sempro": 2,
        "Semhas": 3,
        "Skripsi": 4
    }

    status_encoded = status_map[
        status_skripsi
    ]

    # =========================
    # FEATURE ENGINEERING
    # =========================

    rata_ip = (
        ips1 +
        ips2 +
        ips3 +
        ips4
    ) / 4

    progress_sks = (
        sks_lulus / 144
    )

    tren_ip = (
        ips4 - ips1
    )

    # =========================
    # PREDIKSI
    # =========================

    if st.button("Prediksi"):

        data = [[
            tahun_masuk,
            ips1,
            ips2,
            ips3,
            ips4,
            ipk,
            mk_gagal,
            sks_lulus,
            status_encoded,
            jumlah_bimbingan,
            rata_ip,
            progress_sks,
            tren_ip
        ]]

        prediksi = model.predict(data)

        if prediksi[0] == 1:

            st.success(
                "✅ Mahasiswa Diprediksi Lulus Tepat Waktu"
            )

        else:

            st.error(
                "⚠️ Mahasiswa Berisiko Tidak Lulus Tepat Waktu"
            )

            st.warning("""
            Rekomendasi:
            - Tingkatkan bimbingan akademik
            - Monitoring progres skripsi
            - Evaluasi mata kuliah gagal
            - Konsultasi dengan dosen wali
            """)

st.markdown("""
---
Sistem Prediksi Kelulusan Mahasiswa  
Menggunakan Machine Learning Random Forest & CRISP-DM
""")