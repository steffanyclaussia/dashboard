# fomo.py
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import re
import os

# ===== Konfigurasi Halaman =====
st.set_page_config(
    page_title="Dashboard FOMO dan Kesejahteraan Mahasiswa",
    layout="wide"
)

# ===== Load Dataset =====
if os.path.exists("Data Threeasure_Cleaned.csv"):
    df = pd.read_csv("Data Threeasure_Cleaned.csv")
else:
    st.warning("File `Data Threeasure_Cleaned.csv` tidak ditemukan. Silakan upload CSV.")
    uploaded = st.file_uploader("Upload file CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        st.stop()

# ===== Preprocessing =====
# Bersihkan nama kolom
df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]

# Konversi angka dari format rupiah
def currency_to_float(x):
    if pd.isna(x): return np.nan
    s = str(x)
    s = s.replace("Rp", "").replace("rp", "").replace(".", "").replace(",", "")
    s = re.sub(r"[^0-9]", "", s)
    return float(s) if s else np.nan

if "rata-rata_uang_saku_perbulan" in df.columns:
    df["uang_saku_num"] = df["rata-rata_uang_saku_perbulan"].apply(currency_to_float)
if "pengeluaran_untuk_fomo_per_bulan" in df.columns:
    df["pengeluaran_fomo_num"] = df["pengeluaran_untuk_fomo_per_bulan"].apply(currency_to_float)

# Tambahkan kategori FOMO
if "frekuensi_fomo_pengeluaran" in df.columns:
    fomo_map = {1:"Tidak Pernah",2:"Jarang",3:"Kadang-kadang",4:"Sering",5:"Sangat Sering"}
    df["kategori_fomo"] = df["frekuensi_fomo_pengeluaran"].map(fomo_map)
    df["skor_fomo"] = df["frekuensi_fomo_pengeluaran"]

# Tambahkan kategori keuangan
if "kemampuan_mengelola_keuangan" in df.columns:
    keuangan_map = {1:"Buruk",2:"Kurang Baik",3:"Cukup Baik",4:"Baik",5:"Sangat Baik"}
    df["kategori_keuangan"] = df["kemampuan_mengelola_keuangan"].map(keuangan_map)
    df["skor_keuangan"] = df["kemampuan_mengelola_keuangan"]

# Tambahkan kategori pengeluaran FOMO
if "uang_saku_num" in df.columns and "pengeluaran_fomo_num" in df.columns:
    df["proporsi_fomo"] = df["pengeluaran_fomo_num"] / df["uang_saku_num"]
    df["kategori_pengeluaran"] = pd.cut(
        df["proporsi_fomo"],
        bins=[-np.inf,0.2,0.5,np.inf],
        labels=["Rendah (<20%)","Sedang (20–50%)","Tinggi (>50%)"]
    )

# Skor psikologis (jumlah dari indikator)
psiko_cols = [
    "pengaruh_fomo_terhadap_emosi",
    "frekuensi_stres_karena_finansial",
    "frekuensi_hilang_semangat_kuliah_karena_tekanan_finansial",
    "frekuensi_stres_fomo"
]
for c in psiko_cols:
    if c not in df.columns:
        df[c] = 0
df["skor_psikologis"] = df[psiko_cols].sum(axis=1)

# ===== Warna Palet =====
custom_palette = ["#FDA19B", "#E47A7B", "#CB5D66", "#B14454", "#982E46", "#7F1D3A", "#660F2F"]

# ===== Sidebar Navigasi =====
st.sidebar.title("Navigasi Menu")
menu = st.sidebar.radio(
    "Pilih Halaman:",
    ["Pendahuluan", "Visualisasi Data", "Kesimpulan"]
)

# ===== Halaman: Pendahuluan =====
if menu == "Pendahuluan":
    st.title("Analisis FOMO, Pengelolaan Keuangan, dan Kesejahteraan Psikologis Mahasiswa")

    st.markdown("""
    Penelitian ini menganalisis tiga aspek utama yang mempengaruhi kehidupan mahasiswa masa kini:
    - *Fear of Missing Out (FOMO)*: rasa takut tertinggal dari tren dan aktivitas sosial.
    - *Pengelolaan Keuangan*: kemampuan mahasiswa mengatur uang dengan bijak.
    - *Kesejahteraan Psikologis*: kondisi emosional yang stabil dan seimbang.
    
    Dashboard ini membantu memahami hubungan ketiga variabel tersebut secara visual.
    """)

    st.image(
        "https://cdn.pixabay.com/photo/2018/03/30/08/29/people-3271252_1280.jpg",
        caption="Ilustrasi Mahasiswa dan Media Sosial",
        use_container_width=True
    )

# ===== Halaman: Visualisasi Data =====
elif menu == "Visualisasi Data":
    sub = st.sidebar.selectbox(
        "Pilih Visualisasi:",
        [
            "Distribusi Fakultas",
            "Proporsi Pengeluaran FOMO",
            "Distribusi FOMO",
            "Hubungan FOMO vs Keuangan",
            "Hubungan FOMO vs Psikologis",
            "Hubungan Keuangan vs Psikologis"
        ]
    )

    st.title(sub)

    if sub == "Distribusi Fakultas":
        fig = px.histogram(df, x="fakultas", color="fakultas",
                           color_discrete_sequence=custom_palette,
                           title="Distribusi Fakultas Mahasiswa")
        st.plotly_chart(fig, use_container_width=True)

    elif sub == "Proporsi Pengeluaran FOMO":
        fig = px.pie(df, names="kategori_pengeluaran",
                     color_discrete_sequence=custom_palette,
                     title="Proporsi Pengeluaran FOMO Mahasiswa")
        st.plotly_chart(fig, use_container_width=True)

    elif sub == "Distribusi FOMO":
        fig = px.histogram(df, x="kategori_fomo", color="kategori_fomo",
                           color_discrete_sequence=custom_palette,
                           title="Distribusi Tingkat FOMO")
        st.plotly_chart(fig, use_container_width=True)

    elif sub == "Hubungan FOMO vs Keuangan":
        fig = px.scatter(df, x="skor_fomo", y="skor_keuangan", color="kategori_fomo",
                         color_discrete_sequence=custom_palette,
                         title="Hubungan antara FOMO dan Kemampuan Pengelolaan Keuangan")
        st.plotly_chart(fig, use_container_width=True)

    elif sub == "Hubungan FOMO vs Psikologis":
        fig = px.scatter(df, x="skor_fomo", y="skor_psikologis", color="kategori_fomo",
                         color_discrete_sequence=custom_palette,
                         title="Hubungan antara FOMO dan Kesejahteraan Psikologis")
        st.plotly_chart(fig, use_container_width=True)

    elif sub == "Hubungan Keuangan vs Psikologis":
        fig = px.scatter(df, x="skor_keuangan", y="skor_psikologis", color="kategori_keuangan",
                         color_discrete_sequence=custom_palette,
                         title="Hubungan antara Kemampuan Keuangan dan Kesejahteraan Psikologis")
        st.plotly_chart(fig, use_container_width=True)

# ===== Halaman: Kesimpulan =====
elif menu == "Kesimpulan":
    st.title("Kesimpulan")
    st.markdown("""
    - Tingkat FOMO tinggi cenderung berkaitan dengan pengelolaan keuangan yang kurang baik.
    - Mahasiswa dengan pengelolaan keuangan yang baik memiliki kesejahteraan psikologis yang lebih tinggi.
    - Literasi keuangan dan kesadaran media sosial menjadi kunci kesejahteraan mahasiswa.
    """)
