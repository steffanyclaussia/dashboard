# dashboard_threeasure_full.py
# Dashboard Threeasure (3 Halaman)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Threeasure — Dashboard FOMO & Kesejahteraan", layout="wide")

# ---------------------------
# Color Palette
# ---------------------------
PALET_WARNA = ["#FDA19B", "#E47A7B", "#CB5D66", "#B14454", "#982E46", "#7F1D3A", "#660F2F"]

PRIMARY = "#7F1D3A"
SECONDARY = "#B14454"
ACCENT = "#E47A7B"
PAGE_BG = "#FFF8F8"

# ---------------------------
# CSS
# ---------------------------
st.markdown(f"""
<style>
* {{ font-family: 'Times New Roman', Times, serif !important; }}
.stApp {{ background-color: {PAGE_BG}; }}
.kpi {{
    background: linear-gradient(135deg, #FDA19B, #E47A7B, #CB5D66);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    color: #3B0A1A;
    box-shadow: 0px 6px 15px rgba(203, 93, 102, 0.25);
}}
.kpi h3 {{
    font-size: 22px;
    margin: 0;
    font-weight: bold;
}}
.kpi .small {{
    font-size: 14px;
    color: #4A0D1A;
    margin-bottom: 6px;
}}
.card {{
    background: #FFF1F1;
    padding: 14px;
    border-radius: 10px;
    margin-top: 10px;
}}
.footer {{
    background:{PRIMARY};
    color:white;
    text-align:center;
    padding:14px;
    border-radius:10px;
}}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Load data
# ---------------------------
@st.cache_data
def load_df(path="Data Eda Threeasure_Updated.csv"):
    return pd.read_csv(path)

try:
    df = load_df()
except FileNotFoundError:
    st.error("File data tidak ditemukan. Pastikan file 'Data Eda Threeasure_Updated.csv' ada.")
    st.stop()

# ---------------------------
# Column Mapping & Preprocessing
# ---------------------------
cols = [c.lower().strip() for c in df.columns]
colmap = {c.lower().strip(): c for c in df.columns}

def find_col(keywords):
    for k in keywords:
        for key in colmap:
            if k.lower() in key:
                return colmap[key]
    return None

# Column detection
col_fakultas = find_col(["fakultas"])
col_uang_saku = find_col(["uang", "saku"]) or find_col(["x1"])
col_pengeluaran_fomo = find_col(["pengeluaran", "fomo"]) or find_col(["x2"])
col_kemampuan = find_col(["kemampuan", "mengelola", "keuangan"]) or find_col(["x3"])
col_fomo_text = find_col(["sering_merasa_fomo", "sering merasa fomo", "fomo"])
col_freq_fomo = find_col(["frekuensi", "tingkat", "skor", "x4"])
col_kesejahteraan_explicit = find_col(["kesejahteraan_psikologis", "kesejahteraan"])
col_distress_1 = find_col(["pengaruh_emosi", "emosi", "x5"])
col_distress_2 = find_col(["frekuensi_stres_fin", "stres_fin", "x6"])
col_distress_3 = find_col(["hilang_semangat", "hilang semangat", "x7"])
col_distress_4 = find_col(["frekuensi_stres_fomo", "stres_fomo", "x8"])

def rupiah_to_num(x):
    try:
        if pd.isna(x):
            return np.nan
        s = str(x)
        s = s.replace("Rp", "").replace("rp", "").replace(",", "").replace(".", "").strip()
        return float(s) if s not in ["", "-"] else np.nan
    except:
        return np.nan

df_work = df.copy()

if col_uang_saku in df_work.columns:
    df_work["uang_saku_num"] = df_work[col_uang_saku].apply(rupiah_to_num)
else:
    df_work["uang_saku_num"] = np.nan

if col_pengeluaran_fomo in df_work.columns:
    df_work["pengeluaran_fomo_num"] = df_work[col_pengeluaran_fomo].apply(rupiah_to_num)
else:
    df_work["pengeluaran_fomo_num"] = np.nan

df_work["proporsi_fomo_pct"] = (df_work["pengeluaran_fomo_num"] / df_work["uang_saku_num"]) * 100
df_work["proporsi_fomo_pct"].replace([np.inf, -np.inf], np.nan, inplace=True)

def safe_numeric(col):
    if col and col in df_work.columns:
        return pd.to_numeric(df_work[col], errors="coerce")
    else:
        return pd.Series([np.nan]*len(df_work), index=df_work.index)

# FOMO mapping
if col_fomo_text and col_fomo_text in df_work.columns:
    mapping = {"tidak": 1, "tidak pernah":1, "ya":5, "sering":4, "sangat sering":5,
               "kadang-kadang":3, "kadang":3, "jarang":2, "jarang sekali":2}
    series = df_work[col_fomo_text].astype(str).str.strip().str.lower()
    mapped = series.map(lambda v: mapping.get(v, np.nan))
    if mapped.notna().sum() >= len(df_work)*0.1:
        df_work["fomo_num"] = mapped
    else:
        parsed = pd.to_numeric(series, errors="coerce")
        df_work["fomo_num"] = parsed
else:
    possible = find_col(["frekuensi", "tingkat", "skor", "x4"])
    df_work["fomo_num"] = safe_numeric(possible)

# Kesejahteraan
if col_kesejahteraan_explicit and col_kesejahteraan_explicit in df_work.columns:
    df_work["kesejahteraan_score"] = pd.to_numeric(df_work[col_kesejahteraan_explicit], errors="coerce")
else:
    distress_cols = [c for c in [col_distress_1, col_distress_2, col_distress_3, col_distress_4] if c and c in df_work.columns]
    if distress_cols:
        for c in distress_cols:
            df_work[c] = pd.to_numeric(df_work[c], errors="coerce")
        df_work["mean_distress"] = df_work[distress_cols].mean(axis=1)
        df_work["kesejahteraan_score"] = 6 - df_work["mean_distress"]
    else:
        df_work["kesejahteraan_score"] = np.nan

df_work["kemampuan_num"] = safe_numeric(col_kemampuan)

if col_fakultas and col_fakultas in df_work.columns:
    df_work["fakultas_clean"] = df_work[col_fakultas].astype(str).str.strip()
else:
    df_work["fakultas_clean"] = "Unknown"

# ================================
# Sidebar Navigasi Halaman
# ================================
page = st.sidebar.radio("📑 Navigasi", [
    "Halaman 1 - Dataset & KPI",
    "Halaman 2 - Visualisasi Data",
    "Halaman 3 - Kesimpulan"
])

# ================================
# Halaman 1: Dataset & KPI
# ================================
if page.startswith("Halaman 1"):
    st.markdown(f"""
<div style="background:{PRIMARY}; padding:22px; border-radius:10px; text-align:center;">
  <h1 style="margin:6px; color:white;">
    Analisis Dampak Fear of Missing Out (FOMO) dan Pengelolaan Keuangan terhadap Kesejahteraan Psikologis Mahasiswa
  </h1>
  <div style="color:white; font-weight:600; margin-top:10px; font-size:16px;">
    Kelompok Threeasure — Steffany Claussia Fernanda (24083010026) • Fanny Widya Cahyani (24083010045) • Izzati Kamila Putri (24083010059)
  </div>
  <div style="font-size:14px; color:white; margin-top:6px;">
    Program Studi Sains Data • UPN "Veteran" Jawa Timur — 2025
  </div>
</div>
""", unsafe_allow_html=True)

    # ---------------------------
    # Deskripsi Dataset (judul saja)
    # ---------------------------
    st.markdown("""
    <div style="background:linear-gradient(135deg,#FDA19B,#E47A7B,#CB5D66);
                padding:14px; border-radius:12px; margin:20px 0; 
                text-align:center; color:white;
                font-family:'Times New Roman', serif;
                font-size:20px; font-weight:bold;
                box-shadow:0px 4px 12px rgba(203,93,102,0.25);">
        DESKRIPSI DATASET
    </div>
    """, unsafe_allow_html=True)


    # ---------------------------
    # Narasi sebelum KPI
    # ---------------------------
    st.markdown("""
    <div style="background:#FFF1F1; padding:16px; border-radius:12px; margin-bottom:20px; font-size:15px; line-height:1.6;">
    <b>Ringkasan Awal:</b><br>
    Bagian ini menyajikan indikator utama dari hasil survei mahasiswa UPNVJT. 
    Melalui <i>Key Performance Indicators (KPI)</i>, dapat dilihat gambaran umum mengenai 
    uang saku, pengeluaran terkait FOMO, kemampuan mengelola keuangan, serta tingkat kesejahteraan psikologis mahasiswa.
    </div>
    """, unsafe_allow_html=True)

    # ---------------------------
    # KPI row
    # ---------------------------
    total_n = len(df_work)
    mean_uang_saku = df_work["uang_saku_num"].mean()
    mean_pengeluaran_fomo = df_work["pengeluaran_fomo_num"].mean()
    mean_kemampuan = df_work["kemampuan_num"].mean()
    mean_kesejahteraan = df_work["kesejahteraan_score"].mean()
    mean_proporsi = df_work["proporsi_fomo_pct"].mean()

    def fmt_money(x):
        if pd.isna(x):
            return "-"
        try:
            return "Rp " + f"{int(round(x)):,}"
        except:
            return str(x)

    val_uang = fmt_money(mean_uang_saku)
    val_pengeluaran = fmt_money(mean_pengeluaran_fomo)
    val_kemampuan = f"{mean_kemampuan:.2f}" if not pd.isna(mean_kemampuan) else "-"
    val_kesejahteraan = f"{mean_kesejahteraan:.2f}" if not pd.isna(mean_kesejahteraan) else "-"
    val_proporsi = f"{mean_proporsi:.1f}%" if not pd.isna(mean_proporsi) else "-"

    # KPI Styling
    st.markdown("""
    <style>
    .kpi {
        background: linear-gradient(135deg, #FDA19B, #E47A7B, #CB5D66);
        border-radius: 18px;
        padding: 20px;
        text-align: center;
        color: #3B0A1A;
        font-family: 'Times New Roman', serif;
        box-shadow: 0px 6px 15px rgba(203, 93, 102, 0.25);
        transition: 0.3s ease-in-out;
    }
    .kpi:hover {
        transform: translateY(-3px);
        box-shadow: 0px 10px 25px rgba(228, 122, 123, 0.35);
    }
    .kpi h3 {
        font-size: 26px;
        margin: 5px 0 0 0;
        font-weight: bold;
        color: #4A0D1A;
    }
    .kpi .small {
        font-size: 15px;
        letter-spacing: 0.3px;
        color: #5B1C26;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # KPI Columns
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(f"<div class='kpi'><div class='small'>Jumlah responden</div><h3>{total_n}</h3></div>", unsafe_allow_html=True)
    with k2:
        st.markdown(f"<div class='kpi'><div class='small'>Rata-rata uang saku</div><h3>{val_uang}</h3></div>", unsafe_allow_html=True)
    with k3:
        st.markdown(f"<div class='kpi'><div class='small'>Rata-rata pengeluaran FOMO</div><h3>{val_pengeluaran}</h3></div>", unsafe_allow_html=True)
    with k4:
        st.markdown(f"<div class='kpi'><div class='small'>Rata-rata kemampuan keuangan</div><h3>{val_kemampuan}</h3></div>", unsafe_allow_html=True)
    with k5:
        st.markdown(f"<div class='kpi'><div class='small'>Rata-rata kesejahteraan psikologis</div><h3>{val_kesejahteraan}</h3></div>", unsafe_allow_html=True)

    # ---------------------------
    # Narasi setelah KPI
    # ---------------------------
    st.markdown(f"""
    <div style="background:#FFF1F1; padding:16px; border-radius:12px; margin-top:20px; font-size:15px; line-height:1.6;">
    <b>Interpretasi Awal:</b><br>
    Dari hasil ringkasan di atas dapat dilihat bahwa rata-rata <b>uang saku</b> mahasiswa adalah {val_uang}, 
    dengan <b>pengeluaran FOMO</b> yang rata-rata mencapai {val_proporsi} dari total uang saku bulanan. 
    Kemampuan keuangan mahasiswa berada pada skor <b>{val_kemampuan}</b>, sedangkan 
    <b>kesejahteraan psikologis</b> mereka berada pada skor rata-rata <b>{val_kesejahteraan}</b>. 
    <br><br>
    Hasil ini menunjukkan adanya kecenderungan bahwa semakin besar pengeluaran FOMO, semakin menurun kesejahteraan psikologis mahasiswa. 
    Hal ini akan dibahas lebih detail pada visualisasi data di halaman berikutnya.
    </div>
    """, unsafe_allow_html=True)

# ================================
# Halaman 2: Visualisasi Data
# ===============================
elif page.startswith("Halaman 2"):
    st.markdown(f"""
    <div style="
        background: linear-gradient(90deg, #FDA19B, #E47A7B, #CB5D66);
        padding: 25px;
        border-radius: 14px;
        text-align: center;
        margin-bottom: 20px;
    ">
        <h1 style="color:white; margin:0;">Visualisasi Data</h1>
        <p style="color:white; font-size:16px; margin-top:6px;">
            Analisis Hubungan FOMO, Pengelolaan Keuangan, dan Kesejahteraan Psikologis
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Distribusi Responden",
        "FOMO → Kesejahteraan",
        "Kemampuan Keuangan → Kesejahteraan",
        "FOMO ↔ Kemampuan Keuangan",
        "Proporsi Pengeluaran FOMO",
        "Korelasi Numerik"
    ])

    # =====================================================
    # TAB 1: Distribusi Responden
    # =====================================================
    with tab1:
        # ---------------------------
        # Section: Distribusi fakultas & Pie FOMO (as in report)
        # ---------------------------
        st.subheader("Distribusi Responden & Proporsi FOMO")
        
        c1, c2 = st.columns([1.4, 1])
        
        # ==========================================================
        # Distribusi Responden per Fakultas
        # ==========================================================
        with c1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.write("**Distribusi responden per fakultas**")
        
            fac_counts = df_work["fakultas_clean"].value_counts().reset_index()
            fac_counts.columns = ["Fakultas", "Jumlah"]
        
            fig_fac = px.bar(
                fac_counts,
                x="Fakultas",
                y="Jumlah",
                text="Jumlah",
                color="Jumlah",
                color_continuous_scale=[
                    "#FDA19B", "#E47A7B", "#CB5D66", "#B14454", "#982E46", "#7F1D3A", "#660F2F"
                ],
                template="simple_white"
            )
            fig_fac.update_layout(
                xaxis_title="",
                yaxis_title="Jumlah responden",
                font_family="Times New Roman",
                title_font_color="#660F2F",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            fig_fac.update_traces(
                texttemplate='%{text}',
                textposition='outside',
                marker_line_color="#7F1D3A",
                marker_line_width=1.2
            )
        
            st.plotly_chart(fig_fac, use_container_width=True)
        
            # Insight dengan background
            st.markdown("""
            <div class="card" style="background:#FDA19B; color:#660F2F;">
                💡 <b>Insight:</b> Responden terbanyak berasal dari Fakultas Ilmu Komputer, menunjukkan partisipasi survei yang paling tinggi.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
            # ==========================================================
            # Proporsi Mahasiswa yang Merasa FOMO
            # ==========================================================
            with c2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.write("**Proporsi mahasiswa yang merasa FOMO**")
            
                if 'col_fomo_text' in locals() and col_fomo_text and col_fomo_text in df_work.columns:
                    pie_series = df_work[col_fomo_text].fillna("Tidak diisi").value_counts()
                    fig_pie = px.pie(
                        names=pie_series.index,
                        values=pie_series.values,
                        color_discrete_sequence=["#FDA19B", "#E47A7B", "#CB5D66", "#B14454", "#982E46", "#7F1D3A", "#660F2F"]
                    )
                    fig_pie.update_traces(
                        textposition='inside',
                        textinfo='percent+label',
                        pull=0   # <-- diset 0 agar menyatu rapat
                    )
                else:
                    if "fomo_num" in df_work.columns and df_work["fomo_num"].notna().sum() > 0:
                        bins = [0, 1.5, 2.5, 3.5, 4.5, 5.5]
                        labels = ["Tidak Pernah", "Jarang", "Kadang-kadang", "Sering", "Sangat Sering"]
                        cat = pd.cut(df_work["fomo_num"].fillna(0), bins=bins, labels=labels)
                        pie_series = cat.value_counts().reindex(labels).fillna(0)
                        fig_pie = px.pie(
                            names=pie_series.index,
                            values=pie_series.values,
                            color_discrete_sequence=["#FDA19B", "#E47A7B", "#CB5D66", "#B14454", "#982E46", "#7F1D3A", "#660F2F"]
                        )
                        fig_pie.update_traces(
                            textposition='inside',
                            textinfo='percent+label',
                            pull=0   # <-- diset 0 agar menyatu rapat
                        )
                    else:
                        st.info("Tidak ada data FOMO yang memadai untuk pie chart.")
                        fig_pie = None
            
                if fig_pie:
                    fig_pie.update_layout(
                        font_family="Times New Roman",
                        title_font_color="#660F2F",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
            
                    # Insight dengan background
                    st.markdown("""
                    <div class="card" style="background:#E47A7B; color:#660F2F;">
                        💡 <b>Insight:</b> Mayoritas mahasiswa tidak merasa FOMO (63,4%), sedangkan 36,6% mengaku merasa FOMO.
                    </div>
                    """, unsafe_allow_html=True)
            
                st.markdown("</div>", unsafe_allow_html=True)

    
    # =====================================================
    # TAB 2: FOMO → Kesejahteraan
    # =====================================================
    with tab2:
        # ---------------------------
        # Section: FOMO -> Kesejahteraan (heatmap & distribusi bar, sampingan)
        # ---------------------------
        st.subheader("Pengaruh FOMO terhadap Kesejahteraan Psikologis")
        PALET_WARNA = ['#FDA19B', '#E47A7B', '#CB5D66', '#B14454', '#982E46', '#7F1D3A', '#660F2F']
        
        if df_work["kesejahteraan_score"].notna().sum() > 0 and df_work["fomo_num"].notna().sum() > 0:
            # Kategorisasi kesejahteraan
            kbins = [-1, 2.5, 3.5, 5.5]
            klabels = ["Buruk", "Cukup Baik", "Baik"]
            df_work["kesejahteraan_cat"] = pd.cut(df_work["kesejahteraan_score"], bins=kbins, labels=klabels)
        
            # Kategorisasi FOMO
            fbins = [0, 1.5, 2.5, 3.5, 4.5, 5.5]
            flabels = ["Tidak Pernah", "Jarang", "Kadang-kadang", "Sering", "Sangat Sering"]
            df_work["fomo_cat"] = pd.cut(df_work["fomo_num"].fillna(0), bins=fbins, labels=flabels)
        
            # Crosstab untuk heatmap
            cross = pd.crosstab(df_work["fomo_cat"], df_work["kesejahteraan_cat"])
            comb = df_work.groupby(["fomo_cat", "kesejahteraan_cat"]).size().reset_index(name="Jumlah")
        
            # Layout dua kolom
            col1, col2 = st.columns(2)
        
            # Heatmap
            with col1:
                st.markdown("**Heatmap Hubungan FOMO dan Kesejahteraan Psikologis**")
                fig_h = px.imshow(
                    cross,
                    text_auto=True,
                    color_continuous_scale=PALET_WARNA,
                    labels=dict(x="Kesejahteraan Psikologis", y="Tingkat FOMO", color="Jumlah Responden")
                )
                fig_h.update_layout(
                    title="Heatmap Hubungan FOMO vs Kesejahteraan",
                    font_family="Times New Roman",
                    title_font_color='#660F2F',
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_h, use_container_width=True)
        
                # Insight khusus heatmap
                st.markdown("""
                <div class="card" style="background:#FDA19B; color:#660F2F;">
                    💡 <b>Insight Heatmap:</b> Mayoritas mahasiswa tanpa FOMO berada pada kesejahteraan Baik (79 responden), sedangkan pada FOMO sangat sering jumlah Baik menurun (40 responden) dan Cukup Baik meningkat (13 responden).
                </div>
                """, unsafe_allow_html=True)
        
            # Distribusi bar (stacked)
            with col2:
                st.markdown("**Distribusi Kesejahteraan Berdasarkan Tingkat FOMO**")
                fig_comb = px.bar(
                    comb,
                    x="fomo_cat",
                    y="Jumlah",
                    color="kesejahteraan_cat",
                    text="Jumlah",
                    barmode="stack",
                    labels={
                        "fomo_cat": "Kategori FOMO",
                        "Jumlah": "Jumlah Responden",
                        "kesejahteraan_cat": "Kategori Kesejahteraan"
                    },
                    color_discrete_sequence=PALET_WARNA[:3]
                )
                fig_comb.update_traces(textposition="outside")
                fig_comb.update_layout(
                    title="Distribusi Kesejahteraan per Kategori FOMO",
                    font_family="Times New Roman",
                    title_font_color='#660F2F',
                    legend_title_text="Kesejahteraan Psikologis",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_comb, use_container_width=True)
        
                # Insight khusus bar chart
                st.markdown("""
                <div class="card" style="background:#E47A7B; color:#660F2F;">
                    💡 <b>Insight Bar:</b> Semakin tinggi tingkat FOMO, proporsi mahasiswa dengan kesejahteraan Baik semakin menurun.
                </div>
                """, unsafe_allow_html=True)
        
        else:
            st.info("Data FOMO numerik dan/atau skor kesejahteraan tidak memadai untuk analisis ini.")
        

    # =====================================================
    # TAB 3: Kemampuan Keuangan → Kesejahteraan
    # =====================================================
    with tab3:
        st.subheader("Pengaruh Kemampuan Mengelola Keuangan terhadap Kesejahteraan Psikologis")
        PALET_WARNA = ['#FDA19B', '#E47A7B', '#CB5D66', '#B14454', '#982E46', '#7F1D3A', '#660F2F']
        
        if "kemampuan_num" in df_work.columns and "kesejahteraan_score" in df_work.columns:
            if df_work["kemampuan_num"].notna().sum() > 0 and df_work["kesejahteraan_score"].notna().sum() > 0:
                # Buat kategori kemampuan dan kesejahteraan
                df_work["kemampuan_cat"] = pd.cut(
                    df_work["kemampuan_num"], 
                    bins=[0, 1.5, 2.5, 3.5, 4.5, 5.5],
                    labels=["Buruk", "Kurang Baik", "Cukup Baik", "Baik", "Sangat Baik"]
                )
        
                df_work["kesejahteraan_cat"] = pd.cut(
                    df_work["kesejahteraan_score"], 
                    bins=[0, 2.5, 3.5, 5.5],
                    labels=["Buruk", "Cukup Baik", "Baik"]
                )
        
                # Data untuk heatmap dan bar chart
                heat_data = pd.crosstab(df_work["kemampuan_cat"], df_work["kesejahteraan_cat"])
                bar_data = df_work.groupby(["kemampuan_cat", "kesejahteraan_cat"]).size().reset_index(name="Jumlah")
        
                # ==========================================================
                # Layout dua kolom
                # ==========================================================
                col1, col2 = st.columns(2)
        
                with col1:
                    st.markdown("**Heatmap Hubungan Kemampuan Keuangan dan Kesejahteraan**")
                    fig_heat = px.imshow(
                        heat_data,
                        text_auto=True,
                        color_continuous_scale=PALET_WARNA,
                        labels=dict(x="Kesejahteraan Psikologis", y="Kemampuan Mengelola Keuangan", color="Jumlah Responden")
                    )
                    fig_heat.update_layout(
                        title="Heatmap Hubungan Kemampuan Keuangan vs Kesejahteraan",
                        font_family="Times New Roman",
                        title_font_color='#660F2F',
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)
        
                    # Insight khusus heatmap
                    st.markdown("""
                    <div class="card" style="background:#FDA19B; color:#660F2F;">
                        💡 <b>Insight Heatmap:</b> Pola warna menunjukkan semakin tinggi kemampuan mengelola keuangan, semakin banyak mahasiswa dengan kesejahteraan baik.
                    </div>
                    """, unsafe_allow_html=True)
        
                with col2:
                    st.markdown("**Distribusi Kesejahteraan Berdasarkan Kemampuan Mengelola Keuangan**")
                    fig_bar = px.bar(
                        bar_data,
                        x="kemampuan_cat",
                        y="Jumlah",
                        color="kesejahteraan_cat",
                        text="Jumlah",
                        barmode="stack",
                        labels={
                            "kemampuan_cat": "Kemampuan Mengelola Keuangan",
                            "Jumlah": "Jumlah Responden",
                            "kesejahteraan_cat": "Kategori Kesejahteraan"
                        },
                        color_discrete_sequence=PALET_WARNA[:3]
                    )
                    fig_bar.update_traces(textposition='outside')
                    fig_bar.update_layout(
                        title="Distribusi Kesejahteraan per Kategori Kemampuan Keuangan",
                        font_family="Times New Roman",
                        title_font_color='#660F2F',
                        legend_title_text="Kesejahteraan Psikologis",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
        
                    # Insight khusus bar chart
                    st.markdown("""
                    <div class="card" style="background:#E47A7B; color:#660F2F;">
                        💡 <b>Insight Bar:</b> Mahasiswa dengan kemampuan keuangan kategori "Baik" dan "Sangat Baik" lebih dominan berada pada kesejahteraan yang tinggi.
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Data kemampuan keuangan atau kesejahteraan tidak memadai untuk analisis ini.")
        else:
            st.warning("Kolom kemampuan_num atau kesejahteraan_score tidak ditemukan di dataset.")
        

    # =====================================================
    # TAB 4: FOMO ↔ Kemampuan Keuangan
    # =====================================================
    with tab4:
        # ---------------------------
        # Section: Hubungan antara FOMO dan Kemampuan Mengelola Keuangan
        # ---------------------------
        st.subheader("Hubungan antara FOMO dan Kemampuan Mengelola Keuangan")
        
        PALET_WARNA = ['#FDA19B', '#E47A7B', '#CB5D66', '#B14454', '#982E46', '#7F1D3A', '#660F2F']
        
        if "fomo_num" in df_work.columns and "kemampuan_num" in df_work.columns:
            if df_work["fomo_num"].notna().sum() > 0 and df_work["kemampuan_num"].notna().sum() > 0:
                # Buat kategori FOMO dan Kemampuan Keuangan
                df_work["fomo_cat"] = pd.cut(
                    df_work["fomo_num"],
                    bins=[0, 1.5, 2.5, 3.5, 4.5, 5.5],
                    labels=["Tidak Pernah", "Jarang", "Kadang-kadang", "Sering", "Sangat Sering"]
                )
        
                df_work["kemampuan_cat"] = pd.cut(
                    df_work["kemampuan_num"],
                    bins=[0, 1.5, 2.5, 3.5, 4.5, 5.5],
                    labels=["Buruk", "Kurang Baik", "Cukup Baik", "Baik", "Sangat Baik"]
                )
        
                # Data untuk heatmap dan stacked bar
                heat_data = pd.crosstab(df_work["fomo_cat"], df_work["kemampuan_cat"])
                bar_data = df_work.groupby(["fomo_cat", "kemampuan_cat"]).size().reset_index(name="Jumlah")
        
                # Layout dua kolom (sampingan)
                col1, col2 = st.columns(2)
        
                # ---------------------------
                # HEATMAP
                # ---------------------------
                with col1:
                    st.markdown("**Heatmap Hubungan FOMO vs Kemampuan Mengelola Keuangan**")
                    fig_heat = px.imshow(
                        heat_data,
                        text_auto=True,
                        color_continuous_scale=PALET_WARNA,
                        labels=dict(x="Kemampuan Mengelola Keuangan", y="Tingkat FOMO", color="Jumlah Responden")
                    )
                    fig_heat.update_layout(
                        title="Heatmap Hubungan FOMO vs Kemampuan Mengelola Keuangan",
                        font_family="Times New Roman",
                        title_font_color="#660F2F",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)
        
                    # Insight heatmap
                    st.markdown("""
                    <div class="card" style="background:#FDA19B; color:#660F2F;">
                        💡 <b>Insight Heatmap:</b> Pola warna memperlihatkan bahwa semakin tinggi tingkat FOMO, 
                        semakin sedikit mahasiswa dengan kemampuan keuangan baik.
                    </div>
                    """, unsafe_allow_html=True)
        
                # ---------------------------
                # STACKED BAR
                # ---------------------------
                with col2:
                    st.markdown("**Distribusi Kemampuan Mengelola Keuangan Berdasarkan Tingkat FOMO**")
                    fig_bar = px.bar(
                        bar_data,
                        x="fomo_cat",
                        y="Jumlah",
                        color="kemampuan_cat",
                        text="Jumlah",
                        barmode="stack",
                        labels={
                            "fomo_cat": "Tingkat FOMO",
                            "Jumlah": "Jumlah Responden",
                            "kemampuan_cat": "Kemampuan Mengelola Keuangan"
                        },
                        color_discrete_sequence=PALET_WARNA[:5]
                    )
                    fig_bar.update_traces(textposition="outside")
                    fig_bar.update_layout(
                        title="Distribusi Kemampuan Keuangan Berdasarkan Tingkat FOMO",
                        font_family="Times New Roman",
                        
                        title_font_color="#660F2F",
                        legend_title_text="Kemampuan Keuangan",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
        
                    # Insight bar chart
                    st.markdown("""
                    <div class="card" style="background:#E47A7B; color:#660F2F;">
                        💡 <b>Insight Bar:</b> Mahasiswa dengan FOMO rendah lebih banyak memiliki kemampuan keuangan 
                        kategori "Baik" hingga "Sangat Baik", sedangkan FOMO tinggi didominasi kategori "Buruk".
                    </div>
                    """, unsafe_allow_html=True)
        
            else:
                st.info("Data FOMO atau kemampuan keuangan tidak memadai untuk analisis ini.")
        else:
            st.warning("Kolom FOMO atau kemampuan keuangan tidak ditemukan di dataset.")
            
    # =====================================================
    # TAB 5: Proporsi Pengeluaran FOMO dari Uang Saku
    # =====================================================
    with tab5:
        # ---------------------------
        # Section: Proporsi Pengeluaran FOMO terhadap Uang Saku (%)
        # ---------------------------
        st.subheader("Proporsi Pengeluaran FOMO terhadap Uang Saku (%)")
        
        if df_work["proporsi_fomo_pct"].notna().sum() > 0:
            # Layout dua kolom (sampingan, bukan atas–bawah)
            c1, c2 = st.columns(2)
        
            # ==========================================================
            # PIE CHART (Proporsi Kategori)
            # ==========================================================
            with c1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("**Proporsi Pengeluaran FOMO dari Uang Saku**")
            
                bins = [0, 20, 50, 100]
                labels = ["Rendah (<20%)", "Sedang (20–50%)", "Tinggi (>50%)"]
                df_work["kategori_proporsi"] = pd.cut(df_work["proporsi_fomo_pct"], bins=bins, labels=labels, include_lowest=True)
                proporsi_counts = df_work["kategori_proporsi"].value_counts().reindex(labels)
            
                fig_pie = px.pie(
                    values=proporsi_counts.values,
                    names=proporsi_counts.index,
                    color=proporsi_counts.index,
                    color_discrete_sequence=['#FDD6D8', '#F98980', '#B14454']
                )
                fig_pie.update_traces(
                    textinfo="label+percent",
                    textposition="inside",
                    pull=0,  # <-- diset 0 agar menyatu rapat
                    marker=dict(line=dict(color='rgba(0,0,0,0)', width=0))  # <-- hilangkan garis putih
                )
                fig_pie.update_layout(
                    title="Proporsi Pengeluaran FOMO dari Uang Saku",
                    font_family="Times New Roman",
                    title_font_color="#660F2F",
                    legend=dict(title="", orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05),
                    margin=dict(l=10, r=10, t=40, b=10),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
                # Insight card
                st.markdown("""
                <div class="card" style="background:#FDA19B; color:#660F2F;">
                    💡 <b>Insight:</b> Sebagian besar responden berada pada kategori sedang (20–50%), 
                    menandakan proporsi pengeluaran FOMO yang moderat terhadap uang saku.
                </div>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        
            # ==========================================================
            # HISTOGRAM (Distribusi Proporsi)
            # ==========================================================
            with c2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("**Distribusi Proporsi Pengeluaran FOMO dari Uang Saku**")
        
                mean_proporsi = df_work["proporsi_fomo_pct"].mean()
        
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=df_work["proporsi_fomo_pct"],
                    nbinsx=20,
                    marker_color="#E47A7B",
                    opacity=0.8
                ))
                fig_hist.add_vline(
                    x=mean_proporsi,
                    line_dash="dash",
                    line_color="#B14454",
                    annotation_text=f"Rata-rata: {mean_proporsi:.1f}%",
                    annotation_position="top right",
                    annotation_font_size=12,
                    annotation_font_color="#660F2F"
                )
                fig_hist.update_layout(
                    title="Distribusi Proporsi Pengeluaran FOMO dari Uang Saku",
                    xaxis_title="Proporsi Pengeluaran FOMO (%)",
                    yaxis_title="Jumlah Responden",
                    font_family="Times New Roman",
                    title_font_color="#660F2F",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                st.plotly_chart(fig_hist, use_container_width=True)
        
                # Insight card
                st.markdown(f"""
                <div class="card" style="background:#E47A7B; color:#660F2F;">
                    💡 <b>Insight:</b> Sebagian besar responden memiliki proporsi FOMO di bawah 50%, 
                    dengan rata-rata sekitar {mean_proporsi:.1f}%.
                </div>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Data pengeluaran FOMO dan/atau uang saku tidak memadai untuk analisis proporsi.")
        

    # =====================================================
    # TAB 6: Korelasi Numerik
    # =====================================================
    with tab6:
        st.subheader("Korelasi Antar Variabel Numerik (Pearson)")
        
        num_df = df_work.select_dtypes(include=[np.number]).copy()
        num_df = num_df.loc[:, num_df.notna().any()]  # drop all-empty cols
        
        if num_df.shape[1] > 1:
            corr = num_df.corr().round(2)
        
            # Membuat annotated heatmap manual dengan plotly.graph_objects
            fig_corr = go.Figure()
        
            fig_corr.add_trace(go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale=[
                    [0.0, '#FDA19B'],
                    [0.2, '#E47A7B'],
                    [0.4, '#CB5D66'],
                    [0.6, '#B14454'],
                    [0.8, '#982E46'],
                    [1.0, '#660F2F']
                ],
                zmin=-1,
                zmax=1,
                hovertemplate="Variabel X=%{x}<br>Variabel Y=%{y}<br>Korelasi=%{z}<extra></extra>"
            ))
        
            # Tambahkan anotasi nilai korelasi di setiap sel
            annotations = []
            for i, row in enumerate(corr.values):
                for j, val in enumerate(row):
                    annotations.append(
                        dict(
                            x=corr.columns[j],
                            y=corr.index[i],
                            text=str(val),
                            showarrow=False,
                            font=dict(color="white" if abs(val) > 0.5 else "#330A1C", size=12)
                        )
                    )
        
            fig_corr.update_layout(
                title="Heatmap Korelasi (Pearson) dengan Nilai Korelasi",
                font_family="Times New Roman",
                title_font_color='#660F2F',
                height=700,
                annotations=annotations,
                xaxis=dict(side="bottom"),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
        
            st.plotly_chart(fig_corr, use_container_width=True)
        
            # Insight dengan background (seragam seperti bagian lain)
            st.markdown("""
            <div class="card" style="background:#FDA19B; color:#660F2F;">
                💡 <b>Insight:</b> Korelasi positif terlihat kuat pada variabel-variabel keuangan, 
                sedangkan korelasi negatif muncul antara FOMO dengan kesejahteraan. 
                Nilai di atas 0,7 menandakan hubungan yang sangat kuat.
            </div>
            """, unsafe_allow_html=True)
        
        else:
            st.info("Tidak cukup variabel numerik untuk menampilkan korelasi.")
    


# ================================
# Halaman 3: Kesimpulan
# ================================
# ================================
# Halaman 3: Kesimpulan (Nuansa Palet Pink-Maroon)
# ================================
elif page.startswith("Halaman 3"):
    st.markdown(f"""
    <div style="background:#7F1D3A; padding:22px; border-radius:10px; text-align:center;">
      <h1 style="margin:6px; color:white;">Kesimpulan Penelitian</h1>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # ================================
    # CSS untuk Kesimpulan
    # ================================
    st.markdown("""
    <style>
    .kesimpulan-card {
        background: linear-gradient(135deg, #FDA19B, #CB5D66, #982E46);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        color: #330A1C;
        font-family: 'Times New Roman', serif;
        box-shadow: 0px 6px 15px rgba(152, 46, 70, 0.3);
        transition: 0.3s ease-in-out;
    }
    .kesimpulan-card:hover {
        transform: translateY(-3px);
        box-shadow: 0px 10px 25px rgba(101, 15, 47, 0.35);
    }
    .kesimpulan-card h3 {
        margin: 0 0 10px 0;
        color: #660F2F;
        font-size: 20px;
        font-weight: bold;
    }
    .kesimpulan-card p {
        margin: 0;
        font-size: 15px;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)

    # ================================
    # Isi Kesimpulan
    # ================================
    st.markdown("""
    <div class="kesimpulan-card">
        <h3>Partisipasi Responden</h3>
        <p>Mayoritas responden berasal dari <b>Fakultas Ilmu Komputer</b>, menunjukkan partisipasi survei yang paling tinggi.</p>
    </div>

    <div class="kesimpulan-card">
        <h3>Fenomena FOMO</h3>
        <p>Sebanyak <b>63,4%</b> mahasiswa tidak merasa FOMO, sementara <b>36,6%</b> mengaku mengalami FOMO.</p>
    </div>

    <div class="kesimpulan-card">
        <h3>FOMO & Kesejahteraan Psikologis</h3>
        <p>Mahasiswa yang jarang FOMO cenderung memiliki <b>kesejahteraan psikologis yang Baik</b> (79 responden). 
        Namun, pada kategori <b>FOMO sangat sering</b>, jumlah mahasiswa dengan kesejahteraan Baik menurun (40 responden) 
        dan meningkat pada kategori Cukup Baik (13 responden).</p>
    </div>

    <div class="kesimpulan-card">
        <h3>Kemampuan Keuangan & Kesejahteraan</h3>
        <p>Mahasiswa dengan kemampuan mengelola keuangan <b>Baik hingga Sangat Baik</b> lebih dominan berada pada 
        kesejahteraan yang tinggi. Sebaliknya, kemampuan keuangan rendah banyak dikaitkan dengan kesejahteraan Buruk.</p>
    </div>

    <div class="kesimpulan-card">
        <h3>FOMO vs Kemampuan Keuangan</h3>
        <p>Tingkat FOMO yang tinggi berkorelasi dengan <b>penurunan kemampuan keuangan</b>. 
        Mahasiswa dengan FOMO rendah didominasi kategori “Baik–Sangat Baik”, sedangkan FOMO tinggi banyak berada di kategori “Buruk”.</p>
    </div>

    <div class="kesimpulan-card">
        <h3>Proporsi Pengeluaran FOMO</h3>
        <p>Mayoritas responden memiliki proporsi pengeluaran FOMO <b>sedang (20–50%)</b> terhadap uang saku, 
        dengan rata-rata sekitar <b>34,1%</b>.</p>
    </div>

    <div class="kesimpulan-card">
        <h3>Korelasi Antar Variabel</h3>
        <p>Korelasi positif yang kuat ditemukan pada variabel-variabel keuangan, sedangkan korelasi negatif terlihat 
        antara <b>tingkat FOMO dengan kesejahteraan psikologis</b>, menandakan semakin tinggi FOMO maka kesejahteraan cenderung menurun.</p>
    </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown(f"""
    <div style="background:#660F2F; color:white; text-align:center; padding:12px; border-radius:10px; margin-top:30px;">
        Disusun oleh <b>Kelompok Threeasure</b> • UPN "Veteran" Jawa Timur (2025)
    </div>
    """, unsafe_allow_html=True)


