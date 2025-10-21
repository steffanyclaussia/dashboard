# threeasure_dashboard.py
# Dashboard FOMO, Keuangan, & Kesejahteraan Psikologis
# Author: Nona Steffany (UPN Veteran Jawa Timur)
# App by: ChatGPT (GPT-5 Thinking)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import re
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Threeasure Dashboard — FOMO • Keuangan • Kesejahteraan",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================
# Helpers
# ========================

def currency_to_float(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float)): return float(x)
    s = str(x)
    s = s.replace("Rp", "").replace("rp", "").replace(".", "").replace(",", "")
    s = re.sub(r"[^0-9\-]", "", s)
    try: return float(s)
    except: return np.nan

def yesno_to_binary(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    if s in ["ya","yes","y","1","true"]: return 1.0
    if s in ["tidak","no","n","0","false"]: return 0.0
    try: return float(s)
    except: return np.nan

FOMO_MAP = {1:"Tidak Pernah",2:"Jarang",3:"Kadang-kadang",4:"Sering",5:"Sangat Sering"}
KEUANGAN_MAP = {1:"Buruk",2:"Kurang Baik",3:"Cukup Baik",4:"Baik",5:"Sangat Baik"}

def score_kesejahteraan(df):
    cols = [
        "pengaruh_fomo_terhadap_emosi",
        "frekuensi_stres_karena_finansial",
        "frekuensi_hilang_semangat_kuliah_karena_tekanan_finansial",
        "frekuensi_stres_fomo",
        "kebutuhan_akan_dukungan_emosional_dan_bantuan_psikologis_bin"
    ]
    sc = df[cols].sum(axis=1, min_count=1)
    conds = np.select(
        [sc <= 10, (sc >= 11) & (sc <= 16), sc >= 17],
        ["Baik", "Cukup Baik", "Buruk"],
        default=np.nan
    )
    return sc, conds

def stacked_bar_by(df, by_col, hue_col, title):
    agg = df.groupby([by_col, hue_col]).size().reset_index(name="count")
    total = agg.groupby(by_col)["count"].transform("sum")
    agg["pct"] = (agg["count"]/total*100).round(1)
    fig = px.bar(agg, x=by_col, y="count", color=hue_col, text="pct", title=title, barmode="stack")
    fig.update_traces(texttemplate="%{text}%")
    fig.update_layout(legend_title="", xaxis_title="", yaxis_title="Jumlah Responden")
    return fig

def heatmap_count(df, row, col, title):
    ct = pd.crosstab(df[row], df[col])
    fig = px.imshow(ct, text_auto=True, aspect="auto", title=title)
    fig.update_layout(xaxis_title=col, yaxis_title=row)
    return fig

def corr_heatmap(df_num, title):
    corr = df_num.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title=title)
    return fig

def fit_ols(y, X, add_const=True):
    if add_const: X = sm.add_constant(X, has_constant="add")
    return sm.OLS(y, X, missing="drop").fit()

# ========================
# Load Data (Fix CachedWidgetWarning)
# ========================

@st.cache_data(show_spinner=False)
def load_csv(path_or_buffer):
    return pd.read_csv(path_or_buffer)

def load_data():
    try:
        df = load_csv("Data Threeasure_Cleaned.csv")
    except Exception:
        st.info("Unggah file **Data Threeasure_Cleaned.csv** bila file tidak ada di folder yang sama.")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is None: return None
        df = load_csv(uploaded)

    # --- preprocessing ---
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]
    rename_map = {
        "rata-rata_uang_saku_perbulan": "uang_saku_bulanan",
        "pengeluaran_untuk_fomo_per_bulan": "pengeluaran_fomo_bulanan",
        "sering_merasa_fomo": "sering_merasa_fomo",
        "frekuensi_fomo_pengeluaran": "frekuensi_fomo_pengeluaran",
        "kebutuhan_akan_dukungan_emosional_dan_bantuan_psikologis": "butuh_dukungan_emosional"
    }
    for k,v in rename_map.items():
        if k in df.columns: df.rename(columns={k:v}, inplace=True)

    if "uang_saku_bulanan" in df.columns:
        df["uang_saku_bulanan_num"] = df["uang_saku_bulanan"].apply(currency_to_float)
    if "pengeluaran_fomo_bulanan" in df.columns:
        df["pengeluaran_fomo_bulanan_num"] = df["pengeluaran_fomo_bulanan"].apply(currency_to_float)
    if "butuh_dukungan_emosional" in df.columns:
        df["kebutuhan_akan_dukungan_emosional_dan_bantuan_psikologis_bin"] = df["butuh_dukungan_emosional"].apply(yesno_to_binary)

    if "frekuensi_fomo_pengeluaran" in df.columns:
        df["fomo_level"] = df["frekuensi_fomo_pengeluaran"].map(FOMO_MAP)
    if "kemampuan_mengelola_keuangan" in df.columns:
        df["kemampuan_keuangan_level"] = df["kemampuan_mengelola_keuangan"].map(KEUANGAN_MAP)

    if "pengeluaran_fomo_bulanan_num" in df.columns and "uang_saku_bulanan_num" in df.columns:
        df["proporsi_fomo"] = df["pengeluaran_fomo_bulanan_num"]/df["uang_saku_bulanan_num"]
        df["kategori_proporsi_fomo"] = pd.cut(df["proporsi_fomo"], bins=[-np.inf,0.2,0.5,np.inf],
                                              labels=["Rendah (<20%)","Sedang (20–50%)","Tinggi (>50%)"])

    df["skor_psikologis"], df["kategori_kesejahteraan"] = score_kesejahteraan(df)

    if "fakultas" in df.columns:
        df["fakultas"] = df["fakultas"].astype(str).str.strip()

    return df

# ========================
# UI CSS
# ========================

st.markdown("""
<style>
.kpi{background:#fff;border:1px solid #f0f2f5;border-radius:18px;padding:18px 20px;
     box-shadow:0 6px 20px rgba(0,0,0,0.06);transition:transform .2s ease,box-shadow .2s ease;}
.kpi:hover{transform:translateY(-2px);box-shadow:0 10px 28px rgba(0,0,0,0.08);}
.kpi .small{font-size:12px;color:#6b7280;margin-bottom:6px;}
.kpi h3{margin:0;font-size:24px;}
.section h2{display:inline-block;border-bottom:3px solid #2563eb;padding-bottom:6px;margin-bottom:0;}
.fade-in{animation:fade .5s ease-in-out both;}
@keyframes fade{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:translateY(0)}}
section[data-testid="stSidebar"]{width:350px;}
</style>
""", unsafe_allow_html=True)

# ========================
# Sidebar
# ========================

df = load_data()
with st.sidebar:
    st.title("💠 Threeasure Dashboard")
    st.caption("FOMO • Pengelolaan Keuangan • Kesejahteraan Psikologis")
    fakultas_list = ["Semua"] + sorted(df["fakultas"].dropna().unique().tolist()) if df is not None and "fakultas" in df.columns else ["Semua"]
    sel_fak = st.selectbox("Filter Fakultas", fakultas_list, index=0)
    menu = st.radio("Pilih Halaman", ["🏠 Overview","🧠 FOMO → Kesejahteraan","💼 Keuangan → Kesejahteraan",
                                      "🔗 FOMO × Keuangan","🧮 Proporsi Pengeluaran FOMO","📈 Korelasi & Regresi"], index=0)

if df is None: st.stop()
df_view = df.copy()
if sel_fak!="Semua" and "fakultas" in df_view.columns:
    df_view = df_view[df_view["fakultas"]==sel_fak]

# ========================
# Pages
# ========================
if menu=="🏠 Overview":
    st.subheader("Ikhtisar Responden")
    total_n = len(df_view)
    n_fak = df_view["fakultas"].nunique() if "fakultas" in df_view.columns else "-"
    rata_uang = df_view["uang_saku_bulanan_num"].mean()
    rata_peng_fomo = df_view["pengeluaran_fomo_bulanan_num"].mean()

    k1,k2,k3,k4=st.columns(4)
    with k1: st.markdown(f"<div class='kpi'><div class='small'>Jumlah Responden</div><h3>{total_n}</h3></div>",unsafe_allow_html=True)
    with k2: st.markdown(f"<div class='kpi'><div class='small'>Jumlah Fakultas</div><h3>{n_fak}</h3></div>",unsafe_allow_html=True)
    with k3: st.markdown(f"<div class='kpi'><div class='small'>Rata-rata Uang Saku</div><h3>Rp {0 if pd.isna(rata_uang) else int(rata_uang):,}</h3></div>",unsafe_allow_html=True)
    with k4: st.markdown(f"<div class='kpi'><div class='small'>Rata-rata Pengeluaran FOMO</div><h3>Rp {0 if pd.isna(rata_peng_fomo) else int(rata_peng_fomo):,}</h3></div>",unsafe_allow_html=True)

    st.divider()
    c1,c2=st.columns([1.3,1])
    with c1:
        if "fakultas" in df_view.columns:
            fig=px.bar(df_view["fakultas"].value_counts().reset_index(),x="index",y="fakultas",title="Distribusi Responden per Fakultas")
            st.plotly_chart(fig,use_container_width=True)
    with c2:
        if "sering_merasa_fomo" in df_view.columns:
            pie_df=df_view["sering_merasa_fomo"].astype(str).str.title().value_counts().reset_index()
            pie_df.columns=["Kategori","Jumlah"]
            figp=px.pie(pie_df,names="Kategori",values="Jumlah",title="Proporsi: Sering Merasa FOMO")
            st.plotly_chart(figp,use_container_width=True)

elif menu=="🧠 FOMO → Kesejahteraan":
    st.subheader("Pengaruh FOMO terhadap Kesejahteraan")
    c1,c2=st.columns([1.2,1])
    with c1: st.plotly_chart(heatmap_count(df_view,"fomo_level","kategori_kesejahteraan","Heatmap: FOMO vs Kesejahteraan"),use_container_width=True)
    with c2: st.plotly_chart(stacked_bar_by(df_view,"fomo_level","kategori_kesejahteraan","Distribusi Kesejahteraan per Tingkat FOMO"),use_container_width=True)

elif menu=="💼 Keuangan → Kesejahteraan":
    st.subheader("Pengelolaan Keuangan & Kesejahteraan")
    c1,c2=st.columns([1.2,1])
    with c1: st.plotly_chart(heatmap_count(df_view,"kemampuan_keuangan_level","kategori_kesejahteraan","Heatmap: Keuangan vs Kesejahteraan"),use_container_width=True)
    with c2: st.plotly_chart(stacked_bar_by(df_view,"kemampuan_keuangan_level","kategori_kesejahteraan","Distribusi Kesejahteraan per Kemampuan Keuangan"),use_container_width=True)

elif menu=="🔗 FOMO × Keuangan":
    st.subheader("Hubungan FOMO & Kemampuan Keuangan")
    st.plotly_chart(stacked_bar_by(df_view,"fomo_level","kemampuan_keuangan_level","Distribusi Kemampuan Keuangan per Tingkat FOMO"),use_container_width=True)

elif menu=="🧮 Proporsi Pengeluaran FOMO":
    st.subheader("Proporsi Pengeluaran FOMO dari Uang Saku")
    c1,c2=st.columns([1,1])
    with c1:
        pie_df=df_view["kategori_proporsi_fomo"].value_counts(dropna=True).reset_index()
        pie_df.columns=["Kategori","Jumlah"]
        figp=px.pie(pie_df,names="Kategori",values="Jumlah",title="Kategori Proporsi Pengeluaran FOMO")
        st.plotly_chart(figp,use_container_width=True)
    with c2:
        if df_view["proporsi_fomo"].notna().any():
            figh=px.histogram(df_view,x="proporsi_fomo",nbins=25,title="Distribusi Proporsi Pengeluaran FOMO")
            mean_val=df_view["proporsi_fomo"].mean()
            figh.add_vline(x=mean_val,line_dash="dash")
            figh.add_annotation(x=mean_val,y=0,yref="paper",showarrow=False,text=f"Rata-rata: {mean_val:.2f}")
            st.plotly_chart(figh,use_container_width=True)

elif menu=="📈 Korelasi & Regresi":
    st.subheader("Korelasi, Regresi, & Moderasi")
    num_cols=["uang_saku_bulanan_num","pengeluaran_fomo_bulanan_num","kemampuan_mengelola_keuangan",
              "frekuensi_fomo_pengeluaran","pengaruh_fomo_terhadap_emosi","frekuensi_stres_karena_finansial",
              "frekuensi_hilang_semangat_kuliah_karena_tekanan_finansial","frekuensi_stres_fomo",
              "kebutuhan_akan_dukungan_emosional_dan_bantuan_psikologis_bin","skor_psikologis"]
    num_cols=[c for c in num_cols if c in df_view.columns]
    df_num=df_view[num_cols].apply(pd.to_numeric,errors="coerce")
    if not df_num.empty: st.plotly_chart(corr_heatmap(df_num,"Heatmap Korelasi Antar Variabel"),use_container_width=True)

    if {"frekuensi_fomo_pengeluaran","skor_psikologis"}.issubset(df_view.columns):
        model=fit_ols(df_view["skor_psikologis"],df_view[["frekuensi_fomo_pengeluaran"]])
        st.subheader("Regresi Sederhana: FOMO → Skor Kesejahteraan")
        st.text(model.summary().as_text())

    if {"frekuensi_fomo_pengeluaran","kemampuan_mengelola_keuangan","skor_psikologis"}.issubset(df_view.columns):
        df_mod=df_view[["frekuensi_fomo_pengeluaran","kemampuan_mengelola_keuangan","skor_psikologis"]].copy()
        df_mod["interaksi"]=df_mod["frekuensi_fomo_pengeluaran"]*df_mod["kemampuan_mengelola_keuangan"]
        model_m=fit_ols(df_mod["skor_psikologis"],df_mod[["frekuensi_fomo_pengeluaran","kemampuan_mengelola_keuangan","interaksi"]])
        st.subheader("Analisis Moderasi: FOMO × Kemampuan Keuangan → Skor Kesejahteraan")
        st.text(model_m.summary().as_text())

        low=df_view["kemampuan_mengelola_keuangan"].quantile(0.25)
        high=df_view["kemampuan_mengelola_keuangan"].quantile(0.75)
        grid=pd.DataFrame({"frekuensi_fomo_pengeluaran":np.linspace(df_view["frekuensi_fomo_pengeluaran"].min(),df_view["frekuensi_fomo_pengeluaran"].max(),20)})
        grid_low=grid.copy(); grid_low["kemampuan_mengelola_keuangan"]=low; grid_low["interaksi"]=grid_low["frekuensi_fomo_pengeluaran"]*low
        pred_low=model_m.predict(sm.add_constant(grid_low))
        grid_high=grid.copy(); grid_high["kemampuan_mengelola_keuangan"]=high; grid_high["interaksi"]=grid_high["frekuensi_fomo_pengeluaran"]*high
        pred_high=model_m.predict(sm.add_constant(grid_high))

        fig_mod=go.Figure()
        fig_mod.add_trace(go.Scatter(x=grid_low["frekuensi_fomo_pengeluaran"],y=pred_low,mode="lines",name=f"Keuangan Rendah (Q1={low:.1f})"))
        fig_mod.add_trace(go.Scatter(x=grid_high["frekuensi_fomo_pengeluaran"],y=pred_high,mode="lines",name=f"Keuangan Tinggi (Q3={high:.1f})"))
        fig_mod.update_layout(title="Kurva Moderasi (Prediksi Skor Kesejahteraan)",xaxis_title="Frekuensi FOMO Pengeluaran",yaxis_title="Skor Kesejahteraan (Prediksi)")
        st.plotly_chart(fig_mod,use_container_width=True)

# ========================
# Footer
# ========================
st.caption("© 202
