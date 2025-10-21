# dashboard_threeasure_full.py
# Dashboard Threeasure (3 Halaman)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import os

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
# Load data (diperbaiki sesuai file yang Nona unggah)
# ---------------------------
@st.cache_data
def load_df(path="Data Threeasure_Cleaned.csv"):
    return pd.read_csv(path)

if os.path.exists("Data Threeasure_Cleaned.csv"):
    df = load_df("Data Threeasure_Cleaned.csv")
else:
    st.warning("File 'Data Threeasure_Cleaned.csv' tidak ditemukan. Silakan upload file CSV.")
    uploaded = st.file_uploader("Upload file CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
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
