# app.py
# HuggingFace Spaces / Gradio
# STL + RBEAST berjalan bersamaan
# Dropdown value = pos_id
# Label frontend = nama_pos

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import gradio as gr

from supabase import create_client
from statsmodels.tsa.seasonal import STL
from Rbeast import beast

# =====================================================
# SUPABASE
# =====================================================
SUPABASE_URL = "ISI_URL_SUPABASE"
SUPABASE_KEY = "ISI_ANON_KEY"

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

# =====================================================
# TEMA PLOT
# =====================================================
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

# =====================================================
# AMBIL DATA POS
# tampil: nama_pos
# value : pos_id
# =====================================================
def get_pos():

    q = supabase.table("data_ch")\
        .select("pos_id,nama_pos")\
        .execute()

    df = pd.DataFrame(q.data)

    if df.empty:
        return []

    df = df.drop_duplicates(subset=["pos_id"])
    df = df.sort_values("nama_pos")

    pilihan = []

    for _, r in df.iterrows():
        pilihan.append(
            (
                str(r["nama_pos"]),
                str(r["pos_id"])
            )
        )

    return pilihan

# =====================================================
# AMBIL DATA HUJAN
# tanggal = yyyy-mm-dd
# =====================================================
def get_data(pos_id):

    q = supabase.table("data_ch")\
        .select("tanggal,rain")\
        .eq("pos_id", pos_id)\
        .order("tanggal")\
        .execute()

    df = pd.DataFrame(q.data)

    if df.empty:
        return df

    df["tanggal"] = pd.to_datetime(df["tanggal"])
    df["rain"] = pd.to_numeric(
        df["rain"],
        errors="coerce"
    ).round(0)

    return df

# =====================================================
# AGREGASI
# =====================================================
def agregasi(df, periode, metode):

    d = df.copy()
    d = d.set_index("tanggal")

    if periode == "Harian":
        hasil = d.resample("D")

    elif periode == "Bulanan":
        hasil = d.resample("MS")

    else:
        hasil = d.resample("YS")

    if metode == "Rerata":
        out = hasil.mean()

    elif metode == "Minimum":
        out = hasil.min()

    elif metode == "Maksimum":
        out = hasil.max()

    else:
        out = hasil.sum()

    out = out.dropna()
    out["rain"] = out["rain"].round(0)

    return out

# =====================================================
# STL
# =====================================================
def plot_stl(df):

    y = df["rain"]

    if len(y) < 24:
        raise ValueError("Data terlalu sedikit.")

    stl = STL(
        y,
        period=12,
        robust=True
    )

    r = stl.fit()

    fig, ax = plt.subplots(
        3, 1,
        figsize=(12,8),
        sharex=True
    )

    ax[0].plot(df.index, r.trend, color="green")
    ax[0].set_ylabel("Tren")

    ax[1].plot(df.index, r.seasonal, color="red")
    ax[1].set_ylabel("Musiman")

    ax[2].plot(df.index, r.resid, color="gray")
    ax[2].set_ylabel("Residu")
    ax[2].set_xlabel("Tahun")

    for a in ax:
        a.grid(alpha=0.25)

    plt.tight_layout()
    return fig

# =====================================================
# RBEAST
# =====================================================
def plot_beast(df):

    y = df["rain"].values.astype(float)

    tahun_awal = df.index[0].year
    bulan_awal = df.index[0].month

    start_year = tahun_awal + (bulan_awal - 1)/12

    hasil = beast(
        y,
        start=start_year,
        deltat=1/12,
        freq=12,
        season="harmonic"
    )

    trend = hasil.trend.Y
    seasonal = hasil.season.Y
    resid = y - trend - seasonal

    fig, ax = plt.subplots(
        3, 1,
        figsize=(12,8),
        sharex=True
    )

    ax[0].plot(df.index, trend, color="green")
    ax[0].set_ylabel("Tren")

    ax[1].plot(df.index, seasonal, color="red")
    ax[1].set_ylabel("Musiman")

    ax[2].plot(df.index, resid, color="gray")
    ax[2].set_ylabel("Residu")
    ax[2].set_xlabel("Tahun")

    for a in ax:
        a.grid(alpha=0.25)

    plt.tight_layout()
    return fig

# =====================================================
# PROSES UTAMA
# langsung STL + RBEAST
# =====================================================
def proses(pos_id, periode, metode):

    df = get_data(pos_id)

    if df.empty:
        raise gr.Error("Data kosong.")

    data = agregasi(df, periode, metode)

    fig1 = plot_stl(data)
    fig2 = plot_beast(data)

    info = f"""
Jumlah data : {len(data):,}
Awal data : {data.index.min().date()}
Akhir data : {data.index.max().date()}
Periode : {periode}
Metode : {metode}
    """

    return fig1, fig2, info

# =====================================================
# UI
# =====================================================
tema = gr.themes.Soft()

with gr.Blocks(title="STL & RBEAST Curah Hujan") as demo:

    gr.Markdown("""
# 🌧️ Dekomposisi Curah Hujan

Metode berjalan bersamaan:

- STL
- RBEAST
""")

    with gr.Row():

        pos = gr.Dropdown(
            choices=get_pos(),
            label="Pos Hujan",
            value=None
        )

        periode = gr.Dropdown(
            choices=[
                "Harian",
                "Bulanan",
                "Tahunan"
            ],
            value="Bulanan",
            label="Periode"
        )

        metode = gr.Dropdown(
            choices=[
                "Kumulatif",
                "Rerata",
                "Minimum",
                "Maksimum"
            ],
            value="Kumulatif",
            label="Metode"
        )

    btn = gr.Button(
        "Proses",
        variant="primary"
    )

    info = gr.Textbox(
        label="Ringkasan"
    )

    with gr.Row():
        out1 = gr.Plot(label="STL")
        out2 = gr.Plot(label="RBEAST")

    btn.click(
        fn=proses,
        inputs=[pos, periode, metode],
        outputs=[out1, out2, info]
    )

demo.launch()