# app.py
# HuggingFace Spaces - Gradio
# Curah Hujan Dashboard
# PostgreSQL Supabase + STL + RBEAST

import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import gradio as gr

from statsmodels.tsa.seasonal import STL
from Rbeast import beast

# =====================================================
# DATABASE
# =====================================================
DATABASE_URL = os.getenv("DATABASE_URL")

conn = psycopg2.connect(
    DATABASE_URL,
    sslmode="require"
)

# =====================================================
# STYLE PLOT
# =====================================================
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white"
})

# =====================================================
# LIST POS
# =====================================================
def get_pos():

    sql = """
    SELECT DISTINCT pos_id, nama_pos
    FROM data_ch
    ORDER BY nama_pos
    """

    df = pd.read_sql(sql, conn)

    pilihan = []

    for _, r in df.iterrows():
        pilihan.append(
            (r["nama_pos"], str(r["pos_id"]))
        )

    return pilihan


# =====================================================
# QUERY DATA
# =====================================================
def ambil_data(pos_id):

    sql = """
    SELECT tanggal, rain
    FROM data_ch
    WHERE pos_id = %s
    ORDER BY tanggal
    """

    df = pd.read_sql(
        sql,
        conn,
        params=[pos_id]
    )

    df["tanggal"] = pd.to_datetime(df["tanggal"])
    df["rain"] = pd.to_numeric(
        df["rain"],
        errors="coerce"
    )

    df = df.dropna()

    return df


# =====================================================
# AGREGASI
# =====================================================
def agregasi(df, periode, metode):

    d = df.copy()
    d = d.set_index("tanggal")

    if periode == "Harian":
        grp = d.resample("D")

    elif periode == "Bulanan":
        grp = d.resample("MS")

    else:
        grp = d.resample("YS")

    if metode == "Kumulatif":
        out = grp.sum()

    elif metode == "Rerata":
        out = grp.mean()

    elif metode == "Minimum":
        out = grp.min()

    else:
        out = grp.max()

    out = out.dropna()
    out["rain"] = out["rain"].round(0)

    return out


# =====================================================
# STL
# =====================================================
def buat_stl(data):

    y = data["rain"]

    if len(y) < 24:
        raise Exception("Data terlalu sedikit untuk STL.")

    if len(data) > 24:
        period = 12
    else:
        period = max(2, len(data)//2)

    model = STL(
        y,
        period=period,
        robust=True
    )

    hasil = model.fit()

    fig, ax = plt.subplots(
        3, 1,
        figsize=(12, 8),
        sharex=True
    )

    ax[0].plot(data.index, hasil.trend, color="green")
    ax[0].set_ylabel("Tren")

    ax[1].plot(data.index, hasil.seasonal, color="red")
    ax[1].set_ylabel("Musiman")

    ax[2].plot(data.index, hasil.resid, color="gray")
    ax[2].set_ylabel("Residu")
    ax[2].set_xlabel("Tahun")

    for a in ax:
        a.grid(alpha=0.25)

    plt.tight_layout()

    return fig


# =====================================================
# RBEAST
# =====================================================
def buat_beast(data):

    y = data["rain"].values.astype(float)

    tahun_awal = data.index[0].year
    bulan_awal = data.index[0].month

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
        figsize=(12, 8),
        sharex=True
    )

    ax[0].plot(data.index, trend, color="green")
    ax[0].set_ylabel("Tren")

    ax[1].plot(data.index, seasonal, color="red")
    ax[1].set_ylabel("Musiman")

    ax[2].plot(data.index, resid, color="gray")
    ax[2].set_ylabel("Residu")
    ax[2].set_xlabel("Tahun")

    for a in ax:
        a.grid(alpha=0.25)

    plt.tight_layout()

    return fig


# =====================================================
# EXPORT CSV
# =====================================================
def simpan_csv(data):

    path = "/tmp/hasil.csv"
    data.to_csv(path)

    return path


# =====================================================
# PROSES
# =====================================================
def proses(pos_id, periode, metode):

    t0 = time.time()

    df = ambil_data(pos_id)

    if df.empty:
        raise gr.Error("Data kosong.")

    data = agregasi(df, periode, metode)

    fig_stl = buat_stl(data)
    fig_beast = buat_beast(data)

    csv_file = simpan_csv(data)

    dt = round(time.time() - t0, 2)

    info = f"""
Jumlah data : {len(data):,}
Awal data : {data.index.min().date()}
Akhir data : {data.index.max().date()}
Periode : {periode}
Metode : {metode}
Waktu proses : {dt} detik
    """

    return fig_stl, fig_beast, info, csv_file


# =====================================================
# UI
# =====================================================
tema = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate"
)

css = """
.gradio-container{
    max-width:1200px !important;
}
"""

with gr.Blocks(
    title="Curah Hujan Dashboard",
    theme=tema,
    css=css
) as demo:

    gr.Markdown("""
# 🌧️ Dashboard Dekomposisi Curah Hujan

Metode berjalan bersamaan:

- STL
- RBEAST
""")

    with gr.Row():

        pos = gr.Dropdown(
            choices=get_pos(),
            label="Pos Hujan"
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

    tombol = gr.Button(
        "Proses Data",
        variant="primary"
    )

    info = gr.Textbox(
        label="Ringkasan",
        lines=8
    )

    with gr.Row():
        plot1 = gr.Plot(label="STL")
        plot2 = gr.Plot(label="RBEAST")

    file_out = gr.File(
        label="Unduh Data Hasil"
    )

    tombol.click(
        fn=proses,
        inputs=[pos, periode, metode],
        outputs=[
            plot1,
            plot2,
            info,
            file_out
        ]
    )

demo.launch(server_name="0.0.0.0", server_port=7860)