import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import gradio as gr

from supabase import create_client
from statsmodels.tsa.seasonal import STL
from Rbeast import beast

# =====================================
# SUPABASE
# =====================================
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase = create_client(url, key)

# =====================================
# TEMA
# =====================================
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14
})

# =====================================
# AMBIL POS
# =====================================
def get_pos():

    q = supabase.table("data_ch")\
        .select("pos_id")\
        .execute()

    data = pd.DataFrame(q.data)

    return sorted(data["pos_id"].dropna().unique().tolist())

# =====================================
# QUERY DATA
# =====================================
def load_data(pos_id):

    q = supabase.table("data_ch")\
        .select("tanggal,rain")\
        .eq("pos_id", pos_id)\
        .order("tanggal")\
        .execute()

    df = pd.DataFrame(q.data)

    df["tanggal"] = pd.to_datetime(df["tanggal"])
    df["rain"] = pd.to_numeric(df["rain"], errors="coerce").round(0)

    return df

# =====================================
# AGREGASI
# =====================================
def agregasi(df, periode, metode):

    df = df.copy()
    df = df.set_index("tanggal")

    if periode == "Harian":
        out = df.rename(columns={"rain":"nilai"})
        return out.reset_index()

    kode = {
        "Bulanan":"MS",
        "Tahunan":"YS"
    }[periode]

    if metode == "Rerata":
        out = df.resample(kode).mean()

    elif metode == "Kumulatif":
        out = df.resample(kode).sum()

    elif metode == "Minimum":
        out = df.resample(kode).min()

    elif metode == "Maksimum":
        out = df.resample(kode).max()

    out.columns = ["nilai"]

    return out.reset_index()

# =====================================
# STL
# =====================================
def plot_stl(df):

    ts = df["nilai"].values

    if len(ts) < 24:
        raise Exception("Data terlalu sedikit untuk STL")

    result = STL(ts, period=12, robust=True).fit()

    fig, ax = plt.subplots(3,1, figsize=(11,8), sharex=True)

    x = df["tanggal"]

    ax[0].plot(x, result.trend, color="green")
    ax[0].set_ylabel("Tren")

    ax[1].plot(x, result.seasonal, color="red")
    ax[1].set_ylabel("Musiman")

    ax[2].plot(x, result.resid, color="gray")
    ax[2].set_ylabel("Residu")

    plt.tight_layout()

    return fig

# =====================================
# BEAST
# =====================================
def plot_beast(df):

    y = df["nilai"].values.astype(float)

    start = df["tanggal"].dt.year.iloc[0]

    hasil = beast(
        y,
        start=start,
        deltat=1/12,
        freq=12,
        season="harmonic"
    )

    trend = hasil.trend.Y
    seasonal = hasil.season.Y
    resid = y - trend - seasonal

    fig, ax = plt.subplots(3,1, figsize=(11,8), sharex=True)

    x = df["tanggal"]

    ax[0].plot(x, trend, color="green")
    ax[0].set_ylabel("Tren")

    ax[1].plot(x, seasonal, color="red")
    ax[1].set_ylabel("Musiman")

    ax[2].plot(x, resid, color="gray")
    ax[2].set_ylabel("Residu")

    plt.tight_layout()

    return fig

# =====================================
# PROSES
# =====================================
def proses(pos_id, periode, metode, model):

    t0 = time.time()

    df = load_data(pos_id)
    df = agregasi(df, periode, metode)

    if model == "STL":
        fig = plot_stl(df)
    else:
        fig = plot_beast(df)

    durasi = round(time.time() - t0, 2)

    info = f"""
Jumlah Data : {len(df)}
Periode : {periode}
Metode : {metode}
Model : {model}
Waktu Proses : {durasi} detik
"""

    return fig, info, df.head(20)

# =====================================
# UI
# =====================================
tema = gr.themes.Soft()

with gr.Blocks(theme=tema, title="Curah Hujan") as demo:

    gr.Markdown("""
# 🌧️ Analisis Curah Hujan

Sumber data: Supabase  
Metode: STL dan BEAST
""")

    with gr.Row():

        pos = gr.Dropdown(
            choices=get_pos(),
            label="Pos ID"
        )

        periode = gr.Dropdown(
            choices=["Harian","Bulanan","Tahunan"],
            value="Bulanan",
            label="Periode"
        )

        metode = gr.Dropdown(
            choices=["Rerata","Kumulatif","Minimum","Maksimum"],
            value="Kumulatif",
            label="Metode"
        )

        model = gr.Radio(
            choices=["STL","BEAST"],
            value="STL",
            label="Model"
        )

    btn = gr.Button("Proses", variant="primary")

    plot = gr.Plot()

    info = gr.Textbox(label="Ringkasan")

    tabel = gr.Dataframe()

    btn.click(
        fn=proses,
        inputs=[pos, periode, metode, model],
        outputs=[plot, info, tabel],
        api_name="proses"
    )

demo.launch(server_name="0.0.0.0", server_port=7860)