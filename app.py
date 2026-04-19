# app.py
# Hugging Face Spaces / Gradio
# Rbeast Curah Hujan - Python version dari kode R Anda

import io
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import gradio as gr
from Rbeast import beast

# ===============================
# TEMA MATPLOTLIB
# ===============================
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "white"
})

# ===============================
# FUNGSI PROSES
# ===============================
def proses_beast(file):

    if file is None:
        raise gr.Error("Silakan unggah file CSV terlebih dahulu.")

    # =========================
    # BACA CSV
    # =========================
    df = pd.read_csv(
        file.name,
        sep=";",
        encoding="utf-8-sig"
    )

    df.columns = ["Tanggal", "Data"]

    # =========================
    # KONVERSI
    # =========================
    df["Tanggal"] = pd.to_datetime(
        df["Tanggal"],
        format="%d/%m/%y",
        errors="coerce"
    )

    df["Data"] = pd.to_numeric(
        df["Data"],
        errors="coerce"
    )

    df = df.dropna(subset=["Tanggal"])

    # =========================
    # BULANAN
    # =========================
    df["year"] = df["Tanggal"].dt.year
    df["month"] = df["Tanggal"].dt.month

    df_monthly = (
        df.groupby(["year", "month"])["Data"]
        .sum()
        .reset_index()
    )

    df_monthly.columns = ["year", "month", "rain"]

    # =========================
    # DATA TIME SERIES
    # =========================
    y = df_monthly["rain"].values.astype(float)

    start_year = (
        df_monthly.loc[0, "year"] +
        (df_monthly.loc[0, "month"] - 1) / 12
    )

    # =========================
    # BEAST
    # =========================
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

    # =========================
    # INDEX TANGGAL
    # =========================
    dates = pd.date_range(
        start=f"{df_monthly.loc[0,'year']}-{df_monthly.loc[0,'month']:02d}-01",
        periods=len(y),
        freq="MS"
    )

    # =========================
    # PLOT
    # =========================
    fig, axes = plt.subplots(
        3, 1,
        figsize=(12, 8),
        sharex=True
    )

    locator = mdates.YearLocator(10)
    formatter = mdates.DateFormatter("%Y")

    # Tren
    axes[0].plot(
        dates, trend,
        color="green",
        linewidth=2
    )
    axes[0].set_ylabel("Tren (mm)")
    axes[0].grid(alpha=0.25)

    # Musiman
    axes[1].plot(
        dates, seasonal,
        color="red",
        linewidth=1.5
    )
    axes[1].set_ylabel("Musiman (mm)")
    axes[1].grid(alpha=0.25)

    # Residu
    axes[2].plot(
        dates, resid,
        color="darkgray",
        linewidth=1.2
    )
    axes[2].set_ylabel("Residu (mm)")
    axes[2].set_xlabel("Tahun")
    axes[2].grid(alpha=0.25)

    for ax in axes:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    plt.tight_layout()

    # =========================
    # RINGKASAN
    # =========================
    teks = f"""
Jumlah data harian : {len(df):,}
Jumlah data bulanan : {len(df_monthly):,}
Periode awal : {dates.min().strftime('%Y-%m')}
Periode akhir : {dates.max().strftime('%Y-%m')}
Metode : BEAST Harmonic Seasonal
    """

    return fig, teks


# ===============================
# TEMA HF / GRADIO
# ===============================
tema = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="gray"
)

css = """
body{
    background:#f4f6f9;
}
.gradio-container{
    max-width:1100px !important;
}
h1{
    text-align:center;
}
"""

# ===============================
# UI
# ===============================
with gr.Blocks(theme=tema, css=css, title="Rbeast Curah Hujan") as demo:

    gr.Markdown("""
# 🌧️ Dekomposisi Curah Hujan dengan Rbeast

Unggah file CSV berformat:

`Tanggal;Data`

Contoh:

`01/01/80;33`
""")

    with gr.Row():
        file_input = gr.File(
            label="Unggah CSV",
            file_types=[".csv"]
        )

    tombol = gr.Button(
        "Proses Data",
        variant="primary"
    )

    hasil_plot = gr.Plot(
        label="Hasil Dekomposisi"
    )

    hasil_text = gr.Textbox(
        label="Ringkasan",
        lines=8
    )

    tombol.click(
        fn=proses_beast,
        inputs=file_input,
        outputs=[hasil_plot, hasil_text]
    )

# ===============================
# RUN
# ===============================
demo.launch(server_name="0.0.0.0", server_port=7860)