# app.py
# HuggingFace Spaces - Gradio
# STL + RBEAST + Download PNG

import os
import io
import zipfile
import tempfile
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
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

conn = psycopg2.connect(
    DATABASE_URL,
    sslmode="require"
)

# =====================================================
# STYLE
# =====================================================
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
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
# DATA
# =====================================================
def ambil_data(pos_id, th1, th2):

    sql = """
    SELECT tanggal, rain
    FROM data_ch
    WHERE pos_id = %s
    AND EXTRACT(YEAR FROM tanggal) BETWEEN %s AND %s
    ORDER BY tanggal
    """

    df = pd.read_sql(
        sql,
        conn,
        params=[pos_id, th1, th2]
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
        out = d.resample("D").sum()

    elif periode == "Bulanan":

        grp = d.resample("MS")

        if metode == "Kumulatif":
            out = grp.sum()

        elif metode == "Rerata":
            out = grp.mean()

        elif metode == "Minimum":
            out = grp.min()

        else:
            out = grp.max()

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
def plot_stl(data):

    y = data["rain"]

    if len(y) < 24:
        raise Exception("Data terlalu sedikit.")

    model = STL(
        y,
        period=12,
        robust=True
    )

    r = model.fit()

    fig, ax = plt.subplots(
        3,1,
        figsize=(12,8),
        sharex=True
    )

    ax[0].plot(data.index, r.trend, color="green")
    ax[0].set_ylabel("Tren")

    ax[1].plot(data.index, r.seasonal, color="red")
    ax[1].set_ylabel("Musiman")

    ax[2].plot(data.index, r.resid, color="gray")
    ax[2].set_ylabel("Residu")
    ax[2].set_xlabel("Tahun")

    for a in ax:
        a.grid(alpha=0.25)

    plt.tight_layout()

    return fig


# =====================================================
# RBEAST
# =====================================================
def plot_beast(data):

    try:
        y = data["rain"].astype(float).values

        hasil = beast(
            y,
            start=data.index[0].year,
            deltat=1/12,
            freq=12,
            season="harmonic"
        )

        trend = hasil.trend.Y
        seasonal = hasil.season.Y
        resid = y - trend - seasonal

    except Exception as e:

        fig, ax = plt.subplots(figsize=(10,4))
        ax.text(
            0.5,0.5,
            "RBEAST gagal dijalankan",
            ha="center",
            va="center",
            fontsize=14
        )
        ax.axis("off")
        return fig


# =====================================================
# SAVE PNG ZIP
# =====================================================
def simpan_zip(fig1, fig2, nama, th1, th2):

    tmp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".zip"
    )

    zip_path = tmp.name

    png1 = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".png"
    ).name

    png2 = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".png"
    ).name

    fig1.savefig(
        png1,
        dpi=200,
        bbox_inches="tight"
    )

    fig2.savefig(
        png2,
        dpi=200,
        bbox_inches="tight"
    )

    with zipfile.ZipFile(
        zip_path,
        "w"
    ) as z:

        z.write(
            png1,
            f"STL_{nama}_{th1}_{th2}.png"
        )

        z.write(
            png2,
            f"BEAST_{nama}_{th1}_{th2}.png"
        )

    return zip_path


# =====================================================
# PROSES
# =====================================================
def proses(pos_id, periode, metode, th1, th2):

    nama_pos = dict(get_pos())[pos_id]

    df = ambil_data(
        pos_id,
        th1,
        th2
    )

    if df.empty:
        raise gr.Error("Data kosong.")

    # availability
    hari_total = (
        pd.date_range(
            f"{th1}-01-01",
            f"{th2}-12-31",
            freq="D"
        ).size
    )

    hari_ada = len(df)

    persen = round(
        hari_ada / hari_total * 100,
        2
    )

    data = agregasi(
        df,
        periode,
        metode
    )

    fig1 = plot_stl(data)
    fig2 = plot_beast(data)

    zip_file = simpan_zip(
        fig1,
        fig2,
        nama_pos,
        th1,
        th2
    )

    ringkasan = f"""
Pos Hujan : {nama_pos}
Periode : {periode}
Metode : {metode}
Rentang Tahun : {th1}-{th2}

Data tersedia : {hari_ada:,} hari
Data seharusnya : {hari_total:,} hari
Ketersediaan : {persen} %

Jumlah data olahan : {len(data):,}
"""

    return fig1, fig2, ringkasan, zip_file


# =====================================================
# DROPDOWN DINAMIS
# =====================================================
def ubah_metode(periode):

    if periode == "Harian":
        return gr.update(
            visible=False
        )

    return gr.update(
        visible=True
    )


# =====================================================
# UI
# =====================================================
tema = gr.themes.Soft()

css = """
.gradio-container{
    max-width:1450px !important;
    margin:auto !important;
    font-family:'Poppins',sans-serif !important;
}

footer{display:none !important;}

@media (max-width:900px){
    body{
        zoom:0.8;
    }
}
"""

gr.HTML("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap" rel="stylesheet">
""")

with gr.Blocks(
    theme=tema,
    title="Dekomposisi Curah Hujan",
    css=css
) as demo:

    gr.Markdown("""
# Dekomposisi Curah Hujan  
STL dan BEAST berjalan bersamaan.
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

    with gr.Row():

        th1 = gr.Number(
            value=1980,
            label="Tahun Awal"
        )

        th2 = gr.Number(
            value=2025,
            label="Tahun Akhir"
        )

    btn = gr.Button(
        "Proses",
        variant="primary"
    )

    info = gr.Textbox(
        label="Ringkasan",
        lines=10
    )

    with gr.Row():
        out1 = gr.Plot(label="STL")
        out2 = gr.Plot(label="RBEAST")

    unduh = gr.File(
        label="Unduh PNG (ZIP)"
    )

    periode.change(
        fn=ubah_metode,
        inputs=periode,
        outputs=metode
    )

    btn.click(
        fn=proses,
        inputs=[
            pos,
            periode,
            metode,
            th1,
            th2
        ],
        outputs=[
            out1,
            out2,
            info,
            unduh
        ]
    )

demo.queue().launch(
    server_name="0.0.0.0",
    server_port=7860
)