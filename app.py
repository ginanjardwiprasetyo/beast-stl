# ==========================================================
# app.py
# FINAL 
# - parameter STL / BEAST menyesuaikan harian bulanan tahunan
# ==========================================================

import os
import zipfile
import tempfile
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
import gradio as gr

from statsmodels.tsa.seasonal import STL

# ==========================================================
# OPTIONAL RBEAST
# ==========================================================
try:
    from Rbeast import beast
    BEAST_READY = True
except:
    BEAST_READY = False


# ==========================================================
# DATABASE
# ==========================================================
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

conn = psycopg2.connect(
    DATABASE_URL,
    sslmode="require"
)


# ==========================================================
# STYLE
# ==========================================================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13
})


# ==========================================================
# MASTER POS
# ==========================================================
def get_pos():

    sql = """
    SELECT DISTINCT pos_id, nama_pos
    FROM data_ch
    ORDER BY nama_pos
    """

    df = pd.read_sql(sql, conn)

    return [
        (r["nama_pos"], str(r["pos_id"]))
        for _, r in df.iterrows()
    ]


# ==========================================================
# DATA
# ==========================================================
def ambil_data(pos_id, th1, th2):

    sql = """
    SELECT tanggal, rain
    FROM data_ch
    WHERE pos_id=%s
    AND EXTRACT(YEAR FROM tanggal) BETWEEN %s AND %s
    ORDER BY tanggal
    """

    df = pd.read_sql(
        sql,
        conn,
        params=[pos_id, int(th1), int(th2)]
    )

    df["tanggal"] = pd.to_datetime(df["tanggal"])
    df["rain"] = pd.to_numeric(df["rain"], errors="coerce")
    df = df.dropna()

    return df


# ==========================================================
# AGREGASI
# ==========================================================
def agregasi(df, periode, metode):

    d = df.copy().set_index("tanggal")

    if periode == "Harian":
        out = d.resample("D").sum()

    elif periode == "Bulanan":

        grp = d.resample("MS")

        if metode == "Rerata":
            out = grp.mean()
        elif metode == "Minimum":
            out = grp.min()
        elif metode == "Maksimum":
            out = grp.max()
        else:
            out = grp.sum()

    else:

        grp = d.resample("YS")

        if metode == "Rerata":
            out = grp.mean()
        elif metode == "Minimum":
            out = grp.min()
        elif metode == "Maksimum":
            out = grp.max()
        else:
            out = grp.sum()

    out = out.dropna()
    out["rain"] = out["rain"].round(0)

    return out


# ==========================================================
# STATUS DATA
# ==========================================================
def cek_data(pos_id, th1, th2):

    if not pos_id:
        raise gr.Error("Pilih pos hujan dahulu.")

    nama = dict((v, k) for k, v in get_pos())[pos_id]

    df = ambil_data(pos_id, th1, th2)

    total = len(
        pd.date_range(
            f"{int(th1)}-01-01",
            f"{int(th2)}-12-31",
            freq="D"
        )
    )

    ada = len(df)
    hilang = total - ada
    persen = round(ada / total * 100, 2)

    if persen >= 90:
        status = "🟢 Sangat Baik"
    elif persen >= 75:
        status = "🟡 Cukup"
    else:
        status = "🔴 Warning"

    teks = f"""
Pos Hujan       : {nama}
Rentang Tahun   : {int(th1)} - {int(th2)}
Data Ada        : {ada:,}
Data Hilang     : {hilang:,}
Ketersediaan    : {persen} %
Status          : {status}
"""

    return gr.update(
        value=teks,
        visible=True
    )


# ==========================================================
# PARAMETER BERDASAR PERIODE
# ==========================================================
def get_param(periode):

    # period STL
    # freq BEAST
    # deltat interval tahun

    if periode == "Harian":
        return {
            "period": 365,
            "freq": 365,
            "deltat": 1/365
        }

    elif periode == "Bulanan":
        return {
            "period": 12,
            "freq": 12,
            "deltat": 1/12
        }

    else:
        return {
            "period": 5,
            "freq": 1,
            "deltat": 1
        }


# ==========================================================
# STL
# ==========================================================
def plot_stl(data, periode):

    prm = get_param(periode)

    y = data["rain"].ffill()

    p = min(prm["period"], max(2, len(y)//2))

    model = STL(
        y,
        period=p,
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

    for a in ax:
        a.grid(alpha=.25)

    plt.tight_layout()
    return fig


# ==========================================================
# RBEAST
# ==========================================================
def plot_beast(data, periode):

    if not BEAST_READY:

        fig, ax = plt.subplots(figsize=(10,4))
        ax.text(.5,.5,"RBEAST tidak tersedia",ha="center")
        ax.axis("off")
        return fig

    try:

        prm = get_param(periode)

        y = data["rain"].values.astype(float)

        hasil = beast(
            y,
            start=data.index[0].year,
            deltat=prm["deltat"],
            freq=prm["freq"],
            season="harmonic",
            hasOutlier=True
        )

        trend = hasil.trend.Y
        seasonal = hasil.season.Y
        resid = y - trend - seasonal

        fig, ax = plt.subplots(
            3,1,
            figsize=(12,8),
            sharex=True
        )

        # trend
        ax[0].plot(data.index, trend, color="green")

        # confidence band
        try:
            sd = hasil.trend.SD
            ax[0].fill_between(
                data.index,
                trend-sd,
                trend+sd,
                alpha=.2
            )
        except:
            pass

        # change point
        try:
            cp = hasil.trend.cp
            for c in cp:
                i = int(c)
                if i < len(data):
                    ax[0].axvline(
                        data.index[i],
                        color="blue",
                        ls="--",
                        alpha=.7
                    )
        except:
            pass

        ax[0].set_ylabel("Tren")

        ax[1].plot(data.index, seasonal, color="red")
        ax[1].set_ylabel("Musiman")

        ax[2].plot(data.index, resid, color="gray")
        ax[2].set_ylabel("Residu")

        for a in ax:
            a.grid(alpha=.25)

        plt.tight_layout()
        return fig

    except Exception as e:

        fig, ax = plt.subplots(figsize=(10,4))
        ax.text(.5,.5,str(e),ha="center")
        ax.axis("off")
        return fig


# ==========================================================
# ZIP PNG
# ==========================================================
def simpan_zip(fig1, fig2, nama, th1, th2):

    zf = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".zip"
    ).name

    p1 = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".png"
    ).name

    p2 = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".png"
    ).name

    fig1.savefig(p1, dpi=220, bbox_inches="tight")
    fig2.savefig(p2, dpi=220, bbox_inches="tight")

    with zipfile.ZipFile(zf,"w") as z:
        z.write(p1, f"STL_{nama}_{th1}_{th2}.png")
        z.write(p2, f"BEAST_{nama}_{th1}_{th2}.png")

    return zf


# ==========================================================
# PROSES
# ==========================================================
def proses(pos_id, periode, metode, th1, th2):

    nama = dict((v,k) for k,v in get_pos())[pos_id]

    df = ambil_data(pos_id, th1, th2)

    if df.empty:
        raise gr.Error("Data kosong.")

    data = agregasi(df, periode, metode)

    fig1 = plot_stl(data, periode)
    fig2 = plot_beast(data, periode)

    zipf = simpan_zip(fig1, fig2, nama, th1, th2)

    info = f"""
Pos Hujan     : {nama}
Periode       : {periode}
Metode        : {metode}
Jumlah Data   : {len(data):,}
"""

    return (
        gr.update(visible=False),   # hide status data
        gr.update(visible=True),    # show hasil
        info,
        fig1,
        fig2,
        zipf
    )


# ==========================================================
# METODE
# ==========================================================
def ubah_metode(periode):

    if periode == "Harian":
        return gr.update(visible=False)

    return gr.update(visible=True)


# ==========================================================
# CSS
# ==========================================================
css = """
.gradio-container{
max-width:1900px !important;
padding:30px 50px !important;
}

footer{
display:none !important;
}

/* dropdown list 5 item */
.wrap.svelte-1ipelgc{
max-height:190px !important;
overflow-y:auto !important;
}

textarea{
font-size:15px !important;
}

button{
height:50px !important;
}
"""


# ==========================================================
# UI
# ==========================================================
with gr.Blocks(css=css, title="Curah Hujan") as demo:

    gr.Markdown("# 🌧️ Dashboard Dekomposisi Curah Hujan")

    with gr.Row():

        pos = gr.Dropdown(
            choices=get_pos(),
            label="Pos Hujan",
            scale=3
        )

        periode = gr.Dropdown(
            ["Harian","Bulanan","Tahunan"],
            value="Bulanan",
            scale=1
        )

        metode = gr.Dropdown(
            ["Kumulatif","Rerata","Minimum","Maksimum"],
            value="Kumulatif",
            scale=1
        )

    with gr.Row():

        th1 = gr.Number(value=1980, label="Tahun Awal")
        th2 = gr.Number(value=2025, label="Tahun Akhir")

    with gr.Row():

        btncek = gr.Button("🔍 Cek Data")
        btn = gr.Button("⚙️ Proses")

    cekbox = gr.Textbox(
        label="Status Data",
        lines=8,
        visible=False
    )

    hasil = gr.Column(visible=False)

    with hasil:

        ring = gr.Textbox(
            label="Ringkasan",
            lines=6
        )

        with gr.Row():
            out1 = gr.Plot(label="STL")
            out2 = gr.Plot(label="RBEAST")

        unduh = gr.File(label="Unduh PNG")

    # cek data
    btncek.click(
        fn=cek_data,
        inputs=[pos, th1, th2],
        outputs=cekbox
    ).then(
        fn=lambda: gr.update(visible=False),
        outputs=hasil
    )

    # proses
    btn.click(
        fn=lambda: gr.update(visible=False),
        outputs=cekbox
    ).then(
        fn=proses,
        inputs=[pos, periode, metode, th1, th2],
        outputs=[cekbox, hasil, ring, out1, out2, unduh]
    )

    periode.change(
        fn=ubah_metode,
        inputs=periode,
        outputs=metode
    )


# ==========================================================
# RUN
# ==========================================================
if __name__ == "__main__":

    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860
    )