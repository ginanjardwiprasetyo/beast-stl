# ==========================================================
# app.py
# HuggingFace Spaces - Gradio
# STL + RBEAST + Change Point + Loader + Scroll Dropdown
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
    "font.size": 11
})


# ==========================================================
# POS HUJAN
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
# CEK DATA
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

    return f"""
Pos Hujan       : {nama}
Tahun           : {int(th1)} - {int(th2)}
Data Ada        : {ada:,}
Data Hilang     : {hilang:,}
Ketersediaan    : {persen} %
Status          : {status}
"""


# ==========================================================
# STL
# ==========================================================
def plot_stl(data):

    y = data["rain"].ffill()

    p = 12 if len(y) >= 24 else max(2, int(len(y)/2))

    model = STL(
        y,
        period=p,
        robust=True
    )

    r = model.fit()

    fig, ax = plt.subplots(3,1, figsize=(12,8), sharex=True)

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
def plot_beast(data):

    if not BEAST_READY:
        fig, ax = plt.subplots(figsize=(10,4))
        ax.text(.5,.5,"RBEAST tidak tersedia",ha="center")
        ax.axis("off")
        return fig

    try:

        y = data["rain"].astype(float).values

        hasil = beast(
            y,
            start=data.index[0].year,
            deltat=1/12,
            freq=12,
            season="harmonic",
            hasOutlier=True
        )

        trend = hasil.trend.Y
        seasonal = hasil.season.Y
        resid = y - trend - seasonal

        fig, ax = plt.subplots(3,1, figsize=(12,8), sharex=True)

        # trend
        ax[0].plot(data.index, trend, color="green")

        # confidence band jika ada
        try:
            sd = hasil.trend.SD
            ax[0].fill_between(
                data.index,
                trend - sd,
                trend + sd,
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

        # seasonal
        ax[1].plot(data.index, seasonal, color="red")
        ax[1].set_ylabel("Musiman")

        # resid
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
# ZIP
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

    data = agregasi(df, periode, metode)

    fig1 = plot_stl(data)
    fig2 = plot_beast(data)

    zipf = simpan_zip(fig1, fig2, nama, th1, th2)

    info = cek_data(pos_id, th1, th2)

    return (
        gr.update(visible=False),
        gr.update(visible=True),
        info,
        fig1,
        fig2,
        zipf
    )


# ==========================================================
# CSS
# ==========================================================
css = """
.gradio-container{
max-width:1900px !important;
padding:30px 50px !important;
}

footer{display:none !important;}

textarea{font-size:15px !important;}

button{height:52px !important;}

#loaderbox{
text-align:center;
padding:40px;
background:white;
border-radius:18px;
}

.spin{
width:55px;
height:55px;
border:6px solid #dbeafe;
border-top:6px solid #2563eb;
border-radius:50%;
margin:auto;
animation:putar 1s linear infinite;
}

@keyframes putar{
from{transform:rotate(0)}
to{transform:rotate(360deg)}
}

/* dropdown scroll */
.wrap.svelte-1ipelgc{
max-height:380px !important;
overflow-y:auto !important;
}
"""


# ==========================================================
# UI
# ==========================================================
with gr.Blocks(css=css, title="Curah Hujan") as demo:

    gr.Markdown("# 🌧️ Dashboard Curah Hujan")

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
        th1 = gr.Number(value=1980)
        th2 = gr.Number(value=2025)

    with gr.Row():
        btncek = gr.Button("🔍 Cek Data")
        btn = gr.Button("⚙️ Proses", variant="primary")

    cekbox = gr.Textbox(lines=8, label="Status Data")

    loader = gr.HTML()

    hasil = gr.Column(visible=False)

    with hasil:

        ring = gr.Textbox(lines=8, label="Ringkasan")

        with gr.Row():
            out1 = gr.Plot(label="STL")
            out2 = gr.Plot(label="RBEAST")

        unduh = gr.File(label="Unduh PNG")

    btncek.click(
        fn=cek_data,
        inputs=[pos, th1, th2],
        outputs=cekbox
    )

    btn.click(
        fn=lambda: (
            gr.update(value="""
<div id='loaderbox'>
<div class='spin'></div>
<div id='timer'>Memproses... 0 detik</div>
</div>
"""),
            gr.update(visible=False),
            gr.update(value="")
        ),
        outputs=[loader, hasil, cekbox],
        js="""
() => {
window.detik=0;
window.loop=setInterval(()=>{
window.detik++;
let t=document.getElementById("timer");
if(t){t.innerText="Memproses... "+window.detik+" detik";}
},1000);
}
"""
    ).then(
        fn=proses,
        inputs=[pos, periode, metode, th1, th2],
        outputs=[loader, hasil, ring, out1, out2, unduh]
    ).then(
        fn=lambda:"",
        outputs=loader,
        js="""
() => {clearInterval(window.loop);}
"""
    )


if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860
    )