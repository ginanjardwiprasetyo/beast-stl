# ==========================================================
# app.py
# HuggingFace Spaces - Gradio
# STL + RBEAST + Cek Data + Loader + Supabase
# ==========================================================

import os
import zipfile
import tempfile
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
import gradio as gr

from statsmodels.tsa.seasonal import STL

# ----------------------------------------------------------
# OPTIONAL RBEAST
# ----------------------------------------------------------
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
# MATPLOTLIB
# ==========================================================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
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

    pilihan = []

    for _, r in df.iterrows():
        pilihan.append(
            (r["nama_pos"], str(r["pos_id"]))
        )

    return pilihan


# ==========================================================
# AMBIL DATA
# ==========================================================
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

    d = df.copy()
    d = d.set_index("tanggal")

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

    nama = next(
        nama for nama, pid in get_pos()
        if pid == pos_id
    )

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

    persen = round((ada / total) * 100, 2)

    if persen >= 90:
        status = "🟢 Sangat Baik"
    elif persen >= 75:
        status = "🟡 Cukup"
    elif persen >= 50:
        status = "🟠 Warning"
    else:
        status = "🔴 Buruk"

    teks = f"""
Pos Hujan       : {nama}
Rentang Tahun   : {int(th1)} - {int(th2)}

Data Ada        : {ada:,} hari
Data Hilang     : {hilang:,} hari
Data Seharusnya : {total:,} hari

Ketersediaan    : {persen} %
Status          : {status}
"""

    return teks


# ==========================================================
# STL
# ==========================================================
def plot_stl(data):

    y = data["rain"]

    period = 12 if len(y) >= 24 else max(2, int(len(y)/2))

    model = STL(
        y,
        period=period,
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


# ==========================================================
# RBEAST
# ==========================================================
def plot_beast(data):

    if not BEAST_READY:
        fig, ax = plt.subplots(figsize=(10,4))
        ax.text(0.5,0.5,"RBEAST tidak tersedia",ha="center",va="center")
        ax.axis("off")
        return fig

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

        fig, ax = plt.subplots(
            3,1,
            figsize=(12,8),
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

    except:

        fig, ax = plt.subplots(figsize=(10,4))
        ax.text(0.5,0.5,"RBEAST gagal dijalankan",ha="center",va="center")
        ax.axis("off")
        return fig


# ==========================================================
# ZIP PNG
# ==========================================================
def simpan_zip(fig1, fig2, nama, th1, th2):

    zip_path = tempfile.NamedTemporaryFile(
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

    with zipfile.ZipFile(zip_path, "w") as z:

        z.write(
            p1,
            f"STL_{nama}_{int(th1)}_{int(th2)}.png"
        )

        z.write(
            p2,
            f"BEAST_{nama}_{int(th1)}_{int(th2)}.png"
        )

    return zip_path


# ==========================================================
# PROSES
# ==========================================================
def proses(pos_id, periode, metode, th1, th2):

    nama = next(
        nama for nama, pid in get_pos()
        if pid == pos_id
    )

    df = ambil_data(pos_id, th1, th2)

    if df.empty:
        raise gr.Error("Data kosong.")

    data = agregasi(df, periode, metode)

    fig1 = plot_stl(data)
    fig2 = plot_beast(data)

    zip_file = simpan_zip(
        fig1, fig2, nama, th1, th2
    )

    ringkas = cek_data(pos_id, th1, th2)

    return (
        gr.update(visible=False),
        gr.update(visible=True),
        ringkas,
        fig1,
        fig2,
        zip_file
    )


# ==========================================================
# SHOW / HIDE METODE
# ==========================================================
def ubah_metode(periode):

    if periode == "Harian":
        return gr.update(visible=False)

    return gr.update(visible=True)


# ==========================================================
# CSS DESKTOP
# ==========================================================
css = """
body{
    background:#eef2f7;
}

.gradio-container{
    max-width:1850px !important;
    margin:auto !important;
    padding:25px 45px !important;
    font-family:Arial,sans-serif !important;
}

h1,h2,h3{
    text-align:center;
}

textarea{
    font-size:15px !important;
}

button{
    height:52px !important;
    font-size:16px !important;
}

#loaderbox{
    text-align:center;
    background:white;
    padding:40px;
    border-radius:18px;
    box-shadow:0 8px 20px rgba(0,0,0,.05);
}

.spin{
    width:56px;
    height:56px;
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

#timer{
    margin-top:18px;
    font-size:18px;
    font-weight:600;
}
"""


# ==========================================================
# UI
# ==========================================================
with gr.Blocks(
    title="Curah Hujan",
    css=css
) as demo:

    gr.Markdown("""
# 🌧️ Dashboard Dekomposisi Curah Hujan

Cek data dahulu, lalu olah dengan STL dan RBEAST.
""")

    with gr.Row():

        pos = gr.Dropdown(
            choices=get_pos(),
            label="Pos Hujan",
            scale=3
        )

        periode = gr.Dropdown(
            choices=["Harian","Bulanan","Tahunan"],
            value="Bulanan",
            label="Periode",
            scale=1
        )

        metode = gr.Dropdown(
            choices=["Kumulatif","Rerata","Minimum","Maksimum"],
            value="Kumulatif",
            label="Metode",
            scale=1
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

    with gr.Row():

        btn_cek = gr.Button("🔍 Cek Ketersediaan Data")
        btn_proses = gr.Button("⚙️ Proses Data", variant="primary")

    cek_box = gr.Textbox(
        label="Status Data",
        lines=10
    )

    loader = gr.HTML("""
<div id="loaderbox" style="display:none;">
<div class="spin"></div>
<div id="timer">Memproses... 0 detik</div>
</div>
""")

    hasil = gr.Column(visible=False)

    with hasil:

        ringkasan = gr.Textbox(
            label="Ringkasan",
            lines=10
        )

        with gr.Row():
            out1 = gr.Plot(label="STL")
            out2 = gr.Plot(label="RBEAST")

        unduh = gr.File(
            label="Unduh PNG (ZIP)"
        )

    # --------------------------------------
    # EVENT
    # --------------------------------------
    periode.change(
        fn=ubah_metode,
        inputs=periode,
        outputs=metode
    )

    btn_cek.click(
        fn=cek_data,
        inputs=[pos, th1, th2],
        outputs=cek_box
    )

    btn_proses.click(
    fn=lambda: (
        gr.update(
            value="""
<div id='loaderbox'>
<div class='spin'></div>
<div id='timer'>Memproses... 0 detik</div>
</div>
"""
        ),
        gr.update(visible=False)
    ),
    outputs=[loader, cek_box],
    js="""
() => {
let box=document.getElementById("loaderbox");
if(box){box.style.display="block";}
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
        outputs=[
            loader,
            hasil,
            ringkasan,
            out1,
            out2,
            unduh
        ]
    ).then(
        fn=lambda: None,
        js="""
() => {
clearInterval(window.loop);
let box=document.getElementById("loaderbox");
if(box){box.style.display="none";}
}
"""
    )


# ==========================================================
# RUN
# ==========================================================
if __name__ == "__main__":

    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860
    )