# ==========================================================
# app.py
# Dekomposisi STL & BEAST — port dari thesis
# Sumber data: PostgreSQL | 6 metode agregasi
# ==========================================================

import os
import sys
import zipfile
import tempfile
import logging
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger("beast")

import numpy as np
import pandas as pd
import psycopg2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import gradio as gr

from statsmodels.tsa.seasonal import STL

# ==========================================================
# OPTIONAL RBEAST
# ==========================================================
try:
    from Rbeast import beast
    BEAST_READY = True
except Exception:
    BEAST_READY = False


# ==========================================================
# DATABASE — lazy connection
# ==========================================================
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

_conn = None

def get_conn():
    global _conn
    if _conn is None or _conn.closed:
        log.info("Connecting to DB...")
        _conn = psycopg2.connect(DATABASE_URL, sslmode="require")
        log.info("DB connected OK")
    return _conn


# ==========================================================
# MATPLOTLIB STYLE (thesis)
# ==========================================================
plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13
})

NAMA_BULAN = [
    "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "Oktober", "November", "Desember"
]
NAMA_MUSIM = ["JFM (Jan-Mar)", "AMJ (Apr-Jun)", "JAS (Jul-Sep)", "OND (Okt-Des)"]


# ==========================================================
# MASTER POS
# ==========================================================
def get_pos():
    sql = "SELECT DISTINCT pos_id, nama_pos FROM data_ch ORDER BY nama_pos"
    df = pd.read_sql(sql, get_conn())
    return [(r["nama_pos"], str(r["pos_id"])) for _, r in df.iterrows()]


# ==========================================================
# AMBIL DATA
# ==========================================================
def ambil_data(pos_id, th1, th2):
    sql = """
    SELECT tanggal, rain
    FROM data_ch
    WHERE pos_id=%s
    AND EXTRACT(YEAR FROM tanggal) BETWEEN %s AND %s
    ORDER BY tanggal
    """
    df = pd.read_sql(sql, get_conn(), params=[pos_id, int(th1), int(th2)])
    df["Tanggal"] = pd.to_datetime(df["tanggal"])
    df["Data"] = pd.to_numeric(df["rain"], errors="coerce")
    df = df.dropna(subset=["Data"])
    df["year"] = df["Tanggal"].dt.year
    df["month"] = df["Tanggal"].dt.month
    df["season"] = ((df["month"] - 1) // 3) + 1
    return df


# ==========================================================
# OUTLIER DETECTION (port dari R check_outliers)
# ==========================================================
def has_outlier(data_vector):
    vec = data_vector[~np.isnan(data_vector)]
    if len(vec) < 4:
        return False
    q1, q3 = np.percentile(vec, [25, 75])
    iqr = q3 - q1
    return bool(np.any((vec < q1 - 1.5 * iqr) | (vec > q3 + 1.5 * iqr)))


# ==========================================================
# AGREGASI (6 metode dari thesis)
# ==========================================================
def agregasi(df, metode, bulan=None, musim=None):

    d = df.copy()

    if metode == "Maksimum Harian Tahunan":
        df_agg = d.groupby("year")["Data"].max().reset_index()
        df_agg["Tanggal"] = pd.to_datetime(df_agg["year"].astype(str) + "-01-01")
        df_agg = df_agg.rename(columns={"Data": "val"})[["Tanggal", "val"]]
        return df_agg, False

    elif metode == "Kumulatif Bulanan":
        df_agg = d.groupby(["year", "month"])["Data"].sum().reset_index()
        df_agg["Tanggal"] = pd.to_datetime(
            df_agg["year"].astype(str) + "-" + df_agg["month"].astype(str) + "-01"
        )
        df_agg = df_agg.rename(columns={"Data": "val"})[["Tanggal", "val"]]
        return df_agg, True

    elif metode == "Kumulatif Bulanan Khusus":
        m = int(bulan)
        df_f = d[d["month"] == m]
        df_agg = df_f.groupby("year")["Data"].sum().reset_index()
        df_agg["Tanggal"] = pd.to_datetime(
            df_agg["year"].astype(str) + f"-{m:02d}-01"
        )
        df_agg = df_agg.rename(columns={"Data": "val"})[["Tanggal", "val"]]
        return df_agg, False

    elif metode == "Kumulatif Musiman":
        df_agg = d.groupby(["year", "season"])["Data"].sum().reset_index()
        smap = {1: "01", 2: "04", 3: "07", 4: "10"}
        df_agg["Tanggal"] = pd.to_datetime(
            df_agg["year"].astype(str) + "-" + df_agg["season"].map(smap) + "-01"
        )
        df_agg = df_agg.sort_values("Tanggal").reset_index(drop=True)
        df_agg = df_agg.rename(columns={"Data": "val"})[["Tanggal", "val"]]
        return df_agg, True

    elif metode == "Kumulatif Musiman Khusus":
        s = int(musim)
        df_f = d[d["season"] == s]
        df_agg = df_f.groupby("year")["Data"].sum().reset_index()
        smap = {1: "01", 2: "04", 3: "07", 4: "10"}
        df_agg["Tanggal"] = pd.to_datetime(
            df_agg["year"].astype(str) + f"-{smap[s]}-01"
        )
        df_agg = df_agg.rename(columns={"Data": "val"})[["Tanggal", "val"]]
        return df_agg, False

    elif metode == "Kumulatif Tahunan":
        df_agg = d.groupby("year")["Data"].sum().reset_index()
        df_agg["Tanggal"] = pd.to_datetime(df_agg["year"].astype(str) + "-01-01")
        df_agg = df_agg.rename(columns={"Data": "val"})[["Tanggal", "val"]]
        return df_agg, False

    raise ValueError(f"Metode tidak dikenal: {metode}")


# ==========================================================
# PARAMETER DEKOMPOSISI (dari thesis)
# ==========================================================
def get_stl_param(has_seasonality, metode):
    if not has_seasonality:
        return None
    if metode == "Kumulatif Bulanan":
        return {"period": 12, "seasonal": 13, "trend": 21}
    elif metode == "Kumulatif Musiman":
        return {"period": 4, "seasonal": 7, "trend": 11}
    return None


def get_beast_param(has_seasonality, metode):
    if metode == "Kumulatif Bulanan":
        return {"freq": 12, "deltat": 1/12, "season": "harmonic"}
    elif metode == "Kumulatif Musiman":
        return {"freq": 4, "deltat": 1/4, "season": "harmonic"}
    return {"freq": 1, "deltat": 1, "season": "none"}


# ==========================================================
# STL DEKOMPOSISI (thesis style — trend only)
# ==========================================================
def plot_stl(df_agg, has_seasonality, metode, nama_pos):

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 5))
    TREND_COLOR = "#00B300"

    if has_seasonality:
        prm = get_stl_param(True, metode)
        y = df_agg["val"].ffill()
        if len(y) < prm["period"] * 2:
            y = y.reindex(range(max(len(y), prm["period"] * 2))).ffill()
        stl = STL(y, period=prm["period"], seasonal=prm["seasonal"],
                  trend=prm["trend"], robust=True)
        result = stl.fit()
        trend = result.trend
        title_detail = metode
        ax.plot(df_agg.index, trend, color=TREND_COLOR, linewidth=2.0)
    else:
        window_size = max(3, len(df_agg) // 4)
        if window_size % 2 == 0:
            window_size += 1
        trend = df_agg["val"].rolling(window=window_size, center=True, min_periods=1).mean()
        title_detail = metode + " (Moving Average)"
        ax.plot(df_agg.index, trend, color=TREND_COLOR, linewidth=2.0)

    ax.set_title(
        f"$\\mathit{{Trend}}$ Data Hujan: {nama_pos}\nMetode: {title_detail}",
        fontsize=15, fontweight="bold", pad=15
    )
    ax.set_ylabel(r"$\mathit{Trend}$ (mm)")
    ax.set_xlabel("Tahun")

    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.tick_params(axis="both", which="both", bottom=True, left=True,
                   direction="out", length=6, width=1.2, color="#333333")
    for side in ["bottom", "left", "top", "right"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("#333333")
        ax.spines[side].set_linewidth(1.2)

    plt.tight_layout()
    return fig, trend.values


# ==========================================================
# BEAST DEKOMPOSISI (thesis style — trend only)
# ==========================================================
def plot_beast(df_agg, has_seasonality, metode, nama_pos):

    if not BEAST_READY:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.text(0.5, 0.5, "RBEAST tidak tersedia", ha="center", va="center")
        ax.set_title(f"BEAST — {nama_pos}", fontsize=15, fontweight="bold")
        ax.axis("off")
        return fig, [], [], [], []

    try:
        prm = get_beast_param(has_seasonality, metode)
        y = df_agg["val"].values.astype(float)

        if metode == "Kumulatif Bulanan":
            start_year = df_agg.index[0].year + (df_agg.index[0].month - 1) / 12
        elif metode == "Kumulatif Musiman":
            month_first = df_agg.index[0].month
            season_first = ((month_first - 1) // 3) + 1
            start_year = df_agg.index[0].year + (season_first - 1) / 4
        else:
            start_year = df_agg.index[0].year

        outlier_flag = has_outlier(y)
        tseg_min_val = 24 if prm["freq"] == 12 else 8 if prm["freq"] == 4 else max(3, len(y) // 4)

        hasil = beast(
            y,
            start=start_year,
            deltat=prm["deltat"],
            freq=prm["freq"],
            season=prm["season"],
            scp_minmax=[0, 1],
            sorder_minmax=[1, 3],
            tcp_minmax=[0, 4],
            torder_minmax=[0, 1],
            tseg_min=tseg_min_val,
            hasOutlier=outlier_flag,
            mcmc_samples=8000,
            mcmc_chains=3
        )

        trend = hasil.trend.Y

        sd_vals = []
        try:
            sd_vals = hasil.trend.SD.tolist()
        except Exception:
            pass

        cp_indices = []
        cp_probs = []
        try:
            ncp = int(hasil.trend.ncp[0])
            for k in range(ncp):
                idx = int(hasil.trend.cp[k])
                pr = float(hasil.trend.cpPr[k])
                if idx < len(df_agg) and pr >= 0.5:
                    cp_indices.append(idx)
                    cp_probs.append(round(pr, 4))
        except Exception:
            pass
        cp_dates = [df_agg.index[i].strftime("%Y-%m-%d") for i in cp_indices]

        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(12, 5))
        TREND_COLOR = "#00B300"

        ax.plot(df_agg.index, trend, color=TREND_COLOR, linewidth=2.0)

        if len(sd_vals):
            sd = np.array(sd_vals)
            ax.fill_between(df_agg.index, trend - sd, trend + sd,
                            alpha=0.2, color=TREND_COLOR)

        for i in cp_indices:
            ax.axvline(df_agg.index[i], color="blue", ls="--", alpha=0.7)

        title_detail = metode + (" (BEAST + Outlier)" if outlier_flag else " (BEAST)")
        ax.set_title(
            f"$\\mathit{{Trend}}$ Data Hujan: {nama_pos}\nMetode: {title_detail}",
            fontsize=15, fontweight="bold", pad=15
        )
        ax.set_ylabel(r"$\mathit{Trend}$ (mm)")
        ax.set_xlabel("Tahun")

        ax.xaxis.set_major_locator(mdates.YearLocator(10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.grid(False)
        ax.yaxis.grid(False)
        ax.tick_params(axis="both", which="both", bottom=True, left=True,
                       direction="out", length=6, width=1.2, color="#333333")
        for side in ["bottom", "left", "top", "right"]:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color("#333333")
            ax.spines[side].set_linewidth(1.2)

        plt.tight_layout()
        return fig, trend.tolist(), sd_vals, cp_dates, cp_probs

    except Exception as e:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.text(0.5, 0.5, str(e), ha="center", va="center", wrap=True)
        ax.set_title(f"BEAST Error — {nama_pos}", fontsize=15, fontweight="bold")
        ax.axis("off")
        return fig, [], [], [], []


# ==========================================================
# ZIP PNG
# ==========================================================
def simpan_zip(fig1, fig2, nama, th1, th2):
    zf = tempfile.NamedTemporaryFile(delete=False, suffix=".zip").name
    p1 = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    p2 = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    fig1.savefig(p1, dpi=220, bbox_inches="tight")
    fig2.savefig(p2, dpi=220, bbox_inches="tight")
    with zipfile.ZipFile(zf, "w") as z:
        z.write(p1, f"STL_{nama}_{th1}_{th2}.png")
        z.write(p2, f"BEAST_{nama}_{th1}_{th2}.png")
    return zf


# ==========================================================
# CEK DATA
# ==========================================================
def cek_data(pos_id, th1, th2):
    if not pos_id:
        raise gr.Error("Pilih pos hujan dahulu.")
    nama = dict((v, k) for k, v in get_pos())[pos_id]
    df = ambil_data(pos_id, th1, th2)
    total = len(pd.date_range(f"{int(th1)}-01-01", f"{int(th2)}-12-31", freq="D"))
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
    return gr.update(value=teks, visible=True)


# ==========================================================
# PROSES
# ==========================================================
def proses(pos_id, metode, th1, th2, bulan, musim):

    if not pos_id:
        raise gr.Error("Pilih pos hujan dahulu.")

    nama = dict((v, k) for k, v in get_pos())[pos_id]

    if metode in ("Kumulatif Bulanan Khusus",) and not bulan:
        raise gr.Error("Pilih bulan untuk Kumulatif Bulanan Khusus.")
    if metode in ("Kumulatif Musiman Khusus",) and not musim:
        raise gr.Error("Pilih musim untuk Kumulatif Musiman Khusus.")

    df = ambil_data(pos_id, th1, th2)
    if df.empty:
        raise gr.Error("Data kosong.")

    df_agg, has_seasonality = agregasi(df, metode, bulan, musim)
    df_agg = df_agg.set_index("Tanggal").sort_index()
    df_agg["val"] = df_agg["val"].ffill().bfill()

    fig1, trend_stl = plot_stl(df_agg, has_seasonality, metode, nama)
    fig2, trend_beast, _, _, _ = plot_beast(df_agg, has_seasonality, metode, nama)
    zipf = simpan_zip(fig1, fig2, nama, th1, th2)

    trend_label = "Ada tren" if len(trend_stl) > 1 else "-"
    outlier_flag = has_outlier(df_agg["val"].values)

    info = f"""
Pos Hujan     : {nama}
Metode        : {metode}
Jumlah Data   : {len(df_agg):,}
Outlier       : {"Ya" if outlier_flag else "Tidak"}
Seasonal      : {"Ya" if has_seasonality else "Tidak (Moving Average)"}
Trend STL     : {trend_label}
"""

    return (
        gr.update(visible=False),
        gr.update(visible=True),
        info,
        fig1,
        fig2,
        zipf
    )


# ==========================================================
# UI VISIBILITY
# ==========================================================
def ubah_metode(metode):
    show_bulan = metode == "Kumulatif Bulanan Khusus"
    show_musim = metode == "Kumulatif Musiman Khusus"
    return (
        gr.update(visible=show_bulan),
        gr.update(visible=show_musim),
    )


# ==========================================================
# API ENDPOINT — JSON untuk project01
# ==========================================================
def api_analyze(pos_id, metode, th1, th2, bulan=None, musim=None):
    import json as _json

    if not pos_id:
        return _json.dumps({"error": "Pilih pos hujan."})

    nama_map = {v: k for k, v in get_pos()}
    nama = nama_map.get(pos_id, pos_id)

    df = ambil_data(pos_id, th1, th2)
    if df.empty:
        return _json.dumps({"error": "Data kosong."})

    df_agg, has_seasonality = agregasi(df, metode, bulan, musim)
    df_agg = df_agg.set_index("Tanggal").sort_index()
    df_agg["val"] = df_agg["val"].ffill().bfill()

    outlier_flag = has_outlier(df_agg["val"].values)

    try:
        _, trend_stl = plot_stl(df_agg, has_seasonality, metode, nama)
    except Exception:
        trend_stl = []

    try:
        _, trend_beast, sd_vals, cp_dates, cp_probs = plot_beast(df_agg, has_seasonality, metode, nama)
    except Exception:
        trend_beast, sd_vals, cp_dates, cp_probs = [], [], [], []

    ci_lower = []
    ci_upper = []
    if trend_beast and sd_vals and len(trend_beast) == len(sd_vals):
        ci_lower = [round(t - s, 2) for t, s in zip(trend_beast, sd_vals)]
        ci_upper = [round(t + s, 2) for t, s in zip(trend_beast, sd_vals)]

    result = {
        "pos": nama,
        "metode": metode,
        "dates": [d.strftime("%Y-%m-%d") for d in df_agg.index],
        "values": [round(v, 2) for v in df_agg["val"].tolist()],
        "trend_stl": [round(v, 2) for v in trend_stl] if len(trend_stl) else [],
        "trend_beast": [round(v, 2) for v in trend_beast] if len(trend_beast) else [],
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "change_points": cp_dates,
        "change_point_probs": cp_probs,
        "has_seasonality": has_seasonality,
        "has_outlier": outlier_flag,
        "count": len(df_agg),
    }

    return _json.dumps(result)


# ==========================================================
# CSS
# ==========================================================
css = """
.gradio-container{
max-width:1900px !important;
padding:30px 50px !important;
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
METODE_LIST = [
    "Kumulatif Bulanan",
    "Kumulatif Bulanan Khusus",
    "Kumulatif Musiman",
    "Kumulatif Musiman Khusus",
    "Kumulatif Tahunan",
    "Maksimum Harian Tahunan",
]

with gr.Blocks(css=css, title="Dekomposisi Curah Hujan") as demo:

    gr.Markdown("# 🌧️ Dashboard Dekomposisi Curah Hujan")

    with gr.Row():
        pos = gr.Dropdown(
            choices=[],
            label="Pos Hujan",
            scale=3,
            interactive=True
        )
        metode = gr.Dropdown(
            METODE_LIST,
            value="Kumulatif Bulanan",
            label="Metode Agregasi",
            scale=2
        )
        bulan = gr.Dropdown(
            [(b, str(i+1)) for i, b in enumerate(NAMA_BULAN)],
            label="Pilih Bulan",
            visible=False,
            scale=1
        )
        musim = gr.Dropdown(
            [(m, str(i+1)) for i, m in enumerate(NAMA_MUSIM)],
            label="Pilih Musim",
            visible=False,
            scale=1
        )

    with gr.Row():
        th1 = gr.Number(value=1980, label="Tahun Awal")
        th2 = gr.Number(value=2025, label="Tahun Akhir")

    with gr.Row():
        btncek = gr.Button("🔍 Cek Data")
        btn = gr.Button("⚙️ Proses")
        status_loading = gr.Textbox(
            value="", visible=False, interactive=False,
            elem_id="loading_status"
        )

    cekbox = gr.Textbox(label="Status Data", lines=8, visible=False)

    hasil = gr.Column(visible=False)
    with hasil:
        ring = gr.Textbox(label="Ringkasan", lines=7)
        with gr.Row():
            out1 = gr.Plot(label="STL — Trend")
            out2 = gr.Plot(label="BEAST — Trend")
        unduh = gr.File(label="Unduh PNG")

    # events
    def load_stations():
        try:
            log.info("Loading stations...")
            stations = get_pos()
            log.info(f"Loaded {len(stations)} stations")
            return gr.update(choices=stations)
        except Exception as e:
            log.error(f"Failed to load stations: {e}")
            return gr.update(choices=[], value=None)

    demo.load(fn=load_stations, outputs=pos)

    btncek.click(
        fn=cek_data,
        inputs=[pos, th1, th2],
        outputs=cekbox
    ).then(
        fn=lambda: gr.update(visible=False),
        outputs=hasil
    )

    btn.click(
        fn=lambda: (gr.update(visible=False), gr.update(visible=True, value="⏳ Memproses STL & BEAST... Mohon tunggu.")),
        outputs=[cekbox, status_loading]
    ).then(
        fn=proses,
        inputs=[pos, metode, th1, th2, bulan, musim],
        outputs=[cekbox, hasil, ring, out1, out2, unduh]
    ).then(
        fn=lambda: gr.update(visible=False),
        outputs=status_loading
    )

    metode.change(
        fn=ubah_metode,
        inputs=metode,
        outputs=[bulan, musim]
    )

    # API endpoint — hidden, callable via /api/api_analyze
    api_pos = gr.Textbox(visible=False)
    api_met = gr.Textbox(visible=False)
    api_th1 = gr.Number(visible=False)
    api_th2 = gr.Number(visible=False)
    api_bulan = gr.Textbox(visible=False)
    api_musim = gr.Textbox(visible=False)
    api_out = gr.Textbox(visible=False)

    api_btn = gr.Button(visible=False)
    api_btn.click(
        fn=api_analyze,
        inputs=[api_pos, api_met, api_th1, api_th2, api_bulan, api_musim],
        outputs=api_out
    )


# ==========================================================
# RUN
# ==========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=port
    )
