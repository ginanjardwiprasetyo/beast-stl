# Dekomposisi Curah Hujan — STL & BEAST

Dashboard dekomposisi time series curah hujan menggunakan **STL** (Seasonal-Trend decomposition using LOESS) dan **BEAST** (Bayesian Estimator of Abrupt change, Seasonal change, and Trend).

**Live:** https://beast-stl.rekayasa-sipil.my.id/

## Fitur

- 21 metode agregasi (Kumulatif, Rerata, Maksimum, Minimum — Bulanan, Musiman, Tahunan, Khusus)
- STL dekomposisi dengan parameter terkalibrasi
- BEAST change point detection (Bayesian)
- Outlier detection otomatis (IQR)
- Moving Average fallback untuk data non-seasonal
- Gradio API untuk integrasi dengan web lain
- Keep-alive via GitHub Actions (tiap 10 menit)

## API

Gradio expose endpoint otomatis:

```
POST https://beast-stl.rekayasa-sipil.my.id/gradio_api/api/api_analyze
POST https://beast-stl.rekayasa-sipil.my.id/gradio_api/api/api_analyze_data
```

### api_analyze — data dari database

```json
{
  "data": ["POS_ID", "METODE", TAHUN_AWAL, TAHUN_AKHIR, "BULAN", "MUSIM"]
}
```

Contoh:

```bash
curl -X POST https://beast-stl.rekayasa-sipil.my.id/gradio_api/api/api_analyze \
  -H "Content-Type: application/json" \
  -d '{"data":["37","Kumulatif Bulanan",2000,2025,"",""]}'
```

### api_analyze_data — data upload langsung

```json
{
  "data": ["[{\"date\":\"2020-01-01\",\"value\":10},...]", "METODE", "BULAN", "MUSIM"]
}
```

## Deploy

1. Push ke GitHub
2. Render - New Web Service - Python
3. Set env var: `DATABASE_URL = postgresql://...`
4. Build: `pip install -r requirements.txt`
5. Start: `python app.py`

## Lokal

```bash
pip install -r requirements.txt
export DATABASE_URL="postgresql://..."
python app.py
```

Service jalan di http://localhost:7860
