# 🌧️ Dekomposisi Curah Hujan — STL & BEAST

Dashboard dekomposisi time series curah hujan menggunakan **STL** (Seasonal-Trend decomposition using LOESS) dan **BEAST** (Bayesian Estimator of Abrupt change, Seasonal change, and Trend).

## Fitur

- 6 metode agregasi
- Outlier detection otomatis (IQR)
- Moving Average fallback untuk data non-seasonal
- BEAST dengan parameter terkalibrasi
- Gradio API untuk integrasi dengan web lain

## Deploy ke Render

1. Push ke GitHub
2. Render → New Web Service → Python
3. Set env var: `DATABASE_URL = postgresql://...`
4. Build: `pip install -r requirements.txt`
5. Start: `python app.py`

## API

Gradio otomatis expose endpoint:

```
POST https://your-app.onrender.com/api/predict
```

## Lokal

```bash
pip install -r requirements.txt
export DATABASE_URL="postgresql://..."
python app.py
```
