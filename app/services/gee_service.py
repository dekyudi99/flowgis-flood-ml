import ee
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

KEY_PATH = os.path.join(os.path.dirname(__file__), ".private-key.json")
SERVICE_ACCOUNT_EMAIL = os.getenv("GEE_SERVICE_ACCOUNT")
DEFAULT_AOI_ASSET = os.getenv("GEE_AOI_ASSET")

_gee_initialized = False
_GEE_LAG_CACHE: Dict[str, Tuple[Dict[str, list], Dict[str, Any]]] = {}

# Inisialisasi GEE persis seperti pada flowgis-floodrisk-services
try:
    if not SERVICE_ACCOUNT_EMAIL and os.path.exists(KEY_PATH):
        with open(KEY_PATH, "r", encoding="utf-8") as _f:
            SERVICE_ACCOUNT_EMAIL = json.load(_f).get("client_email")
    credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT_EMAIL, KEY_PATH)
    ee.Initialize(credentials)
    _gee_initialized = True
    logging.info("GEE initialized successfully")
except Exception as e:
    logging.warning(f"GEE Initialization issue: {e}")


def init_gee() -> bool:
    global _gee_initialized
    if _gee_initialized:
        return True
    try:
        email = SERVICE_ACCOUNT_EMAIL
        if not email and os.path.exists(KEY_PATH):
            with open(KEY_PATH, "r", encoding="utf-8") as _f:
                email = json.load(_f).get("client_email")
        credentials = ee.ServiceAccountCredentials(email, KEY_PATH)
        ee.Initialize(credentials)
        _gee_initialized = True
        logging.info("GEE initialized successfully")
        return True
    except Exception as e:
        logging.warning(f"GEE Initialization issue: {e}")
        return False


def is_gee_ready() -> bool:
    return _gee_initialized


def fetch_gee_chirps_sum(lat: float, lon: float, start_date: str, end_date: str) -> Optional[float]:
    """
    Mengambil akumulasi curah hujan historis (mm) dari Google Earth Engine (UCSB-CHG/CHIRPS/DAILY).
    """
    if not init_gee():
        return None

    try:
        point = ee.Geometry.Point([lon, lat])
        collection = (
            ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
            .filterDate(start_date, end_date)
            .select("precipitation")
        )
        total_img = collection.sum()
        val = (
            total_img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=5000,
            )
            .get("precipitation")
            .getInfo()
        )
        if val is not None:
            return float(val)
    except Exception as e:
        logging.debug(f"GEE CHIRPS fetch fallback: {e}")

    return None


def fetch_gee_imerg_sum(lat: float, lon: float, start_date: str, end_date: str) -> Optional[float]:
    """
    Mengambil akumulasi curah hujan near-real-time (mm) dari Google Earth Engine
    (NASA/GPM_L3/IMERG_V07). Setiap citra berdurasi 30 menit dalam satuan mm/jam.
    """
    if not init_gee():
        return None

    try:
        point = ee.Geometry.Point([lon, lat])
        collection = (
            ee.ImageCollection("NASA/GPM_L3/IMERG_V07")
            .filterDate(start_date, end_date)
            .select("precipitation")
        )
        total_img = collection.sum().multiply(0.5)
        val = (
            total_img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=10000,
            )
            .get("precipitation")
            .getInfo()
        )
        if val is not None:
            return float(val)
    except Exception as e:
        logging.debug(f"GEE IMERG fetch fallback: {e}")

    return None


def fetch_gee_gfs_forecast_rain(lat: float, lon: float, ref_dt: datetime, horizon_days: int) -> Optional[float]:
    """
    Mengambil prakiraan curah hujan (mm) untuk horizon T+1 (24 jam), T+3 (72 jam), atau T+7 (168 jam)
    langsung dari model cuaca global NOAA/GFS0P25 di Google Earth Engine.
    """
    if not init_gee():
        return None

    try:
        point = ee.Geometry.Point([lon, lat])
        gfs_start = (ref_dt - timedelta(days=2)).strftime("%Y-%m-%d")
        gfs_end = (ref_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        max_hours = int(horizon_days * 24)

        collection = (
            ee.ImageCollection("NOAA/GFS0P25")
            .filterDate(gfs_start, gfs_end)
            .filter(ee.Filter.gt("forecast_hours", 0))
            .filter(ee.Filter.lte("forecast_hours", max_hours))
            .select("precipitation_rate")
        )
        mean_rate_img = collection.mean().multiply(86400.0 * horizon_days)
        val = (
            mean_rate_img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=25000,
            )
            .get("precipitation_rate")
            .getInfo()
        )
        if val is not None:
            return max(float(val), 0.0)
    except Exception as e:
        logging.debug(f"GEE NOAA/GFS0P25 fetch fallback: {e}")

    return None


def fetch_gee_rainfall_sum(lat: float, lon: float, start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Mengambil curah hujan dari Google Earth Engine:
    - Mencoba CHIRPS Daily terlebih dahulu untuk data historis.
    - Otomatis beralih ke NASA GPM IMERG V07 untuk tanggal near-real-time.
    """
    chirps_val = fetch_gee_chirps_sum(lat, lon, start_date, end_date)
    if chirps_val is not None and chirps_val > 0:
        return {
            "rain_mm": round(chirps_val, 2),
            "dataset": "Google Earth Engine (UCSB-CHG/CHIRPS/DAILY)",
        }

    imerg_val = fetch_gee_imerg_sum(lat, lon, start_date, end_date)
    if imerg_val is not None:
        return {
            "rain_mm": round(imerg_val, 2),
            "dataset": "Google Earth Engine (NASA/GPM_L3/IMERG_V07)",
        }

    return {
        "rain_mm": round(chirps_val or 0.0, 2),
        "dataset": "Google Earth Engine (UCSB-CHG/CHIRPS/DAILY)",
    }


def fetch_gee_hydro_window(lat: float, lon: float, ref_dt: datetime, horizon_days: int) -> Dict[str, Any]:
    """
    Mengambil paket lengkap curah hujan anteseden 7 hari (IMERG/CHIRPS) dan
    curah hujan prakiraan horizon T+1/T+3/T+7 (NOAA/GFS0P25 atau CHIRPS/IMERG historis)
    murni dari Google Earth Engine sebagai pengganti input dinamis GISTDA.
    """
    start_7d = (ref_dt - timedelta(days=7)).strftime("%Y-%m-%d")
    ref_str = ref_dt.strftime("%Y-%m-%d")
    end_horizon = (ref_dt + timedelta(days=horizon_days)).strftime("%Y-%m-%d")

    past_res = fetch_gee_rainfall_sum(lat, lon, start_7d, ref_str)
    past_7d_mm = past_res["rain_mm"]
    dataset_used = past_res["dataset"]

    # Untuk tanggal historis (> 3 hari lalu), ambil realisasi hujan pada jendela horizon dari CHIRPS/IMERG;
    # Untuk tanggal berjalan/live, ambil prakiraan dari NOAA/GFS0P25 di GEE.
    now_dt = datetime.utcnow()
    is_historical = (ref_dt + timedelta(days=horizon_days)) < (now_dt - timedelta(days=3))

    horizon_mm: Optional[float] = None
    if is_historical:
        hor_res = fetch_gee_rainfall_sum(lat, lon, ref_str, end_horizon)
        horizon_mm = hor_res["rain_mm"]
        forecast_tag = f"{dataset_used} + Horizon Window"
    else:
        horizon_mm = fetch_gee_gfs_forecast_rain(lat, lon, ref_dt, horizon_days)
        forecast_tag = "Google Earth Engine (NASA GPM IMERG + NOAA/GFS0P25 Forecast)"

    if horizon_mm is None:
        horizon_mm = 0.0

    return {
        "past_7d_rain_mm": round(past_7d_mm, 2),
        "horizon_rain_mm": round(horizon_mm, 2),
        "weather_source": forecast_tag,
        "gee_active": is_gee_ready(),
    }


def fetch_gee_point_spatiotemporal_features(
    df: pd.DataFrame,
    prov_code: str,
    ref_dt: datetime,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Mengekstrak fitur spasiotemporal dinamis sesuai dokumentasi peneliti:
    1. Poligon banjir satelit resmi GISTDA Open API v2.0 (/features/flood/1day, /3days, /7days, /30days)
       menggunakan GISDA_API_KEY pada tanggal observasi berjalan.
    2. Observasi satelit Sentinel-1 SAR (COPERNICUS/S1_GRD) + JRC PWB Mask dari Google Earth Engine
       untuk jendela 365 hari penuh serta curah hujan CHIRPS/IMERG untuk Sukhothai & Suphan Buri.
    3. Curah hujan stasiun lapangan aktual dari HII OpenData (daily_rain CSV per Nearest_Station_ID)
       untuk Nakhon Pathom (Rain_Sum_365d).
    """
    from app.services.gistda_service import (
        fetch_gistda_live_flood_layers,
        fetch_hii_station_rainfall,
        is_gistda_configured,
    )

    if not init_gee():
        raise RuntimeError("Google Earth Engine is not initialized; cannot fetch satellite observations.")

    obs_date_str = ref_dt.strftime("%Y-%m-%d")
    end_inclusive = (ref_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    cache_key = f"{prov_code}:{obs_date_str}:{is_gistda_configured()}"
    df_out = df.copy()

    if cache_key in _GEE_LAG_CACHE:
        cached_cols, cached_meteo = _GEE_LAG_CACHE[cache_key]
        for col, vals in cached_cols.items():
            df_out[col] = vals
        return df_out, dict(cached_meteo)

    feats = [
        ee.Feature(ee.Geometry.Point([float(r["X"]), float(r["Y"])]), {"pid": int(i)})
        for i, r in df_out.iterrows()
    ]
    fc = ee.FeatureCollection(feats)

    start_365 = (ref_dt - timedelta(days=364)).strftime("%Y-%m-%d")
    start_7 = (ref_dt - timedelta(days=6)).strftime("%Y-%m-%d")

    pwb_mask = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0).lte(80)
    slope_mask = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003")).lt(10)

    s1_base = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(fc.geometry().bounds())
        .filterDate(start_365, end_inclusive)
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .select("VV")
    )

    def _make_flood_band(days: int, band_name: str, end_bound: str = end_inclusive) -> ee.Image:
        s_d = (ref_dt - timedelta(days=days - 1)).strftime("%Y-%m-%d")
        col = s1_base.filterDate(s_d, end_bound)

        def _to_flood(img):
            return img.lt(-16.0).And(pwb_mask).And(slope_mask).rename(band_name).toFloat()

        zero_img = ee.Image.constant(0).rename(band_name).toFloat()
        return ee.ImageCollection([zero_img]).merge(col.map(_to_flood)).sum().unmask(0).rename(band_name)

    def _make_rain_band(collection_id: str, start_str: str, scale_factor: float, band_name: str) -> ee.Image:
        col = (
            ee.ImageCollection(collection_id)
            .filterDate(start_str, end_inclusive)
            .select("precipitation")
            .map(lambda img: img.multiply(scale_factor).rename(band_name).toFloat())
        )
        zero_img = ee.Image.constant(0).rename(band_name).toFloat()
        return ee.ImageCollection([zero_img]).merge(col).sum().unmask(0).rename(band_name)

    # Cek apakah tanggal observasi berada dalam jendela live GISTDA Open API v2.0 (<= 5 hari dari hari ini)
    is_live_window = abs((datetime.utcnow() - ref_dt).days) <= 5
    gistda_res = (
        fetch_gistda_live_flood_layers(df_out, prov_code, obs_date_str)
        if is_live_window
        else {"active": False}
    )

    # Jika GISTDA live aktif, jendela 30 hari terakhir diambil dari poligon resmi GISTDA,
    # dan GEE menghitung hari ke-31 s.d. 365 agar tidak terjadi perhitungan ganda (double counting).
    end_365_gee = (ref_dt - timedelta(days=29)).strftime("%Y-%m-%d") if gistda_res.get("active") else end_inclusive

    stack = ee.Image.cat([
        _make_flood_band(1, "f1"),
        _make_flood_band(3, "f3"),
        _make_flood_band(7, "f7"),
        _make_flood_band(14, "f14"),
        _make_flood_band(30, "f30"),
        _make_flood_band(365, "f365", end_bound=end_365_gee),
        _make_rain_band("UCSB-CHG/CHIRPS/DAILY", start_365, 1.0, "r365"),
        _make_rain_band("NASA/GPM_L3/IMERG_V07", start_7, 0.5, "r7"),
    ])

    res = stack.reduceRegions(collection=fc, reducer=ee.Reducer.first(), scale=250).getInfo()
    rows = {f["properties"]["pid"]: f["properties"] for f in res.get("features", [])}

    n_pts = len(df_out)
    gee_f1 = [min(float(rows.get(i, {}).get("f1", 0) or 0), 1.0) for i in range(n_pts)]
    gee_f3 = [float(rows.get(i, {}).get("f3", 0) or 0) for i in range(n_pts)]
    gee_f7 = [float(rows.get(i, {}).get("f7", 0) or 0) for i in range(n_pts)]
    gee_f14 = [float(rows.get(i, {}).get("f14", 0) or 0) for i in range(n_pts)]
    gee_f30 = [float(rows.get(i, {}).get("f30", 0) or 0) for i in range(n_pts)]
    gee_f365 = [float(rows.get(i, {}).get("f365", 0) or 0) for i in range(n_pts)]

    if gistda_res.get("active"):
        g1 = gistda_res["f1"].tolist()
        g3 = gistda_res["f3"].tolist()
        g7 = gistda_res["f7"].tolist()
        g30 = gistda_res["f30"].tolist()
        cached_cols: Dict[str, list] = {
            "Flood_Lag_1d": [min(max(g1[i], gee_f1[i]), 1.0) for i in range(n_pts)],
            "Flood_Sum_3d": [max(g3[i], gee_f3[i]) for i in range(n_pts)],
            "Flood_Sum_7d": [max(g7[i], gee_f7[i]) for i in range(n_pts)],
            "Flood_Sum_14d": [max(g30[i], gee_f14[i]) for i in range(n_pts)],
            "Flood_Sum_30d": [max(g30[i], gee_f30[i]) for i in range(n_pts)],
            "Flood_Sum_365d": [g30[i] + gee_f365[i] for i in range(n_pts)],
        }
    else:
        cached_cols = {
            "Flood_Lag_1d": gee_f1,
            "Flood_Sum_3d": gee_f3,
            "Flood_Sum_7d": gee_f7,
            "Flood_Sum_14d": gee_f14,
            "Flood_Sum_30d": gee_f30,
            "Flood_Sum_365d": gee_f365,
        }

    r7_vals = [float(rows.get(i, {}).get("r7", 0) or 0) for i in range(n_pts)]
    r365_vals = [float(rows.get(i, {}).get("r365", 0) or 0) for i in range(n_pts)]

    hii_active_stns = None
    if prov_code == "npt" and "Nearest_Station_ID" in df_out.columns:
        unique_stns = [str(s) for s in df_out["Nearest_Station_ID"].dropna().unique().tolist()]
        hii_r365_map, hii_r7_map, active_stns = fetch_hii_station_rainfall(unique_stns, ref_dt)
        hii_active_stns = f"{active_stns}/{len(unique_stns)}"
        if active_stns > 0:
            r365_vals = [float(hii_r365_map.get(str(s), 0.0)) for s in df_out["Nearest_Station_ID"]]
            hii_r7_list = [float(hii_r7_map.get(str(s), 0.0)) for s in df_out["Nearest_Station_ID"]]
            if sum(hii_r7_list) > 0:
                r7_vals = hii_r7_list
        cached_cols["Rain_Sum_365d"] = r365_vals

    for col, vals in cached_cols.items():
        df_out[col] = vals

    mean_r7 = round(sum(r7_vals) / max(n_pts, 1), 2)
    mean_r365 = round(sum(r365_vals) / max(n_pts, 1), 2)
    flood_points_365d = int(sum(1 for v in cached_cols["Flood_Sum_365d"] if v > 0))

    if gistda_res.get("active"):
        src_label = (
            "GISTDA Open API v2.0 + HII OpenData Stations + GEE S1 SAR"
            if prov_code == "npt"
            else "GISTDA Open API v2.0 (1d/3d/7d/30d) + GEE (S1 SAR 365d & IMERG)"
        )
    else:
        src_label = (
            "HII OpenData Stations (daily_rain) + GEE S1 SAR"
            if prov_code == "npt"
            else "Google Earth Engine (Sentinel-1 SAR + JRC PWB Mask + CHIRPS/IMERG)"
        )

    meteo_summary = {
        "past_7d_rain_mm": mean_r7,
        "rain_365d_mm": mean_r365,
        "flood_points_365d": flood_points_365d,
        "gistda_active": bool(gistda_res.get("active")),
        "gistda_30d_points": gistda_res.get("flooded_pts_30d", 0) if gistda_res.get("active") else None,
        "gistda_7d_points": gistda_res.get("flooded_pts_7d", 0) if gistda_res.get("active") else None,
        "hii_stations_coverage": hii_active_stns,
        "weather_source": src_label,
        "gee_active": is_gee_ready(),
    }

    _GEE_LAG_CACHE[cache_key] = (cached_cols, dict(meteo_summary))
    return df_out, meteo_summary
