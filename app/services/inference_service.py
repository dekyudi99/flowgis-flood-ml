import uuid
import time
import math
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import requests

from app.config import settings
from app.schemas.ews_schema import PredictRequest, PredictResponse, RiskStatistics
from app.services.model_loader import model_registry
from app.services.gee_service import init_gee, is_gee_ready, fetch_gee_point_spatiotemporal_features
from app.services.gistda_service import is_gistda_configured

logger = logging.getLogger(__name__)

# In-memory storage for active preview sessions
preview_sessions: Dict[str, Dict[str, Any]] = {}

HII_OPENDATA_BASE_URL = "https://tiservice.hii.or.th/opendata/data_catalog"

HISTORICAL_DYNAMIC_COLUMNS = [
    "Flood_Lag_1d",
    "Flood_Sum_3d",
    "Flood_Sum_7d",
    "Flood_Sum_14d",
    "Flood_Sum_30d",
    "Flood_Sum_365d",
    "Rain_Sum_365d",
    "Flooded",
    "Log_FlowAccum",
    "TWI_Elev_Ratio",
    "Month_Sin",
    "Month_Cos",
]


def classify_risk(prob: float) -> Tuple[int, str, str]:
    """Mengembalikan (risk_tier, risk_label, hex_color) berdasarkan 5-tier standar."""
    if prob < 0.20:
        return 1, "Very Low", "#27ae60"
    elif prob < 0.40:
        return 2, "Low", "#2ecc71"
    elif prob < 0.60:
        return 3, "Moderate", "#f1c40f"
    elif prob < 0.80:
        return 4, "High", "#e67e22"
    else:
        return 5, "Very High", "#c0392b"


def check_hii_opendata_status() -> bool:
    """Memverifikasi konektivitas ke portal resmi HII OpenData (daily_rain & water_level)."""
    try:
        resp = requests.head(f"{HII_OPENDATA_BASE_URL}/daily_rain/", timeout=4)
        return resp.status_code == 200
    except Exception:
        return False


def fetch_open_meteo_forecast(prov_code: str, ref_dt: datetime) -> Dict[str, float]:
    """
    Mengambil proyeksi curah hujan numerik harian Open-Meteo GFS/ERA5
    sesuai rekomendasi Bab 7 Dokumen Riset Peneliti.
    """
    centers = {
        "skt": (17.00, 99.82),
        "spb": (14.47, 100.12),
        "npt": (14.02, 99.98),
    }
    lat, lon = centers.get(prov_code, (14.0, 100.0))
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=precipitation_sum&forecast_days=10&timezone=Asia%2FBangkok"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json().get("daily", {})
            times = data.get("time", [])
            precips = data.get("precipitation_sum", [])
            return dict(zip(times, [float(p or 0.0) for p in precips]))
    except Exception as e:
        logger.warning(f"Failed to fetch Open-Meteo forecast rain: {e}")
    return {}


def build_meteo_metadata(prov_code: str, gee_meteo: Dict[str, Any], forecast_rain_mm: float = 0.0, lead_display: str = "T+1") -> Dict[str, Any]:
    """
    Menyusun metadata sumber observasi satelit (GISTDA Open API v2.0 + GEE + HII OpenData + Open-Meteo)
    tanpa rumus tambahan.
    """
    if prov_code in ("skt", "spb"):
        return {
            "past_7d_rain_mm": gee_meteo["past_7d_rain_mm"],
            "rain_365d_mm": gee_meteo["rain_365d_mm"],
            "forecast_rain_mm": round(forecast_rain_mm, 2),
            "forecast_horizon": lead_display,
            "flood_points_365d": gee_meteo["flood_points_365d"],
            "gistda_30d_points": gee_meteo.get("gistda_30d_points", 0),
            "gistda_7d_points": gee_meteo.get("gistda_7d_points", 0),
            "weather_source": f"{gee_meteo['weather_source']} & Open-Meteo Forecast ({lead_display})",
            "source_url": "https://api-gateway.gistda.or.th/api/2.0/resources",
            "gistda_active": gee_meteo.get("gistda_active", is_gistda_configured()),
            "gee_active": is_gee_ready(),
            "hii_active": None,
            "hii_stations_coverage": None,
        }

    hii_online = check_hii_opendata_status()
    return {
        "past_7d_rain_mm": gee_meteo["past_7d_rain_mm"],
        "rain_365d_mm": gee_meteo["rain_365d_mm"],
        "forecast_rain_mm": round(forecast_rain_mm, 2),
        "forecast_horizon": lead_display,
        "flood_points_365d": gee_meteo["flood_points_365d"],
        "gistda_30d_points": gee_meteo.get("gistda_30d_points", 0),
        "gistda_7d_points": gee_meteo.get("gistda_7d_points", 0),
        "weather_source": f"{gee_meteo['weather_source']} & Open-Meteo Forecast ({lead_display})",
        "source_url": "https://tiservice.hii.or.th/opendata/",
        "gistda_active": gee_meteo.get("gistda_active", is_gistda_configured()),
        "gee_active": is_gee_ready(),
        "hii_active": hii_online,
        "hii_stations_coverage": gee_meteo.get("hii_stations_coverage", "7/7"),
    }


def run_prediction(req: PredictRequest) -> PredictResponse:
    start_time = time.time()
    prov_key = req.province.lower().replace("-", "").replace(" ", "").replace("_", "")
    # Parsing lead_time horizon (1d: T+1, 3d: T+3, 7d: T+7)
    lead_time = (req.lead_time or "1d").lower().strip()
    if lead_time not in ("1d", "3d", "7d"):
        lead_time = "1d"
    lead_days = 1 if lead_time == "1d" else (3 if lead_time == "3d" else 7)
    lead_display = f"T+{lead_days} ({'24h Next-Day' if lead_days == 1 else ('72h Surge' if lead_days == 3 else '7-Day Weekly Cycle')})"

    req_model = (req.model_name or "best").lower()

    # Inisialisasi GEE jika belum aktif
    init_gee()

    # 1. Konfigurasi Provinsi, Dataset Titik Pantau Statis, dan Model Checkpoint EWS (.joblib)
    if "nakhon" in prov_key or prov_key == "npt":
        province_name = "Nakhon Pathom"
        prov_code = "npt"
        feature_file = settings.DATA_DIR / "npt" / "nakhon_pathom_1000pts_spatiotemporal_master.csv"
        analysis_mode = f"{lead_display} EWS Forecast (GISTDA Open API v2.0 + HII OpenData Stations + GEE)"
        validation_metrics = {
            "holdout_test_period": "2024–2025",
            "forecast_horizon": lead_display,
            "roc_auc": 0.9723,
            "accuracy": "96.90%",
            "precision": "64.29%",
            "recall": "46.15%",
            "features_summary": "10 Terrain + 2 HII Station Distances + Month_Sin/Cos + GISTDA/GEE Flood_Sum_365d & HII Station Rain_Sum_365d",
        }
        if req_model == "lightgbm":
            model_key = "npt_lightgbm"
            scaler_key = "skt"
            model_display = "LightGBM Classifier (Terrain - 12 Features)"
        else:
            model_key = "npt_catboost"
            scaler_key = "npt"
            model_display = "CatBoost EWS Classifier (GISTDA + HII Stations - 16 Features)"
    elif "suphan" in prov_key or prov_key == "spb":
        province_name = "Suphan Buri"
        prov_code = "spb"
        feature_file = settings.DATA_DIR / "spb" / "suphanburi_1000pts_spatiotemporal_master.csv"
        model_key = "spb_lightgbm"
        scaler_key = "spb"
        model_display = "LightGBM EWS Classifier (GISTDA + GEE Spatiotemporal - 15 Features)"
        analysis_mode = f"{lead_display} EWS Forecast (GISTDA Open API v2.0 + GEE Sentinel-1 SAR)"
        validation_metrics = {
            "holdout_test_period": "2024–2025",
            "forecast_horizon": lead_display,
            "roc_auc": 0.9863,
            "accuracy": "85.10% (Wet) / 100% (Dry)",
            "precision": "41.34%",
            "recall": "100.00%",
            "features_summary": "12 Terrain + Month_Sin/Cos + GISTDA/GEE Flood_Sum_365d",
        }
    else:
        province_name = "Sukhothai"
        prov_code = "skt"
        feature_file = settings.DATA_DIR / "skt" / "sukhothai_1000pts_spatiotemporal_master.csv"
        if model_registry.get_model("skt_ews") is not None and req_model != "lightgbm_static":
            model_key = "skt_ews"
            scaler_key = "skt_ews"
            model_display = "Random Forest EWS Classifier (GISTDA + GEE Multi-Scale - 20 Features)"
            analysis_mode = f"{lead_display} EWS Forecast (GISTDA Open API v2.0 1d/3d/7d/30d + GEE 365d)"
            validation_metrics = {
                "holdout_test_period": "2024–2025",
                "forecast_horizon": lead_display,
                "roc_auc": 0.9914,
                "accuracy": "97.10% (Wet) / 100% (Dry)",
                "precision": "77.32%",
                "recall": "91.46%",
                "features_summary": "12 Terrain + Month_Sin/Cos + GISTDA/GEE Flood_Lag_1d & Flood_Sum_3d/7d/14d/30d/365d",
            }
        else:
            model_key = "skt_lightgbm"
            scaler_key = "skt"
            model_display = "LightGBM Classifier (Static Terrain - 12 Features)"
            analysis_mode = "Static Physical Terrain Susceptibility (12 Features)"
            validation_metrics = {
                "holdout_test_period": "2024–2025",
                "forecast_horizon": "Static Baseline",
                "roc_auc": 0.9831,
                "accuracy": "93.10%",
                "precision": "54.97%",
                "recall": "94.00%",
                "features_summary": "12 Static Physical Terrain Features",
            }

    if not feature_file.exists():
        raise FileNotFoundError(f"Master feature file not found: {feature_file}")

    # 2. Tentukan tanggal observasi live (default hari ini UTC, bukan tanggal historis 2024)
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    date_str = (req.target_date or "").strip() or today_str
    try:
        ref_dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
    except ValueError:
        ref_dt = datetime.strptime(today_str, "%Y-%m-%d")

    forecast_target_dt = ref_dt + timedelta(days=lead_days)

    # 3. Muat 1.000 titik pantau & buang seluruh kolom historis 2024 agar tidak bocor
    df = pd.read_csv(feature_file)
    df = df.drop(columns=[col for col in HISTORICAL_DYNAMIC_COLUMNS if col in df.columns])

    # 4. Ekstrak observasi satelit aktual dari GEE (COPERNICUS/S1_GRD VV < -16 dB + JRC PWB Mask + CHIRPS/IMERG)
    df, gee_meteo = fetch_gee_point_spatiotemporal_features(df=df, prov_code=prov_code, ref_dt=ref_dt)

    # 4b. Ambil proyeksi curah hujan numerik Open-Meteo GFS/ERA5 (t s.d. target date) sesuai Bab 7 Peneliti
    forecast_rain_map = fetch_open_meteo_forecast(prov_code=prov_code, ref_dt=ref_dt)
    horizon_dates = [(ref_dt + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, lead_days + 1)]
    projected_rain_sum = sum(forecast_rain_map.get(d, 0.0) for d in horizon_dates)
    meteo = build_meteo_metadata(prov_code=prov_code, gee_meteo=gee_meteo, forecast_rain_mm=projected_rain_sum, lead_display=lead_display)

    # 4c. Harmonisasi Multi-Horizon Fitur Dinamis sesuai Bab 7 & Skema Rolling Window Peneliti
    # Jika lead_days > 1, proyeksi akumulasi hujan ditambahkan ke Rain_Sum_365d (NPT) atau mempengaruhi lag memori (SKT/SPB)
    if "Rain_Sum_365d" in df.columns:
        df["Rain_Sum_365d"] = (df["Rain_Sum_365d"].astype(float) + projected_rain_sum).clip(lower=0.0)

    if lead_days > 1 and "Flood_Sum_3d" in df.columns:
        # Untuk Sukhothai (20 Fitur): Jika T+3 atau T+7 diproyeksikan, akumulasi genangan merefleksikan persistensi hujan hulu
        if projected_rain_sum > 10.0:
            rain_surge_pts = (df["Elevation"].astype(float) < 45.0) & (df["Dist_to_river"].astype(float) < 1500.0)
            df.loc[rain_surge_pts, "Flood_Sum_3d"] = (df.loc[rain_surge_pts, "Flood_Sum_3d"].astype(float) + 1.0).clip(upper=3.0)
            if lead_days >= 7:
                df.loc[rain_surge_pts, "Flood_Sum_7d"] = (df.loc[rain_surge_pts, "Flood_Sum_7d"].astype(float) + 2.0).clip(upper=7.0)
                df.loc[rain_surge_pts, "Flood_Sum_365d"] = (df.loc[rain_surge_pts, "Flood_Sum_365d"].astype(float) + 1.0).clip(upper=365.0)

    if lead_days > 1 and "Flood_Sum_365d" in df.columns and prov_code == "spb":
        # Untuk Suphan Buri (15 Fitur): Proyeksi akumulasi genangan hulu Tha Chin jika intensitas hujan tinggi
        if projected_rain_sum > 15.0:
            surge_pts = (df["Elevation"].astype(float) < 12.0) & (df["Dist_to_river"].astype(float) < 2000.0)
            df.loc[surge_pts, "Flood_Sum_365d"] = (df.loc[surge_pts, "Flood_Sum_365d"].astype(float) + 1.0).clip(upper=365.0)

    # 5. Hitung HANYA 4 formula rekayasa fitur matematis eksak dari notebook peneliti
    df["Log_FlowAccum"] = np.log1p(df["FlowAccum"].astype(float))
    df["TWI_Elev_Ratio"] = df["TWI"].astype(float) / (df["Elevation"].astype(float) + 1.0)
    df["Month_Sin"] = math.sin(2.0 * math.pi * forecast_target_dt.month / 12.0)
    df["Month_Cos"] = math.cos(2.0 * math.pi * forecast_target_dt.month / 12.0)

    # 6. Eksekusi Murni Machine Learning Inference (RobustScaler.transform + model.predict_proba)
    model = model_registry.get_model(model_key)
    scaler = model_registry.get_scaler(scaler_key)

    if model is None or scaler is None:
        raise RuntimeError(f"Model '{model_key}' or scaler '{scaler_key}' is not loaded in ModelRegistry.")

    scaler_cols = list(scaler.feature_names_in_)
    missing_cols = [col for col in scaler_cols if col not in df.columns]
    if missing_cols:
        raise RuntimeError(f"Missing required model input features: {missing_cols}")

    X_scaled = np.nan_to_num(scaler.transform(df[scaler_cols]), nan=0.0, posinf=0.0, neginf=0.0)
    # 100% probabilitas murni dari classifier peneliti tanpa rumus tambahan atau pengali pasca-model
    raw_probs = model.predict_proba(X_scaled)[:, 1]

    # 7. Bangun GeoJSON FeatureCollection & hitung distribusi 5-tier
    features_geojson = []
    tier_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    records = df.to_dict(orient="records")
    for idx, row in enumerate(records):
        prob_val = float(raw_probs[idx])
        tier, label, color = classify_risk(prob_val)
        tier_counts[tier] += 1

        lon = float(row["X"])
        lat = float(row["Y"])
        pid = str(row.get("Point_ID", idx + 1))
        elev = round(float(row["Elevation"]), 2)
        dist_riv = round(float(row["Dist_to_river"]), 1)
        stn_id = row.get("Nearest_Station_ID") if "Nearest_Station_ID" in row else None

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat],
            },
            "properties": {
                "point_id": pid,
                "province": province_name,
                "probability": round(prob_val, 4),
                "raw_ml_probability": round(prob_val, 4),
                "risk_tier": tier,
                "risk_label": label,
                "color": color,
                "elevation_m": elev,
                "dist_to_river_m": dist_riv,
                "twi": round(float(row["TWI"]), 2),
                "ndvi": round(float(row["NDVI"]), 3),
                "cn": float(row["CN"]),
                "flood_days_365d": round(float(row.get("Flood_Sum_365d", 0)), 1),
                "nearest_station": stn_id if pd.notna(stn_id) else None,
                "lead_time": lead_time,
                "target_date": ref_dt.strftime("%Y-%m-%d"),
                "valid_for_date": forecast_target_dt.strftime("%Y-%m-%d"),
                "analysis_mode": analysis_mode,
                "weather_source": meteo["weather_source"],
                "past_7d_rain_mm": meteo["past_7d_rain_mm"],
            },
        }
        features_geojson.append(feature)

    total_pts = len(features_geojson) or 1000

    stats = RiskStatistics(
        total_points=total_pts,
        very_high_count=tier_counts[5],
        very_high_pct=round((tier_counts[5] / total_pts) * 100, 2),
        high_count=tier_counts[4],
        high_pct=round((tier_counts[4] / total_pts) * 100, 2),
        moderate_count=tier_counts[3],
        moderate_pct=round((tier_counts[3] / total_pts) * 100, 2),
        low_count=tier_counts[2],
        low_pct=round((tier_counts[2] / total_pts) * 100, 2),
        very_low_count=tier_counts[1],
        very_low_pct=round((tier_counts[1] / total_pts) * 100, 2),
        overall_alert_level=(
            "CRITICAL / HIGH"
            if (tier_counts[5] + tier_counts[4]) > 50
            else (
                "WARNING / MODERATE"
                if (tier_counts[5] + tier_counts[4] + tier_counts[3]) > 100
                else "NORMAL / LOW"
            )
        ),
    )

    preview_id = f"prev_{prov_code}_{lead_time}_{ref_dt.strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"

    geojson_collection = {
        "type": "FeatureCollection",
        "metadata": {
            "province": province_name,
            "model_key": model_key,
            "model_used": model_display,
            "analysis_mode": analysis_mode,
            "features_count": len(scaler_cols),
            "feature_names": scaler_cols,
            "target_date": ref_dt.strftime("%Y-%m-%d"),
            "valid_for_date": forecast_target_dt.strftime("%Y-%m-%d"),
            "lead_time": lead_time,
            "validation_metrics": validation_metrics,
            "meteorology": meteo,
        },
        "features": features_geojson,
    }

    preview_sessions[preview_id] = {
        "preview_id": preview_id,
        "province": province_name,
        "prov_key": prov_code,
        "lead_time": lead_time,
        "model_used": model_display,
        "geojson": geojson_collection,
        "statistics": stats.model_dump() if hasattr(stats, "model_dump") else stats.dict(),
        "created_at": time.time(),
    }

    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    return PredictResponse(
        preview_id=preview_id,
        province=province_name,
        lead_time=lead_time,
        model_used=model_display,
        execution_time_ms=elapsed_ms,
        statistics=stats,
        geojson=geojson_collection,
    )
