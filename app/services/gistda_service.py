import os
import logging
import requests
import numpy as np
import pandas as pd
from io import StringIO
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
from matplotlib.path import Path
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

from app.config import settings

load_dotenv()

logger = logging.getLogger(__name__)

GISTDA_GATEWAY_BASE = "https://api-gateway.gistda.or.th/api/2.0/resources"
HII_DAILY_RAIN_BASE = "https://tiservice.hii.or.th/opendata/data_catalog/daily_rain"

PROVINCE_PV_IDN = {
    "skt": 64,
    "spb": 72,
    "npt": 73,
}

_GISTDA_CACHE: Dict[str, Dict[str, Any]] = {}
_HII_RAIN_CACHE: Dict[str, Tuple[Dict[str, float], Dict[str, float], int]] = {}


def _get_gistda_api_key() -> str:
    return (
        os.getenv("GISDA_API_KEY")
        or os.getenv("GISTDA_API_KEY")
        or settings.GISDA_API_KEY
        or ""
    ).strip()


def is_gistda_configured() -> bool:
    return bool(_get_gistda_api_key())


def _fetch_gistda_endpoint_features(endpoint_suffix: str, pv_idn: int, api_key: str) -> List[Dict[str, Any]]:
    url = f"{GISTDA_GATEWAY_BASE}/features/flood/{endpoint_suffix}"
    headers = {"API-Key": api_key}
    try:
        r0 = requests.get(
            url,
            headers=headers,
            params={"pv_idn": pv_idn, "limit": 5000, "offset": 0},
            timeout=20,
        )
        if r0.status_code != 200:
            return []
        js0 = r0.json()
        feats = list(js0.get("features", []))
        total = int(js0.get("numberMatched", len(feats)))
        if total > 5000:
            offsets = list(range(5000, min(total, 20000), 5000))

            def _fetch_page(off: int) -> List[Dict[str, Any]]:
                try:
                    resp = requests.get(
                        url,
                        headers=headers,
                        params={"pv_idn": pv_idn, "limit": 5000, "offset": off},
                        timeout=20,
                    )
                    if resp.status_code == 200:
                        return resp.json().get("features", [])
                except Exception:
                    pass
                return []

            with ThreadPoolExecutor(max_workers=4) as ex:
                for batch in ex.map(_fetch_page, offsets):
                    feats.extend(batch)
        return feats
    except Exception as e:
        logger.warning(f"GISTDA API fetch warning ({endpoint_suffix}, pv_idn={pv_idn}): {type(e).__name__}")
        return []


def _points_in_gistda_polygons(pts_xy: np.ndarray, feats: List[Dict[str, Any]]) -> np.ndarray:
    """
    Uji perpotongan spasial (spatial intersection) antara 1.000 titik koordinat (X, Y)
    terhadap poligon banjir GISTDA sesuai metodologi peneliti (PROJECT_DOCUMENTATION_ID.md Bagian 2.1).
    """
    hit = np.zeros(len(pts_xy), dtype=float)
    if not feats or len(pts_xy) == 0:
        return hit

    xmin_all, ymin_all = pts_xy.min(axis=0)
    xmax_all, ymax_all = pts_xy.max(axis=0)

    for f in feats:
        geom = f.get("geometry")
        if not geom:
            continue
        gtype = geom.get("type")
        coords = geom.get("coordinates", [])
        polys = coords if gtype == "MultiPolygon" else [coords]
        for poly in polys:
            if not poly or not poly[0]:
                continue
            ring = np.asarray(poly[0], dtype=float)
            if ring.ndim != 2 or ring.shape[1] < 2:
                continue
            rx0, ry0 = ring[:, 0].min(), ring[:, 1].min()
            rx1, ry1 = ring[:, 0].max(), ring[:, 1].max()
            if rx1 < xmin_all or rx0 > xmax_all or ry1 < ymin_all or ry0 > ymax_all:
                continue
            cand = np.where(
                (pts_xy[:, 0] >= rx0)
                & (pts_xy[:, 0] <= rx1)
                & (pts_xy[:, 1] >= ry0)
                & (pts_xy[:, 1] <= ry1)
            )[0]
            if len(cand) > 0:
                inside = Path(ring[:, :2]).contains_points(pts_xy[cand])
                hit[cand[inside]] = 1.0
    return hit


def fetch_gistda_live_flood_layers(df: pd.DataFrame, prov_code: str, date_key: str) -> Dict[str, Any]:
    """
    Mengambil data poligon genangan banjir resmi dari GISTDA Open API v2.0
    (/features/flood/1day, /3days, /7days, /30days) menggunakan GISDA_API_KEY.
    """
    api_key = _get_gistda_api_key()
    if not api_key:
        return {"active": False}

    pv_idn = PROVINCE_PV_IDN.get(prov_code, 64)
    cache_key = f"{prov_code}:{pv_idn}:{date_key}"
    if cache_key in _GISTDA_CACHE:
        return _GISTDA_CACHE[cache_key]

    pts_xy = df[["X", "Y"]].values.astype(float)

    # Untuk Sukhothai (20 fitur), ambil keempat jendela GISTDA (1day, 3days, 7days, 30days);
    # Untuk Suphan Buri (15 fitur) & Nakhon Pathom (16 fitur), ambil juga keempat jendela agar
    # akumulasi hari genangan 30 hari terakhir terekam presisi.
    endpoints = ["1day", "3days", "7days", "30days"]
    with ThreadPoolExecutor(max_workers=4) as ex:
        feats_list = list(
            ex.map(lambda ep: _fetch_gistda_endpoint_features(ep, pv_idn, api_key), endpoints)
        )

    f1_pts = _points_in_gistda_polygons(pts_xy, feats_list[0])
    f3_pts = _points_in_gistda_polygons(pts_xy, feats_list[1])
    f7_pts = _points_in_gistda_polygons(pts_xy, feats_list[2])
    f30_pts = _points_in_gistda_polygons(pts_xy, feats_list[3])

    poly_counts = {ep: len(fl) for ep, fl in zip(endpoints, feats_list)}
    result = {
        "active": True,
        "pv_idn": pv_idn,
        "polygon_counts": poly_counts,
        "f1": f1_pts,
        "f3": f1_pts + f3_pts,
        "f7": f1_pts + f3_pts + f7_pts,
        "f30": f1_pts + f3_pts + f7_pts + f30_pts,
        "flooded_pts_30d": int((f30_pts > 0).sum()),
        "flooded_pts_7d": int((f7_pts > 0).sum()),
        "flooded_pts_1d": int((f1_pts > 0).sum()),
    }
    _GISTDA_CACHE[cache_key] = result
    return result


def fetch_hii_station_rainfall(
    station_ids: List[str],
    ref_dt: datetime,
) -> Tuple[Dict[str, float], Dict[str, float], int]:
    """
    Mengambil data curah hujan harian aktual per stasiun dari portal HII OpenData
    (https://tiservice.hii.or.th/opendata/data_catalog/daily_rain/{YYYY}/{YYYYMM}/{STATION}.csv)
    untuk jendela lookback 365 hari dan 7 hari hingga ref_dt (t-1), persis sesuai notebook peneliti.
    """
    end_inclusive = (ref_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    start_365 = (ref_dt - timedelta(days=364)).strftime("%Y-%m-%d")
    start_7 = (ref_dt - timedelta(days=6)).strftime("%Y-%m-%d")
    cache_key = f"{end_inclusive}:{','.join(sorted(station_ids))}"

    if cache_key in _HII_RAIN_CACHE:
        return _HII_RAIN_CACHE[cache_key]

    yms = sorted(
        list(
            {
                f"{d.year}/{d.year}{d.month:02d}"
                for d in pd.date_range(ref_dt - timedelta(days=365), ref_dt, freq="D")
            }
        )
    )

    def _fetch_one(args: Tuple[str, str]) -> Tuple[str, float, float, int]:
        stn, ym = args
        url = f"{HII_DAILY_RAIN_BASE}/{ym}/{stn}.csv"
        try:
            r = requests.get(url, timeout=6)
            if r.status_code == 200 and ("," in r.text):
                df_s = pd.read_csv(StringIO(r.text))
                dt_col = (
                    "measure_datetime"
                    if "measure_datetime" in df_s.columns
                    else ("date" if "date" in df_s.columns else None)
                )
                r_col = (
                    "rainfall_daily"
                    if "rainfall_daily" in df_s.columns
                    else ("rain" if "rain" in df_s.columns else None)
                )
                if dt_col and r_col:
                    dts = df_s[dt_col].astype(str).str.slice(0, 10)
                    vals = pd.to_numeric(df_s[r_col], errors="coerce").fillna(0.0).clip(lower=0.0)
                    m365 = (dts >= start_365) & (dts < end_inclusive)
                    m7 = (dts >= start_7) & (dts < end_inclusive)
                    return stn, float(vals[m365].sum()), float(vals[m7].sum()), int(m365.sum())
        except Exception:
            pass
        return stn, 0.0, 0.0, 0

    rain_365: Dict[str, float] = {s: 0.0 for s in station_ids}
    rain_7: Dict[str, float] = {s: 0.0 for s in station_ids}
    days_found: Dict[str, int] = {s: 0 for s in station_ids}

    tasks = [(s, ym) for s in station_ids for ym in yms]
    with ThreadPoolExecutor(max_workers=24) as ex:
        for stn, v365, v7, cnt in ex.map(_fetch_one, tasks):
            rain_365[stn] = round(rain_365[stn] + v365, 2)
            rain_7[stn] = round(rain_7[stn] + v7, 2)
            days_found[stn] += cnt

    active_stns = sum(1 for s in station_ids if days_found[s] > 0)
    out = (rain_365, rain_7, active_stns)
    _HII_RAIN_CACHE[cache_key] = out
    return out

