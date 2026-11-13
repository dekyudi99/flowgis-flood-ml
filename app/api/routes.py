from fastapi import APIRouter, HTTPException, status
from app.schemas.ews_schema import PredictRequest, PredictResponse
from app.services.inference_service import run_prediction, preview_sessions
from app.services.gee_service import is_gee_ready
from app.services.gistda_service import is_gistda_configured
from app.config import settings

router = APIRouter(prefix="/api/v1", tags=["Flood Risk EWS Models"])

@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "gistda_connected": is_gistda_configured(),
        "gee_connected": is_gee_ready(),
        "active_preview_sessions": len(preview_sessions)
    }

@router.get("/provinces")
def list_supported_provinces():
    return {
        "provinces": [
            {
                "id": "sukhothai",
                "code": 64,
                "name": "Sukhothai Province",
                "basin": "Yom River Basin (Lower Northern Thailand)",
                "data_source_mode": "GISTDA Open API v2.0 (/features/flood/1d..30d) + GEE Sentinel-1 SAR (365d)",
                "champion_model": "Random Forest EWS Classifier (20 Spatiotemporal Features)",
                "optimal_horizon": "Next-Day (T+1) Daily Early Warning",
                "metrics": {"roc_auc": 0.9914, "recall": "91.46%", "accuracy": "97.10% (Wet) / 100% (Dry)", "precision": "77.32%"}
            },
            {
                "id": "suphanburi",
                "code": 72,
                "name": "Suphan Buri Province",
                "basin": "Tha Chin River Basin Corridor (Central Plains)",
                "data_source_mode": "GISTDA Open API v2.0 (/features/flood/1d..30d) + GEE Sentinel-1 SAR (365d)",
                "champion_model": "LightGBM EWS Classifier (15 Spatiotemporal Features)",
                "optimal_horizon": "Next-Day (T+1) Daily Early Warning",
                "metrics": {"roc_auc": 0.9863, "recall": "100.00%", "accuracy": "85.10% (Wet) / 100% (Dry)", "precision": "41.34%"}
            },
            {
                "id": "nakhon_pathom",
                "code": 73,
                "name": "Nakhon Pathom Province",
                "basin": "Lower Tha Chin Delta (Tidal Estuary)",
                "data_source_mode": "GISTDA Open API v2.0 + HII OpenData Stations (tiservice.hii.or.th) + GEE Sentinel-1 SAR",
                "champion_model": "CatBoost EWS Classifier (16 Station-Integrated Features)",
                "optimal_horizon": "Next-Day (T+1) Daily Early Warning",
                "metrics": {"roc_auc": 0.9723, "recall": "46.15%", "accuracy": "96.90%", "precision": "64.29%"}
            }
        ],
        "risk_tiers": settings.EWS_TIERS
    }

@router.post("/predict", response_model=PredictResponse)
def predict_flood_risk(request: PredictRequest):
    """
    Menjalankan inferensi murni dari model checkpoint peneliti (.joblib)
    tanpa manipulasi atau pengali buatan pasca-model.
    """
    try:
        return run_prediction(request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )

@router.get("/preview/{preview_id}")
def get_preview(preview_id: str):
    """
    Mengambil kembali data GeoJSON & statistik dari preview sesi aktif.
    """
    session = preview_sessions.get(preview_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Preview session '{preview_id}' not found."
        )
    return session
