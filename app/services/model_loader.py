import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self._models: Dict[str, Any] = {}
        self._scalers: Dict[str, Any] = {}
        self._load_available_checkpoints()

    def _load_available_checkpoints(self):
        try:
            import joblib
        except ImportError:
            logger.warning("joblib not installed in environment; models will be loaded on demand.")
            return

        # 1. Sukhothai (20-Feature EWS Random Forest + 12-Feature LightGBM)
        skt_ews_model_path = self.models_dir / "skt" / "ews_best_model.joblib"
        skt_ews_scaler_path = self.models_dir / "skt" / "ews_robust_scaler.joblib"
        if skt_ews_model_path.exists() and skt_ews_scaler_path.exists():
            try:
                self._models["skt_ews"] = joblib.load(skt_ews_model_path)
                self._scalers["skt_ews"] = joblib.load(skt_ews_scaler_path)
                logger.info("Loaded Sukhothai 20-Feature EWS Random Forest model and scaler.")
            except Exception as e:
                logger.warning(f"Could not load SKT EWS model: {e}")

        skt_model_path = self.models_dir / "skt" / "lightgbm_best_model.joblib"
        skt_scaler_path = self.models_dir / "skt" / "robust_scaler.joblib"
        if skt_model_path.exists():
            try:
                self._models["skt_lightgbm"] = joblib.load(skt_model_path)
                self._scalers["skt"] = joblib.load(skt_scaler_path)
                logger.info("Loaded Sukhothai LightGBM model and scaler.")
            except Exception as e:
                logger.warning(f"Could not load SKT model: {e}")

        # 2. Suphan Buri (15-Feature Spatiotemporal EWS LightGBM)
        spb_model_path = self.models_dir / "spb" / "lightgbm_best_model.joblib"
        spb_scaler_path = self.models_dir / "spb" / "robust_scaler.joblib"
        if spb_model_path.exists():
            try:
                self._models["spb_lightgbm"] = joblib.load(spb_model_path)
                self._scalers["spb"] = joblib.load(spb_scaler_path)
                logger.info("Loaded Suphan Buri LightGBM model and scaler.")
            except Exception as e:
                logger.warning(f"Could not load SPB model: {e}")

        # 3. Nakhon Pathom (16-Feature Station-Integrated EWS CatBoost)
        npt_cb_path = self.models_dir / "npt" / "catboost_best_model.joblib"
        npt_lgb_path = self.models_dir / "npt" / "lightgbm_best_model.joblib"
        npt_scaler_path = self.models_dir / "npt" / "robust_scaler.joblib"
        if npt_cb_path.exists():
            try:
                self._models["npt_catboost"] = joblib.load(npt_cb_path)
                logger.info("Loaded Nakhon Pathom CatBoost champion model.")
            except Exception as e:
                logger.warning(f"Could not load NPT CatBoost model: {e}")
        if npt_lgb_path.exists():
            try:
                self._models["npt_lightgbm"] = joblib.load(npt_lgb_path)
                logger.info("Loaded Nakhon Pathom LightGBM model.")
            except Exception as e:
                logger.warning(f"Could not load NPT LightGBM model: {e}")
        if npt_scaler_path.exists():
            try:
                self._scalers["npt"] = joblib.load(npt_scaler_path)
            except Exception as e:
                logger.warning(f"Could not load NPT scaler: {e}")

    def get_model(self, key: str) -> Optional[Any]:
        return self._models.get(key)

    def get_scaler(self, province: str) -> Optional[Any]:
        return self._scalers.get(province)


from app.config import settings
model_registry = ModelRegistry(settings.MODELS_DIR)
