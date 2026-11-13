import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

class Settings:
    PROJECT_NAME: str = "FlowGIS FloodRisk ML Microservice"
    VERSION: str = "1.0.0"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8008"))
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"

    # Path Directories
    MODELS_DIR: Path = BASE_DIR / "app" / "models"
    DATA_DIR: Path = BASE_DIR / "app" / "data"

    # Google Earth Engine Configuration (synced with flowgis-floodrisk-services)
    GEE_SERVICE_ACCOUNT: str = os.getenv("GEE_SERVICE_ACCOUNT", "")
    GEE_AOI_ASSET: str = os.getenv("GEE_AOI_ASSET", "")
    GEE_PRIVATE_KEY_PATH: str = os.getenv("GEE_PRIVATE_KEY_PATH", ".private-key.json")

    # Official GISTDA Open API Gateway v2.0 Key (supports GISDA_API_KEY or GISTDA_API_KEY from .env)
    GISDA_API_KEY: str = os.getenv("GISDA_API_KEY") or os.getenv("GISTDA_API_KEY", "")

    # EWS Risk Classification Color Hex & Labels
    EWS_TIERS = [
        {"tier": 1, "label": "Very Low", "min": 0.0, "max": 0.20, "color": "#27ae60"},
        {"tier": 2, "label": "Low", "min": 0.20, "max": 0.40, "color": "#2ecc71"},
        {"tier": 3, "label": "Moderate", "min": 0.40, "max": 0.60, "color": "#f1c40f"},
        {"tier": 4, "label": "High", "min": 0.60, "max": 0.80, "color": "#e67e22"},
        {"tier": 5, "label": "Very High", "min": 0.80, "max": 1.00, "color": "#c0392b"},
    ]

settings = Settings()
