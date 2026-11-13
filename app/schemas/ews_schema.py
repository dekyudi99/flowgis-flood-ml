from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    province: str = Field(..., description="Target province: 'sukhothai', 'suphanburi', or 'nakhon_pathom'")
    lead_time: str = Field(default="1d", description="Forecast horizon: '1d', '3d', or '7d'")
    model_name: Optional[str] = Field(default="best", description="Model architecture: 'best', 'lightgbm', or 'catboost'")
    target_date: Optional[str] = Field(default=None, description="ISO Date string (YYYY-MM-DD)")
    threshold: Optional[float] = Field(default=0.50, description="Binary alert decision threshold between 0.0 and 1.0")

class RiskStatistics(BaseModel):
    total_points: int
    very_high_count: int
    very_high_pct: float
    high_count: int
    high_pct: float
    moderate_count: int
    moderate_pct: float
    low_count: int
    low_pct: float
    very_low_count: int
    very_low_pct: float
    overall_alert_level: str

class PredictResponse(BaseModel):
    preview_id: str
    province: str
    lead_time: str
    model_used: str
    execution_time_ms: float
    statistics: RiskStatistics
    geojson: Dict[str, Any]
