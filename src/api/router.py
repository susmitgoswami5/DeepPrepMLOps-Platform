import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import json
import polars as pl
from typing import List, Dict
from datetime import datetime
import redis
from prometheus_client import make_asgi_app, Counter, Histogram
import shap

# Import our custom components
from src.model.architecture import DeepPrepModel
from src.features.engineering import get_mock_embedding, add_cyclic_features
from src.features.queue_engine import calculate_mmc_metrics

app = FastAPI(title="DeepPrep FPT Predictor")

# Prometheus Metrics Integration
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

PREDICTION_COUNT = Counter("fpt_predictions_total", "Total FPT predictions served")
# In real life we'd track error vs actual, we will add a mock metric here
BREACH_COUNT = Counter("fpt_breaches_total", "Count of orders breaching predicted p95 FPT", ["restaurant_id"])

# A/B Testing Tracking
TRAFFIC_CHAMPION_COUNT = Counter("fpt_champion_served_total", "Total requests routed to Champion")
TRAFFIC_CHALLENGER_COUNT = Counter("fpt_challenger_served_total", "Total requests routed to Challenger")

# 1. Global State
MODEL = None
CHALLENGER_MODEL = None # Loaded for A/B testing 50/50 split
R2I = {}
FS = None
# "Self-Correcting" stress factor
REDIS_CLIENT = redis.Redis(host=os.getenv("FEAST_REDIS_HOST", "localhost"), port=6379, db=0, decode_responses=True)
# SHAP Background distributions
BG_DISTS = {}

class OrderInferenceRequest(BaseModel):
    order_id: str
    restaurant_id: str
    created_at: str # ISO format string
    items: List[str]
    
    class Config:
        schema_extra = {
            "example": {
                "order_id": "ORD12345",
                "restaurant_id": "R005",
                "created_at": "2024-03-10T19:30:00",
                "items": ["Pizza", "Coke"]
            }
        }

class FPTPrediction(BaseModel):
    order_id: str
    p50_minutes: float
    p90_minutes: float
    p95_minutes: float
    stress_multiplier_applied: float
    model_version: str = "champion"

class BreachFeedback(BaseModel):
    restaurant_id: str
    actual_fpt: float
    predicted_p95: float
    
    class Config:
        schema_extra = {
            "example": {
                "restaurant_id": "R005",
                "actual_fpt": 35.5,
                "predicted_p95": 30.4
            }
        }

class ExplainResponse(BaseModel):
    order_id: str
    prediction_p95: float
    feature_importance_shap: Dict[str, float]
    key_factors: List[str]

@app.on_event("startup")
async def startup_event():
    global MODEL, R2I
    
    # Load Mapping
    mapping_path = "models/restaurant_mapping.json"
    weights_path = "models/deep_prep_weights.pth"
    
    if os.path.exists(mapping_path):
        with open(mapping_path, "r") as f:
            R2I = json.load(f)
    else:
        # Fallback for dev mode
        R2I = {"R001": 0}
        
    # Load Model Structure
    MODEL = DeepPrepModel(num_restaurants=len(R2I) if R2I else 20)
    CHALLENGER_MODEL = DeepPrepModel(num_restaurants=len(R2I) if R2I else 20)
    
    # Load Weights
    if os.path.exists(weights_path):
        weights = torch.load(weights_path, map_location=torch.device('cpu'))
        MODEL.load_state_dict(weights)
        CHALLENGER_MODEL.load_state_dict(weights) # For demo, we just clone the champion
        
    MODEL.eval()
    CHALLENGER_MODEL.eval()
    
    # Init Feast
    from feast import FeatureStore
    FS = FeatureStore(repo_path="src/feature_store")
    
    # Init SHAP Backgrounds (mocked 10 sample distribution)
    torch.manual_seed(42)
    BG_DISTS["r"] = torch.randint(0, len(R2I) if R2I else 20, (10,), dtype=torch.long)
    BG_DISTS["w"] = torch.randn(10, 10, dtype=torch.float32)
    BG_DISTS["s"] = torch.randn(10, 3, 1, dtype=torch.float32) * 5.0
    
    print("Model Loaded for Inference!")

@app.post("/predict", response_model=FPTPrediction)
async def predict_fpt(req: OrderInferenceRequest):
    PREDICTION_COUNT.inc()
    
    # 1. Map Restaurant
    rest_idx = R2I.get(req.restaurant_id, 0)
    
    # 2. Engineer Time Features
    dt = datetime.fromisoformat(req.created_at)
    # create a fast 1-row polars df simply for the cyclic features
    df = pl.DataFrame({"created_at": [dt]})
    df = add_cyclic_features(df, "created_at")
    
    hour_sin = df["hour_sin"][0]
    hour_cos = df["hour_cos"][0]
    day_sin = df["day_sin"][0]
    day_cos = df["day_cos"][0]
    
    # 3. Engineer Item Embeddings
    embs = [get_mock_embedding(i) for i in req.items]
    import numpy as np
    avg_emb = np.mean(embs, axis=0).tolist() if embs else [0.0, 0.0, 0.0]
    
    # 4. Fetch features from Feast
    feature_vector = FS.get_online_features(
        features=[
            "restaurant_features:load_15m",
            "restaurant_features:load_30m",
            "restaurant_features:load_60m"
        ],
        entity_rows=[{"restaurant_id": req.restaurant_id}]
    ).to_dict()
    
    # Default to 0 if not found
    load_15 = feature_vector["load_15m"][0] or 0
    load_30 = feature_vector["load_30m"][0] or 0
    load_60 = feature_vector["load_60m"][0] or 0
    
    # 5. Apply Queuing Theory (M/M/c)
    lambda_val = load_60
    mu_val = 15.0
    c_val = max(1, min(6, int(lambda_val / 10)))
    utilization, wait_time, queue_length = calculate_mmc_metrics(lambda_val, mu_val, c_val)
    
    # 6. Construct Tensors
    r_tensor = torch.tensor([rest_idx], dtype=torch.long)
    w_tensor = torch.tensor([[hour_sin, hour_cos, day_sin, day_cos, avg_emb[0], avg_emb[1], avg_emb[2], utilization, wait_time, queue_length]], dtype=torch.float32)
    s_tensor = torch.tensor([[[load_15], [load_30], [load_60]]], dtype=torch.float32)
    
    # 7. A/B Testing Router & Forward Pass
    if hash(req.order_id) % 2 == 0:
        # Route to Champion
        active_model = MODEL
        model_version = "champion"
        TRAFFIC_CHAMPION_COUNT.inc()
    else:
        # Route to Challenger
        active_model = CHALLENGER_MODEL
        model_version = "challenger"
        TRAFFIC_CHALLENGER_COUNT.inc()
        
    with torch.no_grad():
        preds = active_model(r_tensor, w_tensor, s_tensor)
        
    p50, p90, p95 = preds[0].tolist()
    
    # Apply Self-Correcting Stress Multiplier if applicable
    stress_val = REDIS_CLIENT.get(f"stress:{req.restaurant_id}")
    stress_mult = float(stress_val) if stress_val else 1.0
    
    p50 *= stress_mult
    p90 *= stress_mult
    p95 *= stress_mult
    
    return FPTPrediction(
        order_id=req.order_id,
        p50_minutes=round(max(1.0, p50), 2),
        p90_minutes=round(max(1.0, p90), 2),
        p95_minutes=round(max(1.0, p95), 2),
        stress_multiplier_applied=stress_mult,
        model_version=model_version
    )

@app.post("/feedback")
async def register_feedback(feedback: BreachFeedback):
    """
    Simulates an MLOps feedback loop. 
    If actual FPT > predicted P95, we mark a breach and increase the stress factor for that restaurant.
    """
    stress_key = f"stress:{feedback.restaurant_id}"
    stress_val = REDIS_CLIENT.get(stress_key)
    current_stress = float(stress_val) if stress_val else 1.0
    
    if feedback.actual_fpt > feedback.predicted_p95:
        BREACH_COUNT.labels(restaurant_id=feedback.restaurant_id).inc()
        # Apply a temporary +5% buffer to this restaurant
        new_stress = min(current_stress * 1.05, 1.50) # Cap at 1.5x
        REDIS_CLIENT.set(stress_key, new_stress)
        return {"status": "breach registered", "new_stress_multiplier": new_stress}
    else:
        # if not breaching, slowly cool down the stress multiplier
        new_stress = current_stress
        if current_stress > 1.0:
            new_stress = max(current_stress * 0.99, 1.0)
            REDIS_CLIENT.set(stress_key, new_stress)
        return {"status": "ok", "new_stress_multiplier": new_stress}


@app.post("/explain", response_model=ExplainResponse)
async def explain_prediction(req: OrderInferenceRequest):
    """
    Returns SHAP value feature importances mapped to the input vector.
    """
    rest_idx = R2I.get(req.restaurant_id, 0)
    dt = datetime.fromisoformat(req.created_at)
    df = pl.DataFrame({"created_at": [dt]})
    df = add_cyclic_features(df, "created_at")
    
    hour_sin = df["hour_sin"][0]
    hour_cos = df["hour_cos"][0]
    day_sin = df["day_sin"][0]
    day_cos = df["day_cos"][0]
    
    embs = [get_mock_embedding(i) for i in req.items]
    import numpy as np
    avg_emb = np.mean(embs, axis=0).tolist() if embs else [0.0, 0.0, 0.0]
    
    feature_vector = FS.get_online_features(
        features=["restaurant_features:load_15m", "restaurant_features:load_30m", "restaurant_features:load_60m"],
        entity_rows=[{"restaurant_id": req.restaurant_id}]
    ).to_dict()
    
    load_15 = feature_vector["load_15m"][0] or 0
    load_30 = feature_vector["load_30m"][0] or 0
    load_60 = feature_vector["load_60m"][0] or 0
    
    lambda_val = load_60
    c_val = max(1, min(6, int(lambda_val / 10)))
    utilization, wait_time, queue_length = calculate_mmc_metrics(lambda_val, 15.0, c_val)
    
    r_tensor = torch.tensor([rest_idx], dtype=torch.long)
    w_tensor = torch.tensor([[hour_sin, hour_cos, day_sin, day_cos, avg_emb[0], avg_emb[1], avg_emb[2], utilization, wait_time, queue_length]], dtype=torch.float32)
    s_tensor = torch.tensor([[[load_15], [load_30], [load_60]]], dtype=torch.float32)
    
    with torch.no_grad():
        preds = MODEL(r_tensor, w_tensor, s_tensor)
    p95 = preds[0][2].item()
    
    # SHAP explainer requires gradients
    MODEL.train()
    
    try:
        explainer = shap.GradientExplainer(MODEL, [BG_DISTS["r"], BG_DISTS["w"], BG_DISTS["s"]])
        shap_values = explainer.shap_values([r_tensor, w_tensor.requires_grad_(), s_tensor.requires_grad_()])
        
        # shap_values is a list of arrays per quantile output. We index 2 for P95 target.
        # Inside that, it's a tuple of arrays matching the inputs (r, w, s).
        p95_shap = shap_values[2] # 3rd output head
        w_shaps = p95_shap[1][0]  # Wide features importances for the first batch item
        s_shaps = p95_shap[2][0].flatten() # Seq features importances
        
        feature_names = [
            "hour_sin", "hour_cos", "day_sin", "day_cos", 
            "item_embed_0", "item_embed_1", "item_embed_2", 
            "cook_utilization", "queue_wait_time", "orders_in_queue",
            "load_15m", "load_30m", "load_60m"
        ]
        
        importances = list(w_shaps) + list(s_shaps)
        
        # Map values
        shap_dict = {name: float(imp) for name, imp in zip(feature_names, importances)}
        
        # Sort factors by absolute magnitude to find key ones
        sorted_factors = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        key_factors = []
        for i in range(min(3, len(sorted_factors))):
            k, v = sorted_factors[i]
            key_factors.append(f"{k} ({'+' if v>0 else ''}{round(v, 2)})")
        
    except Exception as e:
        print(f"SHAP Warning: {e}")
        shap_dict = {"fallback_error": 0.0}
        key_factors = ["Explainer fallback - Mocked Drivers", "Queue High"]
        
    MODEL.eval()

    return ExplainResponse(
        order_id=req.order_id,
        prediction_p95=round(p95, 2),
        feature_importance_shap=shap_dict,
        key_factors=key_factors
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.router:app", host="0.0.0.0", port=8000, reload=True)
