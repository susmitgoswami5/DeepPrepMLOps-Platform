import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import json
import math
import polars as pl
from typing import List, Dict
from datetime import datetime
import redis
from prometheus_client import make_asgi_app, Counter, Histogram
import shap
from confluent_kafka import Producer

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

# Kafka Producer config
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
KAFKA_PRODUCER = None
SURGE_THRESHOLD_MINUTES = 30.0
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
    is_surging: bool = False

class BreachFeedback(BaseModel):
    restaurant_id: str
    actual_fpt: float
    predicted_p95: float
    created_at: str
    items: List[str]
    
    class Config:
        schema_extra = {
            "example": {
                "restaurant_id": "R005",
                "actual_fpt": 35.5,
                "predicted_p95": 30.4,
                "created_at": "2024-03-10T19:30:00",
                "items": ["Burger"]
            }
        }

class ExplainResponse(BaseModel):
    order_id: str
    prediction_p95: float
    feature_importance_shap: Dict[str, float]
    key_factors: List[str]
    is_surging: bool = False

@app.on_event("startup")
async def startup_event():
    global MODEL, CHALLENGER_MODEL, R2I, FS, BG_DISTS, KAFKA_PRODUCER
    
    # Init Kafka Producer for active system control loop
    try:
        KAFKA_PRODUCER = Producer({'bootstrap.servers': KAFKA_BROKER})
    except Exception as e:
        print(f"Failed to init Kafka Producer: {e}")
        KAFKA_PRODUCER = None
    
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
    BG_DISTS["w"] = torch.randn(10, 7, dtype=torch.float32)
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
    w_tensor = torch.tensor([[hour_sin, hour_cos, day_sin, day_cos, avg_emb[0], avg_emb[1], avg_emb[2]]], dtype=torch.float32)
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
    try:
        stress_val = REDIS_CLIENT.get(f"stress:{req.restaurant_id}")
    except Exception as e:
        # Fallback if Redis is unreachable
        stress_val = None
    stress_mult = float(stress_val) if stress_val else 1.0
    
    # 8. LIVE WEATHER STREAM - Geospatial Condition Augmentation
    try:
        weather_sev = REDIS_CLIENT.get("city_weather_severity")
    except Exception as e:
        weather_sev = None
    w_mult = (1.0 + float(weather_sev)) if weather_sev else 1.0
    
    # Accumulate dynamic multipliers natively
    total_mult = stress_mult * w_mult
    
    p50 *= total_mult
    p90 *= total_mult
    p95 *= total_mult
    
    # --- ACTIVE SURGE CONTROL LOOP ---
    is_surging = False
    try:
        # Check if already surging
        if REDIS_CLIENT.get(f"surge:{req.restaurant_id}"):
            is_surging = True
        
        # Trigger surge if threshold breached and not already surging
        if p95 > SURGE_THRESHOLD_MINUTES and not is_surging:
            is_surging = True
            REDIS_CLIENT.setex(f"surge:{req.restaurant_id}", 300, "1") # 5 min TTL
            print(f"🚨 SURGE ACTIVATED for {req.restaurant_id} (ETA: {p95:.1f}m)")
            
            # Broadcast to logistics network via Kafka
            if KAFKA_PRODUCER:
                event = {
                    "event_type": "SURGE_ACTIVATED",
                    "restaurant_id": req.restaurant_id,
                    "p95_minutes": p95,
                    "timestamp": datetime.utcnow().isoformat()
                }
                KAFKA_PRODUCER.produce("restaurant_events", value=json.dumps(event).encode("utf-8"))
                KAFKA_PRODUCER.poll(0)
    except Exception as e:
        pass # Graceful degradation if Redis/Kafka fails
        
    return FPTPrediction(
        order_id=req.order_id,
        p50_minutes=round(max(1.0, p50), 2),
        p90_minutes=round(max(1.0, p90), 2),
        p95_minutes=round(max(1.0, p95), 2),
        stress_multiplier_applied=stress_mult,
        model_version=model_version,
        is_surging=is_surging
    )

@app.post("/feedback")
async def register_feedback(feedback: BreachFeedback):
    """
    Simulates an MLOps Continuous Training loop. 
    1. If actual FPT > predicted P95, we mark a breach and increase the stress factor for that restaurant.
    2. We perform a live stochastic gradient descent step (micro-training) on the Champion model 
       to 'self-heal' towards the true output.
    """
    stress_key = f"stress:{feedback.restaurant_id}"
    stress_val = REDIS_CLIENT.get(stress_key)
    current_stress = float(stress_val) if stress_val else 1.0
    
    new_stress = current_stress # Initialize new_stress
    if feedback.actual_fpt > feedback.predicted_p95:
        BREACH_COUNT.labels(restaurant_id=feedback.restaurant_id).inc()
        # Apply a temporary +5% buffer to this restaurant
        new_stress = min(current_stress * 1.05, 1.50) # Cap at 1.5x
        REDIS_CLIENT.set(stress_key, new_stress)
        
    try:
        # --- ACTIVE MICRO-TRAINING (Self-Healing) ---
        dt = datetime.fromisoformat(feedback.created_at)
        day_of_week = dt.weekday()
        hour = dt.hour + dt.minute / 60.0
        
        day_sin = math.sin(2 * math.pi * day_of_week / 7.0)
        day_cos = math.cos(2 * math.pi * day_of_week / 7.0)
        hour_sin = math.sin(2 * math.pi * hour / 24.0)
        hour_cos = math.cos(2 * math.pi * hour / 24.0)
        
        # We need the items to recreate the embedding
        avg_emb = get_mock_embedding(feedback.items) if hasattr(feedback, 'items') and feedback.items else [0.0, 0.0, 0.0]
        
        # Fetch Feast state (approximated for feedback loop simplicity)
        rest_idx = R2I.get(feedback.restaurant_id, 0)
        
        r_tensor = torch.tensor([rest_idx], dtype=torch.long)
        w_tensor = torch.tensor([[hour_sin, hour_cos, day_sin, day_cos, avg_emb[0], avg_emb[1], avg_emb[2]]], dtype=torch.float32)
        s_tensor = torch.tensor([[[0.0], [0.0], [0.0]]], dtype=torch.float32) # Simplified for feedback
        
        # Forward pass on Champion
        MODEL.train()
        optimizer = torch.optim.SGD(MODEL.parameters(), lr=1e-4) # Very low LR for safety
        
        optimizer.zero_grad()
        preds = MODEL(r_tensor, w_tensor, s_tensor)
        
        # Loss against Actual FPT
        actual_tensor = torch.tensor([[feedback.actual_fpt, feedback.actual_fpt, feedback.actual_fpt]], dtype=torch.float32)

        # MSE Loss
        loss = torch.nn.functional.mse_loss(preds, actual_tensor)
        loss.backward()
        
        # Gradient Clipping to prevent explosion from one bad outlier
        torch.nn.utils.clip_grad_norm_(MODEL.parameters(), max_norm=1.0)
        optimizer.step()
        
        print(f"🧠 [SELF-HEAL] Optimization step complete. Loss: {loss.item():.4f} -> Weights Updated for {feedback.restaurant_id}")
        MODEL.eval()
        
    except Exception as e:
        print(f"⚠️ Micro-training step failed: {e}")

    # if not breaching, slowly cool down the stress multiplier
    if feedback.actual_fpt <= feedback.predicted_p95 and current_stress > 1.0:
        new_stress = max(current_stress * 0.99, 1.0)
        REDIS_CLIENT.set(stress_key, new_stress)
        
    return {"status": "feedback_registered", "new_stress_multiplier": new_stress}


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
    w_tensor = torch.tensor([[hour_sin, hour_cos, day_sin, day_cos, avg_emb[0], avg_emb[1], avg_emb[2]]], dtype=torch.float32)
    s_tensor = torch.tensor([[[load_15], [load_30], [load_60]]], dtype=torch.float32)
    
    with torch.no_grad():
        preds = MODEL(r_tensor, w_tensor, s_tensor)
    p95 = preds[0][2].item()
    
    # SHAP explainer requires gradients
    MODEL.train()
    
    is_surging = False # Default for explanation response
    try:
        # Check if already surging for explanation response
        if REDIS_CLIENT.get(f"surge:{req.restaurant_id}"):
            is_surging = True
    except Exception:
        pass

    try:

        # SHAP cannot differentiate through an Embedding lookup (which requires Long indices).
        # We wrap the model to hardcode the restaurant index and explain the continuous features only.
        class ExplainWrapper(torch.nn.Module):
            def __init__(self, model, r_in):
                super().__init__()
                self.model = model
                self.r_in = r_in
            def forward(self, w, s):
                # Expand the hardcoded restaurant tensor to match the batch dimension of w (which SHAP might provide as size 10 or 1)
                bsz = w.size(0)
                expanded_r = self.r_in.expand(bsz)
                return self.model(expanded_r, w, s)
                
        wrapped_model = ExplainWrapper(MODEL, r_tensor)
        explainer = shap.GradientExplainer(wrapped_model, [BG_DISTS["w"], BG_DISTS["s"]])
        
        # Pass float inputs to explainer
        shap_values = explainer.shap_values([w_tensor.requires_grad_(), s_tensor.requires_grad_()])
        
        # shap_values is a list of arrays mapping to the inputs [w, s].
        # For multi-output PyTorch models, shap sometimes sums or averages across outputs if not specified.
        # w is at index 0, s is at index 1.
        w_shaps = shap_values[0][0]  # Wide features importances for the first batch item
        s_shaps = shap_values[1][0].flatten() # Seq features importances
        
        feature_names = [
            "hour_sin", "hour_cos", "day_sin", "day_cos", 
            "item_embed_0", "item_embed_1", "item_embed_2", 
            "load_15m", "load_30m", "load_60m"
        ]
        
        importances = [float(x) for x in list(w_shaps.flatten()) + list(s_shaps.flatten())]
        
        # Map values
        shap_dict = {name: imp for name, imp in zip(feature_names, importances)}
        
        # Sort factors by absolute magnitude to find key ones
        sorted_factors = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        key_factors = []
        for i in range(min(3, len(sorted_factors))):
            k, v = sorted_factors[i]
            key_factors.append(f"{k} ({'+' if v>0 else ''}{round(v, 2)})")
            
        MODEL.eval()
        return ExplainResponse(
            order_id=req.order_id,
            prediction_p95=round(float(p95), 2),
            feature_importance_shap=shap_dict,
            key_factors=key_factors,
            is_surging=is_surging
        )
        
    except Exception as e:
        print(f"SHAP Warning: {e}")
        shap_dict = {"fallback_error": 0.0}
        key_factors = ["Explainer fallback - Mocked Drivers", "Queue High"]
        
        MODEL.eval()
        return ExplainResponse(
            order_id=req.order_id,
            prediction_p95=round(float(p95), 2),
            feature_importance_shap=shap_dict,
            key_factors=key_factors,
            is_surging=is_surging
        )
class CopilotResponse(BaseModel):
    narrative: str

@app.post("/copilot", response_model=CopilotResponse)
async def generate_copilot_explanation(explain_data: ExplainResponse):
    """
    Template-Based Natural Language Generation (NLG) 
    Simulates a lightweight LLM acting as an Operations Copilot.
    Translates raw SHAP math into actionable restaurant operations insights without requiring external OpenAI API keys.
    """
    # Base starting phrase
    if explain_data.prediction_p95 < 15.0:
        base = f"Order **{explain_data.order_id}** is on a fast track (ETA: {explain_data.prediction_p95}m)."
    elif explain_data.prediction_p95 < 30.0:
        base = f"Order **{explain_data.order_id}** is operating under normal kitchen load (ETA: {explain_data.prediction_p95}m)."
    else:
        base = f"⚠️ Order **{explain_data.order_id}** is significantly delayed (ETA: {explain_data.prediction_p95}m)."

    if explain_data.is_surging:
        base += " 🚨 **SURGE PROTOCOLS ARE CURRENTLY ACTIVE ON THIS RESTAURANT!**"

    # Analyze SHAP drivers
    positive_drivers = []
    negative_drivers = [] # Negative SHAP means it *reduced* prep time
    
    for feature, shap_val in explain_data.feature_importance_shap.items():
        if feature == "fallback_error": continue
        if shap_val > 0.1:
            positive_drivers.append(feature)
        elif shap_val < -0.1:
            negative_drivers.append(feature)
            
    # Translation Dictionary
    t_map = {
        "load_15m": "a sudden spike in orders over the last 15 minutes",
        "load_30m": "sustained high grilling station utilization over the past half-hour",
        "load_60m": "an overwhelming backlog of tickets from the last hour",
        "hour_sin": "the current time of day",
        "hour_cos": "the current time of day",
        "day_sin": "the current day of the week",
        "day_cos": "the current day of the week",
        "item_embed_0": "the complexity of the specific items ordered (e.g., highly customized menu items)",
        "item_embed_1": "item prep requirements touching multiple kitchen stations",
        "item_embed_2": "items requiring extended oven/grill time"
    }

    # Synthesize explanation
    insight = " "
    if positive_drivers:
        insight += f"This delay is primarily being driven up by **{t_map.get(positive_drivers[0], positive_drivers[0])}**"
        if len(positive_drivers) > 1:
            insight += f" and compounded by **{t_map.get(positive_drivers[1], positive_drivers[1])}**."
        else:
            insight += "."
    else:
        if explain_data.prediction_p95 > 25.0:
             insight += " The AI is detecting general baseline friction, though no specific micro-factor is blowing up."

    if negative_drivers and explain_data.prediction_p95 < 20.0:
        insight += f" Prep time was successfully reduced thanks to **{t_map.get(negative_drivers[0], negative_drivers[0])}** helping clear the queue quickly."

    if not positive_drivers and not negative_drivers and "fallback_error" in explain_data.feature_importance_shap:
        insight += " *(Note: Detailed mathematical breakdown is unavailable right now due to an explainer fallback state.)*"

    action = ""
    if explain_data.is_surging:
        action = "\n\n**Copilot Recommendation:** Demand generation should remain throttled until the 60-minute trailing load drops significantly."
    elif "load_15m" in positive_drivers:
        action = "\n\n**Copilot Recommendation:** Ensure all cooks are on the line. A sudden rush just hit."

    final_narrative = f"{base}{insight}{action}"

    return CopilotResponse(narrative=final_narrative.strip())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.router:app", host="0.0.0.0", port=8000, reload=True)
