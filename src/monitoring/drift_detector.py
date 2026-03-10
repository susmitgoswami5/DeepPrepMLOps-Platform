import sys
import os
import json
import torch
import polars as pl

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    EVIDENTLY_AVAILABLE = True
except Exception as e:
    EVIDENTLY_AVAILABLE = False
    print(f"Warning: Evidently could not load (likely Python 3.14 Pydantic V1 incompatibility): {e}")

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.mock_generator import generate_mock_logs
from src.features.engineering import engineer_features
from src.model.architecture import DeepPrepModel
from src.training.dataset import DeepPrepDataset

def get_predictions(df: pl.DataFrame, model_path="models/deep_prep_weights.pth", r2i_path="models/restaurant_mapping.json") -> pl.DataFrame:
    """Helper to add predictions to a DataFrame for Evidently target drift analysis."""
    # Ensure model and mapping exist
    if not os.path.exists(model_path) or not os.path.exists(r2i_path):
        print("Model or mapping not found. Returning fake predictions.")
        return df.with_columns(pl.col("target_fpt_minutes").alias("prediction"))
        
    with open(r2i_path, "r") as f:
        r2i = json.load(f)
        
    dataset = DeepPrepDataset(df, r2i)
    
    # We load a model (num restaurants is len of r2i)
    model = DeepPrepModel(num_restaurants=len(r2i))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    preds_list = []
    with torch.no_grad():
        for i in range(len(dataset)):
            r_idx, w_f, s_f, _ = dataset[i]
            # Add batch dim
            r_idx = r_idx.unsqueeze(0)
            w_f = w_f.unsqueeze(0)
            s_f = s_f.unsqueeze(0)
            
            preds = model(r_idx, w_f, s_f)
            # P50 is index 0
            preds_list.append(preds[0][0].item())
            
    return df.with_columns(pl.Series("prediction", preds_list))

def generate_drift_report():
    if not EVIDENTLY_AVAILABLE:
        print("Skipping drift report generation. Evidently is disabled on this environment.")
        return
        
    print("Generating Reference Data (Past Week)...")
    ref_raw = generate_mock_logs(1000)
    ref_feat = engineer_features(ref_raw)
    ref_scored = get_predictions(ref_feat)
    
    print("Generating Current Data (Today)...")
    curr_raw = generate_mock_logs(500)
    curr_feat = engineer_features(curr_raw)
    curr_scored = get_predictions(curr_feat)
    
    # Evidently operates on Pandas DataFrames
    ref_df = ref_scored.to_pandas()
    curr_df = curr_scored.to_pandas()
    
    # Select feature columns to monitor for drift (excluding lists or embeddings for simplicity)
    numerical_features = [
        "load_15m", "load_30m", "load_60m", 
        "cook_utilization", "queue_wait_time_m", "orders_in_queue"
    ]
    
    report = Report(metrics=[
        DataDriftPreset(num_features=numerical_features),
        # Assuming our model outputs 'prediction' and ground truth is 'target_fpt_minutes' (which we treat as the target)
        # Note: If ground truth is not available in real-time, target drift is usually monitored on prediction distributions
        TargetDriftPreset() 
    ])
    
    # In evidently, target must be defined in column mapping if names differ
    # But if we just map columns in pandas, evidently guesses them. Let's explicitly set.
    col_mapping = {
        "target": "target_fpt_minutes",
        "prediction": "prediction",
        "numerical_features": numerical_features
    }
    
    print("Running Evidently Drift Analysis...")
    report.run(reference_data=ref_df, current_data=curr_df, column_mapping=col_mapping)
    
    os.makedirs("src/monitoring/reports", exist_ok=True)
    report_path = "src/monitoring/reports/model_health_report.html"
    report.save_html(report_path)
    print(f"✅ Drift Report generated and saved to: {report_path}")
    
    # Simple Alerting Logic by reading json output
    json_report = json.loads(report.json())
    dataset_drift = json_report["metrics"][0]["result"]["dataset_drift"]
    
    if dataset_drift:
        print("🚨 ALERT: Data Drift Detected in recent traffic! Triggering retraining pipeline...")
    else:
        print("✅ Data Distribution is stable.")

if __name__ == "__main__":
    generate_drift_report()
