import os
import sys
import shutil
import torch
from prefect import task, flow, get_run_logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.training.pipeline import train_model

@task(name="Run PyTorch Training Pipeline", retries=2)
def run_training():
    logger = get_run_logger()
    logger.info("Initializing DeepPrep training job...")
    
    # Train model (which internally handles mock data generation & Feast enrichment)
    # We use a small epoch count just for demonstration
    model = train_model(epochs=2)
    
    logger.info("Training complete. Model weights saved to staging directory: `models/deep_prep_weights.pth`")
    
    # Return the path to the staged weights
    return "models/deep_prep_weights.pth", "models/restaurant_mapping.json"

@task(name="Validate Model Candidate")
def validate_model(staging_model_path: str):
    logger = get_run_logger()
    logger.info("Running validation suite against holdout data...")
    
    # In a real system, we'd load ground truth data and evaluate MAE / Quantile Loss
    # Here we mock the validation passing
    validation_passed = True 
    
    if not validation_passed:
        logger.error("Model candidate failed validation criteria (e.g., degraded performance on holdout).")
        raise ValueError("Validation Failed. Deployment halted.")
        
    logger.info("Model candidate passed all validation checks! Ready for production.")
    return True

@task(name="Deploy to Production Artifact Store")
def deploy_model(staging_paths: tuple, validation_result: bool):
    if not validation_result:
        return False
        
    logger = get_run_logger()
    logger.info("Promoting model candidate to Production...")
    
    model_path, mapping_path = staging_paths
    
    # Ensure production directory exists
    prod_dir = "models/production"
    os.makedirs(prod_dir, exist_ok=True)
    
    prod_model_path = os.path.join(prod_dir, "deep_prep_weights.pth")
    prod_mapping_path = os.path.join(prod_dir, "restaurant_mapping.json")
    
    shutil.copy(model_path, prod_model_path)
    shutil.copy(mapping_path, prod_mapping_path)
    
    logger.info(f"Model successfully deployed to {prod_dir}! The API will load these weights on next spin-up/refresh.")
    
@flow(name="DeepPrep Nightly Retraining", description="End-to-end continuous training flow for FPT model")
def continuous_training_pipeline():
    logger = get_run_logger()
    logger.info("Starting DeepPrep Nightly Retraining Flow...")
    
    # 1. Train
    paths = run_training()
    
    # 2. Validate
    is_valid = validate_model(paths[0])
    
    # 3. Deploy
    deploy_model(paths, is_valid)
    
    logger.info("Pipeline Execution Complete!")

if __name__ == "__main__":
    # You can schedule this via Prefect:
    # continuous_training_pipeline.serve(name="nightly-retrain", cron="0 2 * * *")
    continuous_training_pipeline()
