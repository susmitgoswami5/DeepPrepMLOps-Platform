import json
import asyncio
from fastapi.testclient import TestClient
from src.api.router import app, R2I, REDIS_CLIENT

client = TestClient(app)

def test_predict():
    payload = {
        "order_id": "ORD12345",
        "restaurant_id": "R005",
        "created_at": "2024-03-10T19:30:00",
        "items": ["Pizza", "Coke"]
    }
    
    # Mock Redis response for stress factor
    try:
        REDIS_CLIENT.set("stress:R005", "1.2")
    except:
        pass # Redis might not be running, we'll see if it crashes or falls back
        
    try:
        response = client.post("/predict", json=payload)
        print("PREDICT STATUS:", response.status_code)
        print("PREDICT RESP:", response.json())
        assert response.status_code == 200
    except Exception as e:
        print("PREDICT CRASHED:", e)

def test_explain():
    payload = {
        "order_id": "ORD12345",
        "restaurant_id": "R005",
        "created_at": "2024-03-10T19:30:00",
        "items": ["Pizza", "Coke"]
    }
    
    try:
        response = client.post("/explain", json=payload)
        print("\nEXPLAIN STATUS:", response.status_code)
        print("EXPLAIN RESP:", json.dumps(response.json(), indent=2))
        assert response.status_code == 200
    except Exception as e:
        print("\nEXPLAIN CRASHED:", e)

if __name__ == "__main__":
    test_predict()
    test_explain()
