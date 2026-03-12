import json
import requests
import os
import redis
from confluent_kafka import Consumer, KafkaException

REDIS_CLIENT = redis.Redis(host=os.getenv("FEAST_REDIS_HOST", "localhost"), port=6379, db=0, decode_responses=True)


# Kafka config
KAFKA_BROKER = 'localhost:9092'
TOPIC_ORDER_RECEIVED = 'order_received'
TOPIC_ENVIRONMENT = 'environment_events'

# Inference API endpoint
INFERENCE_API = "http://127.0.0.1:8000/predict"

def process_order(msg_val):
    if msg_val.get('event_type') == 'CITY_WEATHER_UPDATE':
        severity = msg_val.get('severity_index', 0.0)
        print(f"🌦️ Weather Update Received -> Severity: {severity}")
        # Global city weather state in Redis
        REDIS_CLIENT.set("city_weather_severity", float(severity))
        return

    print("--------------------------------------------------")
    print(f"Received Event: {msg_val['event_type']} - Order {msg_val.get('order_id')}")
    
    if msg_val.get('event_type') == 'ORDER_PLACED':
        r_id = msg_val['restaurant_id']
        # Increment rolling counters in Redis
        REDIS_CLIENT.incr(f"{r_id}:load_15m")
        REDIS_CLIENT.incr(f"{r_id}:load_30m")
        REDIS_CLIENT.incr(f"{r_id}:load_60m")

    if 'order_id' not in msg_val: return
    
    inference_payload = {
        "order_id": msg_val["order_id"],
        "restaurant_id": msg_val["restaurant_id"],
        "created_at": msg_val["timestamp"],
        "items": msg_val.get("items", [])
    }
    
    try:
        req = requests.post(INFERENCE_API, json=inference_payload)
        req.raise_for_status()
        prediction = req.json()
        
        print(f"✅ Prediction Complete: p50={prediction['p50_minutes']}m | p90={prediction['p90_minutes']}m | p95={prediction['p95_minutes']}m")
        print(f"Applied Stress Multiplier: {prediction['stress_multiplier_applied']}")
        if prediction.get("is_surging"):
            print("🚨 SURGE IS ACTIVE FOR THIS ORDER")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to reach Inference API: {e}")

def run_consumer():
    conf = {
        'bootstrap.servers': KAFKA_BROKER,
        'group.id': 'deepprep_inference_group',
        'auto.offset.reset': 'latest'
    }

    c = Consumer(conf)
    c.subscribe([TOPIC_ORDER_RECEIVED, TOPIC_ENVIRONMENT])

    print(f"Starting Inference Consumer on {KAFKA_BROKER}...")
    
    try:
        while True:
            msg = c.poll(timeout=1.0)
            if msg is None: continue
            
            if msg.error():
                if msg.error().code() == KafkaException._PARTITION_EOF: continue
                else: print(msg.error()); break
            
            # Extract JSON
            val = json.loads(msg.value().decode('utf-8'))
            process_order(val)
            
    except KeyboardInterrupt:
        print('Aborted by user')
    finally:
        c.close()

if __name__ == '__main__':
    run_consumer()
