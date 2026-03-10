import json
import requests
from confluent_kafka import Consumer, KafkaException

# Kafka config
KAFKA_BROKER = 'localhost:9092'
TOPIC_ORDER_RECEIVED = 'order_received'

# Inference API endpoint
INFERENCE_API = "http://127.0.0.1:8000/predict"

def process_order(msg_val):
    print("--------------------------------------------------")
    print(f"Received Event: {msg_val['event_type']} - Order {msg_val['order_id']}")
    
    # Normally, another consumer path would push this event to Feast/Redis online store
    # to increment the `load_15m` count. For demonstration, we assume real-time
    # features are already updated in Feast and accessible by the FastAPI endpoint.
    
    inference_payload = {
        "order_id": msg_val["order_id"],
        "restaurant_id": msg_val["restaurant_id"],
        "created_at": msg_val["timestamp"],
        "items": msg_val["items"]
        # Notice we are NO LONGER sending loads. The API fetches them via Feast!
    }
    
    try:
        req = requests.post(INFERENCE_API, json=inference_payload)
        req.raise_for_status()
        prediction = req.json()
        
        print(f"✅ Prediction Complete: p50={prediction['p50_minutes']}m | p90={prediction['p90_minutes']}m | p95={prediction['p95_minutes']}m")
        print(f"Applied Stress Multiplier: {prediction['stress_multiplier_applied']}")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to reach Inference API: {e}")

def run_consumer():
    conf = {
        'bootstrap.servers': KAFKA_BROKER,
        'group.id': 'deepprep_inference_group',
        'auto.offset.reset': 'latest'
    }

    c = Consumer(conf)
    c.subscribe([TOPIC_ORDER_RECEIVED])

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
