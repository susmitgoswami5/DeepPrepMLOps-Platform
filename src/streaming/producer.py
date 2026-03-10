import time
import json
import random
from confluent_kafka import Producer
from datetime import datetime

# Kafka config
KAFKA_BROKER = 'localhost:9092'
TOPIC_ORDER_RECEIVED = 'order_received'

def delivery_report(err, msg):
    if err is not None:
        print(f"Message delivery failed: {err}")
    else:
        print(f"Message delivered to {msg.topic()} [{msg.partition()}]")

def run_producer():
    p = Producer({'bootstrap.servers': KAFKA_BROKER})
    print(f"Starting simulated order stream on {KAFKA_BROKER}...")
    
    try:
        while True:
            restaurant_id = f"R{str(random.randint(1, 20)).zfill(3)}"
            order_id = f"ORD{random.randint(1000, 9999)}"
            
            items_pool = ["Pizza", "Burger", "Coke", "Pasta", "Salad", "Ice Cream"]
            items = random.sample(items_pool, random.randint(1, 4))
            
            event = {
                "event_type": "ORDER_PLACED",
                "order_id": order_id,
                "restaurant_id": restaurant_id,
                "items": items,
                "timestamp": datetime.utcnow().isoformat(),
            }
            
            data = json.dumps(event)
            p.produce(TOPIC_ORDER_RECEIVED, data.encode('utf-8'), callback=delivery_report)
            p.poll(0)
            
            time.sleep(random.uniform(0.5, 3.0)) # Simulate sporadic order stream
            
    except KeyboardInterrupt:
        pass
    finally:
        p.flush()

if __name__ == '__main__':
    run_producer()
