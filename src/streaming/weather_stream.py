import json
import time
import random
import os
from confluent_kafka import Producer
from datetime import datetime

# Kafka config
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
TOPIC_ENVIRONMENT = 'environment_events'

def create_weather_stream():
    try:
        producer = Producer({'bootstrap.servers': KAFKA_BROKER})
        print(f"🌦️ City Weather Simulation Stream Started on {KAFKA_BROKER}")
        print(f"Target Topic: {TOPIC_ENVIRONMENT}")
        
        weather_states = [
            {"condition": "CLEAR", "severity_index": 0.0},
            {"condition": "CLOUDY", "severity_index": 0.1},
            {"condition": "LIGHT_RAIN", "severity_index": 0.4},
            {"condition": "HEAVY_RAIN", "severity_index": 0.7},
            {"condition": "SNOWSTORM", "severity_index": 0.95}
        ]
        
        while True:
            # Shift weather every 10 seconds for simulation purposes
            current_weather = random.choice(weather_states)
            
            event = {
                "event_type": "CITY_WEATHER_UPDATE",
                "city_id": "NYC_MANHATTAN",
                "weather_condition": current_weather["condition"],
                "severity_index": current_weather["severity_index"],
                "timestamp": datetime.utcnow().isoformat()
            }
            
            producer.produce(TOPIC_ENVIRONMENT, value=json.dumps(event).encode('utf-8'))
            producer.poll(0)
            
            print(f"[{event['timestamp']}] Broadcasted Weather Update: {current_weather['condition']} (Severity: {current_weather['severity_index']})")
            
            # Flush periodically
            producer.flush()
            time.sleep(10) # Publish every 10 seconds
            
    except KeyboardInterrupt:
        print("Weather System Stopped.")
    except Exception as e:
        print(f"Failed to start Weather Producer: {e}")

if __name__ == "__main__":
    create_weather_stream()
