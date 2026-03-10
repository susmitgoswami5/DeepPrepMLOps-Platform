import polars as pl
import numpy as np
import random
from datetime import datetime, timedelta

MENU_ITEMS = [
    "Margherita Pizza", "Pepperoni Pizza", "Garlic Bread", "Cheese Burger",
    "Veggie Burger", "French Fries", "Tandoori Chicken", "Butter Chicken", 
    "Naan", "Ice Cream", "Chocolate Cake", "Coke", "Sprite"
]

def generate_mock_logs(n_orders: int = 5000, start_date_str: str = "2024-01-01") -> pl.DataFrame:
    """Generate a realistic mock dataset for Food Preparation Time."""
    start_time = datetime.strptime(start_date_str, "%Y-%m-%d")
    
    orders = []
    
    # Simulate an uneven distribution of times and load
    current_time = start_time
    rest_ids = [f"R{str(i).zfill(3)}" for i in range(1, 21)]  # 20 restaurants
    
    # Intrinsic prep times roughly
    intrinsic_prep = {
        "Pizza": 15, "Burger": 10, "Fries": 5, "Bread": 5,
        "Chicken": 20, "Naan": 5, "Ice Cream": 2, "Cake": 3, "Soda": 1
    }
    
    for order_id in range(1, n_orders + 1):
        # time increment between 1s and 10 mins purely random to simulate bursts
        inc = timedelta(seconds=random.randint(10, 600))
        current_time += inc
        
        rest_id = random.choice(rest_ids)
        
        # High cardinality for item choices
        num_items = random.choices([1, 2, 3, 4, 5], weights=[0.4, 0.3, 0.15, 0.1, 0.05])[0]
        items = random.choices(MENU_ITEMS, k=num_items)
        
        # Derive base prep time
        base_time = max(
            [next((v for k, v in intrinsic_prep.items() if k in item), 5) for item in items]
        )
        
        # Add some random variance based on item complexity
        complexity = len(items)
        variance = np.random.normal(0, base_time * 0.2)
        
        # We'll calculate load effectively when aggregating, but for now just assign baseline FPT
        fpt = base_time + (complexity * 1.5) + variance
        fpt = max(1.0, fpt)  # Minimum 1 min
        
        orders.append({
            "order_id": f"ORD{str(order_id).zfill(8)}",
            "restaurant_id": rest_id,
            "created_at": current_time,
            "items": items,
            "target_fpt_minutes": fpt
        })

    df = pl.DataFrame(orders)
    # We will engineer actual kitchen load in feature engineering step.
    return df

if __name__ == "__main__":
    df = generate_mock_logs(10000)
    print(df.head())
    df.write_parquet("mock_orders.parquet")
