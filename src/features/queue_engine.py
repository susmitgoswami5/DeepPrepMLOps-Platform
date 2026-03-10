import math
import polars as pl
import numpy as np

def calculate_mmc_metrics(arrival_rate_lambda: float, service_rate_mu: float, num_servers_c: int):
    """
    Calculates operational metrics using an analytic M/M/c queuing model.
    Lambda: Arrivals per hour
    Mu: Services (orders finished) per hour, per cook
    C: Number of cooks running concurrently
    """
    # Defensive logic to prevent infinite queues or zero division
    if num_servers_c == 0 or service_rate_mu == 0:
        return 0, 0, 0
    
    rho = arrival_rate_lambda / (num_servers_c * service_rate_mu)
    
    if rho >= 1.0:
        # The system is technically unstable, queue blows to infinity mathematically
        # Cap utilization to 0.99 for stability metrics
        rho = 0.99
    
    # 1. Server utilization
    cook_utilization = rho
    
    # 2. Probability of zero orders in the system (P0)
    p0_sum = sum([(num_servers_c * rho)**n / math.factorial(n) for n in range(num_servers_c)])
    p0_c_term = ((num_servers_c * rho)**num_servers_c / math.factorial(num_servers_c)) * (1 / (1 - rho))
    p0 = 1 / (p0_sum + p0_c_term)
    
    # 3. Expected Wait Time in Queue (Wq)
    # Erlang C Formula Probability of waiting
    erlang_c = p0_c_term * p0
    wq_hours = erlang_c / (num_servers_c * service_rate_mu * (1 - rho))
    wq_minutes = wq_hours * 60
    
    # Number of orders in queue (Lq)
    lq = arrival_rate_lambda * wq_hours
    
    return cook_utilization, wq_minutes, lq

def append_queue_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Simulates operational parameters and appends M/M/c capabilities.
    For this simulation, we'll arbitrarily assume fixed parameters per restaurant
    based on their historic throughput.
    """
    
    # Create random assignments mapped strictly to the restaurant ID index
    # (Since we are in Polars, mapping a fake static dict natively)
    
    def apply_mmc(row):
        # We proxy the real "arrival rate per hour" by extrapolating the 60m load feature
        lambda_val = row["load_60m"]
        # Fake "average kitchen service rate": 15 orders finished per hour per cook
        mu_val = 15.0 
        
        # We assume 3 to 6 cooks arbitrarily depending on the load, capping capability
        c_val = max(1, min(6, int(lambda_val / 10)))
        
        utilization, wait_time, queue_length = calculate_mmc_metrics(lambda_val, mu_val, c_val)
        
        return pl.Series([utilization, wait_time, queue_length])
        
    print("M/M/c Model computing queue lengths and Erlang probabilities...")
        
    # We apply row-wise logic in Polars
    metrics = df.select(["load_60m"]).map_rows(
        lambda r: calculate_mmc_metrics(
            arrival_rate_lambda=r[0], 
            service_rate_mu=15.0, 
            num_servers_c=max(1, min(6, int(r[0] / 10)))
        )
    )
    
    df = df.with_columns([
        pl.Series("cook_utilization", metrics.map_elements(lambda x: x[0], return_dtype=pl.Float64)),
        pl.Series("queue_wait_time_m", metrics.map_elements(lambda x: x[1], return_dtype=pl.Float64)),
        pl.Series("orders_in_queue", metrics.map_elements(lambda x: x[2], return_dtype=pl.Float64))
    ])
    
    return df
