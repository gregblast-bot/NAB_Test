import pandas as pd
import numpy as np

def extract_features(window_data):
    """
    Turns a 12-point sequence into 6 descriptive features.
    """
    # Basic Stats
    mean_val = np.mean(window_data)
    std_val = np.std(window_data)
    
    # Dynamics (How fast is it changing?)
    # max_delta: The biggest single jump between two 5-min readings
    max_delta = np.max(np.abs(np.diff(window_data))) if len(window_data) > 1 else 0
    
    # Shape (Is it trending up or down?)
    # Simple slope calculation using linear regression logic
    x = np.arange(len(window_data))
    slope = np.polyfit(x, window_data, 1)[0] if len(window_data) > 1 else 0
    
    # Outliers within the window
    # Is the last point way higher than the average of the window?
    z_score_last = (window_data[-1] - mean_val) / (std_val + 1e-6)

    return {
        'avg_load': mean_val,
        'volatility': std_val,
        'burstiness': max_delta,
        'trend_slope': slope,
        'instability_index': z_score_last
    }