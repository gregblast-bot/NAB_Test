import numpy as np

def extract_features(window_data):
    """
    Turn data into 6 descriptive features.
    """
    # Basic Stats
    mean_val = np.mean(window_data) # averge load (overall level of resource usage)
    std_val = np.std(window_data) # volatility (how much it varies)
    max_val = np.max(window_data) # peak load (the highest point in the window)
    
    # Dynamics (How fast is it changing?)
    # max_delta: The biggest single jump between two 5-min readings
    max_delta = np.max(np.abs(np.diff(window_data))) if len(window_data) > 1 else 0 # burstiness (sudden spikes)
    
    # Shape (Is it trending up or down?)
    # Simple slope calculation using linear regression logic
    x = np.arange(len(window_data))
    slope = np.polyfit(x, window_data, 1)[0] if len(window_data) > 1 else 0 # trend (is it going up or down?)
    
    # Outliers within the window
    # Is the last point way higher than the average of the window?
    z_score_last = (window_data[-1] - mean_val) / (std_val + 1e-6) # instability (is the latest point an outlier compared to the window?)

    return [mean_val, std_val, max_val, max_delta, slope, z_score_last]