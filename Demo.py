import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Features as feat

# Mapping our predictions to the checklists we defined
CHECKLISTS = {
    'NORMAL': ["No action required. System healthy."],
    'CPU': ["Run 'top -o %CPU'", "Check for new deployments", "Verify Autoscaling"],
    'NETWORK': ["Check 'netstat' for spikes", "Verify LB 5xx error rates", "Check for DDoS patterns"],
    'DISK': ["Check 'df -h' for full partitions", "Run 'iostat -x'", "Rotate/Delete old logs"],
    'MEM': ["Check 'dmesg' for OOM killer", "Inspect Java/Python heap logs", "Restart service"]
}

def run_triage_report(csv_path, X_train, y_train, window_start, window_end, clf):
    # Load and process the incident data
    data = pd.read_csv(csv_path)['value'].values
    last_window = data[window_start:window_end]     # We take the last window (most recent data)
    features = feat.extract_features(last_window)
    
    # Make the Prediction
    prediction = clf.predict([features])[0]
    
    # Generate Evidence
    # Compare against global 'Normal' averages from training
    normal_mean = X_train[y_train == 'NORMAL'].mean(axis=0)
    feature_names = ['Load', 'Volatility', 'Peak Load', 'Burstiness', 'Trend', 'Instability']
    diffs = (features - normal_mean) / (normal_mean + 1e-6)
    top_reason_idx = np.argmax(np.abs(diffs))
    
    # Print the Console Report
    print("="*40)
    print(f"INCIDENT REPORT: {csv_path}")
    print(f"DIAGNOSIS: {prediction}")
    print(f"EVIDENCE: {feature_names[top_reason_idx]} is {diffs[top_reason_idx]:.1f}x higher than normal.")
    print("-" * 20)
    print("SUGGESTED NEXT STEPS:")
    for i, step in enumerate(CHECKLISTS[prediction], 1):
        print(f"  {i}. {step}")
    print("="*40)

    # Visual Output
    plt.figure(figsize=(10, 4))
    plt.plot(data, label='Metric Value', color='gray', alpha=0.5)
    plt.plot(range(window_start, window_end), last_window, color='red', label='Incident Window')
    plt.title(f"Triage Visualization: {prediction}")
    plt.legend()
    plt.show()