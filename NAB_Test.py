import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Load a specific NAB file, looking at CPU utilization
data = pd.read_csv('data\\realAWSCloudwatch\\ec2_cpu_utilization_5f5533.csv')
X = data[['value']]

# Initialize the Model
model = IsolationForest(contamination=0.05, random_state=42)

# Fit and Predict
# Returns 1 for normal, -1 for anomaly
data['anomaly_score'] = model.fit_predict(X)

plt.figure(figsize=(12,6))
plt.plot(data['timestamp'], data['value'], color='blue', label='Normal')
anomalies = data[data['anomaly_score'] == -1]
plt.scatter(anomalies['timestamp'], anomalies['value'], color='red', label='Anomaly')
plt.title('CPU Utilization: Unsupervised Detection')
plt.legend()
plt.show()