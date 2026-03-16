import numpy as np
import pandas as pd
import os
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import Features as feat
import Demo as demo

# Define our class mapping based on NAB file categories
CLASS_MAP = {
    'NORMAL': 'artificialNoAnomaly/',
    'CPU': 'realAWSCloudwatch/ec2_cpu',
    'NETWORK': 'realAWSCloudwatch/ec2_network',
    'DISK': 'realAWSCloudwatch/rds_cpu', # Using RDS as a proxy for I/O heavy
    'MEM': 'realAWSCloudwatch/ec2_cpu_utilization_24ae8d' # test specific file temporarily
}

def create_training_data(base_path, window_size=12):
    X, y = [], []
    
    for label, folder_path in CLASS_MAP.items():
        files = glob.glob(os.path.join(base_path, f"{folder_path}*.csv"))
        
        for file in files:
            data = pd.read_csv(file)['value'].values

            for i in range(len(data) - window_size):
                window = data[i : i + window_size]
                
                # Trust the folder label
                current_label = label
                
                if label == 'CPU':
                    # If the CPU is under 50%, it's functionally 'NORMAL' for training purposes
                    if window[-1] < 50: 
                        current_label = 'NORMAL'
                    else:
                        current_label = 'CPU'
                        
                # For DISK and NETWORK, we keep the original label. 
                # These issues are often about the 'New Baseline', not just one point.      
                X.append(feat.extract_features(window))
                y.append(current_label.upper())
                
    return np.array(X), np.array(y)

X, y = create_training_data('./data/')

# Split and Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

# Evaluation
print(classification_report(y_test, clf.predict(X_test)))

# See where the model gets confused
ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, cmap='Blues')
plt.title("Failure Type Prediction Accuracy")
plt.show()

# Run demo on a real incident file
demo.run_triage_report('data/artificialNoAnomaly/art_daily_no_noise.csv', X_train, y_train, 490, 502, clf) # normal test
demo.run_triage_report('data/realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv', X_train, y_train, 2970, 2982, clf) # cpu issue test (runaway process)
demo.run_triage_report('data/realAWSCloudwatch/ec2_network_in_257a54.csv', X_train, y_train, 1635, 1647, clf) # network issue test (traffic surge)
demo.run_triage_report('data/realAWSCloudwatch/rds_cpu_utilization_cc0c53.csv', X_train, y_train, 3080, 3092, clf) # disk issue test (I/O saturation)
demo.run_triage_report('data/realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv', X_train, y_train, 3545, 3557, clf) # memory issue test (memory leak)