import pandas as pd
import glob
import os
import Features as features

# Define our class mapping based on NAB file categories
CLASS_MAP = {
    'normal': 'artificialNoAnomaly/',
    'cpu': 'realAWSCloudwatch/ec2_cpu',
    'network': 'realAWSCloudwatch/ec2_network',
    'disk': 'realAWSCloudwatch/rds_cpu' # Using RDS as a proxy for I/O heavy
}

def build_labeled_dataset(base_path, window_size=12):
    all_features = []
    
    for label, pattern in CLASS_MAP.items():
        # Find all files matching the pattern
        files = glob.glob(os.path.join(base_path, f"{pattern}*.csv"))
        
        for file in files:
            df = pd.read_csv(file)
            values = df['value'].values
            
            # Slide the window across the data
            for i in range(len(values) - window_size):
                window = values[i : i + window_size]
                
                # Use our previously defined extraction logic
                features = features.extract_features(window) 
                features['target'] = label # The label for the model
                features['source_file'] = os.path.basename(file)
                
                all_features.append(features)
                
    return pd.DataFrame(all_features)

# Usage
# df_train = build_labeled_dataset('./numenta-nab/data/')
# df_train.to_csv('training_data.csv', index=False)