import os
import numpy as np
import pandas as pd

def validate_labels(nparray_folder, csv_file_path):
    # Load the CSV file
    labels_df = pd.read_csv(csv_file_path, skiprows=1)
    mismatches = []

    # Loop through the npy files and compare with CSV
    for root, _, files in os.walk(nparray_folder):
        for file in files:
            if file.endswith('.npy'):
                # Extract the timestamp from the filename
                timestamp = extract_timestamp_from_filename(file)
                #print(f"Processing {timestamp}")
                # Find the corresponding row in the CSV file
                label_row = labels_df[labels_df['Timestamp'].str.contains(timestamp, na=False)]
                if not label_row.empty:
                    original_label = label_row.iloc[0]['Tag']
                    
                    # Load the numpy file and get its label
                    file_path = os.path.join(root, file)
                    npy_data = np.load(file_path, allow_pickle=True).item()
                    npy_label = npy_data['label']
                    
                    # Check if the labels match
                    if npy_label != original_label:
                        mismatches.append((file, npy_label, original_label))

    return mismatches

def extract_timestamp_from_filename(filename):
    # Extract the timestamp from the filename
    parts = filename.split('_')
    timestamp = ':'.join(parts[1:4]).split('.npy')[0]
    return timestamp

# Usage
nparray_folder = 'preprocessed_dataset/nparray_uv_102x76_resizefirst'
csv_file_path = 'Features_1&0.5_Vision.csv'
label_mismatches = validate_labels(nparray_folder, csv_file_path)

# Print mismatches
for mismatch in label_mismatches:
    print(f"Mismatch in file {mismatch[0]}: npy label {mismatch[1]}, original label {mismatch[2]}")
