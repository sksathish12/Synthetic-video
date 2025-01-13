import os
import numpy as np
import pandas as pd

csv_file_path = 'D:\\MSC_Project\\KTP_Project-main\\Features_1&0.5_Vision.csv'
labels_df = pd.read_csv(csv_file_path, skiprows=1)

base_folder = 'D:\\Preprocessed_Dataset'

filename_label_dict = {}

for root, dirs, files in os.walk(base_folder):
    for file in files:
        if file.endswith('.npy'):
            parts = file.split('_')
            if len(parts) < 2:  # if there are not enough parts, skip this file
                print(f"Invalid filename {file}, skipping.")
                continue
            
            timestamp = '_'.join(parts[1:])  # Join the timestamp parts
            timestamp = timestamp.rsplit('.', 1)[0]  # Remove the file extension
            timestamp = timestamp.replace('_', ':', 2)
            print(f"Processing {file} with timestamp {timestamp}")

            matched_rows = labels_df[labels_df['Timestamp'].str.contains(timestamp, na=False)]
            
            if not matched_rows.empty:
                # Use the first matched row
                label_row = matched_rows.iloc[0]
                label = label_row['Tag']

                # print(f"Matched Timestamp in CSV: {label_row['Timestamp']}")
                # print(f"Label being assigned: {label}")

                # Load the numpy file
                file_path = os.path.join(root, file)
                data = np.load(file_path, allow_pickle=True)
                
                # Replace existing label or add new one
                if isinstance(data, dict) and 'array' in data:
                    data['label'] = label
                else:
                    data = {'array': data, 'label': label}
                # Save the updated data back to the numpy file
                np.save(file_path, data)
                
                # Update the filename_label_dict
                filename_label_dict[file] = label
                
            else:
                print(f"No label found for {file}, deleting the file.")
                os.remove(os.path.join(root, file))


for filename, assigned_label in filename_label_dict.items():
    parts = filename.split('_')
    timestamp = f"{parts[1]}T{parts[2]}:{parts[3].split('.', 1)[0]}"
    
    # Find corresponding row in CSV
    label_row = labels_df[labels_df['Timestamp'].str.contains(timestamp, na=False)]
    
    # Extract the original label from the CSV
    if not label_row.empty:
        original_label = label_row.iloc[0]['Tag']
        
        # Check if the original label and assigned label match
        if assigned_label != original_label:
            print(f"Label mismatch for {filename}: assigned {assigned_label}, original {original_label}")