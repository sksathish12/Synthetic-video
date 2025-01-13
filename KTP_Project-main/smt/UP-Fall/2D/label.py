import os
import numpy as np
import pandas as pd

csv_file_path = '../Features_1&0.5_Vision.csv'
labels_df = pd.read_csv(csv_file_path, skiprows=1)

base_folder = '../preprocessed_dataset/nparray_2D'

filename_label_dict = {}

for root, dirs, files in os.walk(base_folder):
    for file in files:
        if file.endswith('.npy'):
            parts = file.split('_')
            if len(parts) < 4:  # if there are not enough parts, skip this file
                print(f"Invalid filename {file}, skipping.")
                continue
            
            timestamp = '_'.join(parts[3:])  # Join the timestamp parts
            timestamp = timestamp.rsplit('.', 1)[0]  # Remove the file extension
            timestamp = timestamp.replace('_', ':', 2)
            #print(f"Processing {file} with timestamp {timestamp}")

            matched_rows = labels_df[labels_df['Timestamp'].str.contains(timestamp, na=False)]
            
            if not matched_rows.empty:
                # Use the first matched row
                label_row = matched_rows.iloc[0]
                label = label_row['Tag']

                #print(f"Matched Timestamp in CSV: {label_row['Timestamp']}")
                #print(f"Label being assigned: {label}")

                # Load the numpy file
                file_path = os.path.join(root, file)
                array = np.load(file_path, allow_pickle=True)
                
                # Replace existing label or add new one
                if isinstance(array, dict) and 'array' in array:
                    array['label'] = label
                else:
                    array = {'array': array, 'label': label}
                
                # Save the updated data back to the numpy file
                np.save(file_path, array)
                
                # Update the filename_label_dict
                filename_label_dict[file] = label
                
            else:
                print(f"No label found for {file}, deleting the file.")
                os.remove(os.path.join(root, file))
