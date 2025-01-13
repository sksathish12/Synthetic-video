import os
import numpy as np

# Base folder containing .npy files
base_folder = 'D:\Preprocessed_Unity_Dataset'

for root, dirs, files in os.walk(base_folder):
    for file in files:
        if file.endswith('.npy'):
            try:
                # Load the numpy file
                file_path = os.path.join(root, file)
                data = np.load(file_path, allow_pickle=True)

                # Assign a label of 1
                if isinstance(data, dict) and 'array' in data:
                    data['label'] = 1
                else:
                    data = {'array': data, 'label': 1}

                # Save the updated data back to the numpy file
                np.save(file_path, data)
                print(f"Labeled file {file} with label 1")
            except Exception as e:
                print(f"Error processing file {file}: {e}")
