import numpy as np

# Replace this with the path to your numpy file
file_path = 'D:\\Preprocessed_Unity_Dataset\\FALL_10\\Avatar_7\\cam1_007\\window_72_90.npy'

# Load the numpy file
data = np.load(file_path, allow_pickle=True)

# Print the contents
print(data)

