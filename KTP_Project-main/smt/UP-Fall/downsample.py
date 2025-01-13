import os
import numpy as np
import shutil

def downsample_dataset(original_folder, output_folder, remove_original):
    np.random.seed(42)
    
    dataset_info = []
    for root, _, files in os.walk(original_folder):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                data = np.load(file_path, allow_pickle=True).item()
                label = data['label']
                dataset_info.append((file_path, label))
                
    fall_samples = [info for info in dataset_info if 1 <= info[1] <= 5]
    non_fall_samples = [info for info in dataset_info if info[1] > 5]
    
    num_falls = int(len(fall_samples) * 0.8)
    
    selected_fall_indices = np.random.choice(len(fall_samples), size = num_falls, replace = False)
    selected_fall_samples = [fall_samples[i] for i in selected_fall_indices]
    selected_non_fall_indices = np.random.choice(len(non_fall_samples), size = num_falls, replace = False)
    selected_non_fall_samples = [non_fall_samples[i] for i in selected_non_fall_indices]
    
    for file_path, _ in np.concatenate((selected_fall_samples, selected_non_fall_samples)):
        relative_path = os.path.relpath(file_path, original_folder)
        new_path = os.path.join(output_folder, relative_path)
        
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        shutil.copyfile(file_path, new_path)
        
        if remove_original:
            os.remove(file_path)
        
if __name__ == "__main__":
    original_folder = 'D:\\testtt'
    output_folder = 'D:/Downsampled'
    
    print(f"Downsampling {original_folder}")
    downsample_dataset(original_folder, output_folder, remove_original = True)
    
    