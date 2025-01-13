import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms

class CustomRandomRotation(transforms.RandomRotation):
    def __init__(self, degrees, *args, **kwargs):
        super().__init__(degrees, *args, **kwargs)
        
    def get_params(self, degrees):
        angle = 0
        while angle == 0:
            angle = float(torch.empty(1).uniform_(*degrees).item())
        return angle
    
class OpticalFlow3DDataset(Dataset):
    def __init__(self, base_folder, raw_frames_base_folder, augment = 1):
        self.base_folder = base_folder
        self.labels = [] 
        self.file_paths = [] 
        self.file_names = []
        self.raw_frames_base_folder = raw_frames_base_folder
        self.augment = augment
        
        self.flip_rotation = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1),
            CustomRandomRotation(degrees=(-10, 10))
        ])
        
        self.rotation = transforms.Compose([
            CustomRandomRotation(degrees=(-10, 10))
        ])
        
        self.horizontal_flip = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1),
        ])
        
        for root, _, files in os.walk(base_folder):
            for file in files:
                if file.endswith(".npy"):
                    file_path = os.path.join(root, file)
                    raw_frame_path = self.construct_raw_frame_path(root, file)
                    self.file_paths.append((file_path, raw_frame_path, False, False))
                    data = np.load(file_path, allow_pickle=True).item()
                    self.labels.append(data['label'])
                    if augment:
                        self.file_paths.append((file_path, raw_frame_path, True, False))
                        self.labels.append(data['label'])
                        # self.file_paths.append((file_path, raw_frame_path, True, True))
                        # self.labels.append(data['label'])
                        # self.file_paths.append((file_path, raw_frame_path, False, True))
                        # self.labels.append(data['label'])
    
    def construct_raw_frame_path(self, root, file):
        base_components = os.path.relpath(root, self.base_folder).split(os.sep)
        subject, activity, trial = base_components  
        
        raw_frame_path = os.path.join(self.raw_frames_base_folder, subject, activity, trial, file)
        return raw_frame_path
                  
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        file_path, raw_frame_path, is_flip, is_rotate = self.file_paths[idx]
        data = np.load(file_path, allow_pickle=True).item()
        
        if data['array'].ndim == 0:
            raise ValueError(f"Encountered zero-dimensional array in file: {file_path}")
        
        optical_flow_sequence = data['array']
        optical_flow_sequence = np.transpose(optical_flow_sequence, (3, 0, 1, 2))
        optical_flow_sequence = torch.tensor(optical_flow_sequence, dtype=torch.float32)
        label = int(data['label'])
        if label in range(1, 6):
            label = 1
        else:
            label = 0
            
        if is_flip and is_rotate:
            augmented_frames = [self.flip_rotation(frame) for frame in optical_flow_sequence]
        elif is_flip:
            augmented_frames = [self.horizontal_flip(frame) for frame in optical_flow_sequence]
        elif is_rotate:
            augmented_frames = [self.rotation(frame) for frame in optical_flow_sequence]
        else:
            augmented_frames = optical_flow_sequence
            
        if isinstance(augmented_frames, list):
            optical_flow_sequence = torch.stack(augmented_frames)
            
        return optical_flow_sequence, torch.tensor(label, dtype=torch.long), raw_frame_path