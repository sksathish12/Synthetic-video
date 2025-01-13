import cv2
import os
import numpy as np
import csv
import re
from ultralytics import YOLO

class DatasetDirectoryHandler:
    def __init__(self, base_folder):
        self.base_folder = base_folder

    def get_subject_folders(self):
        return self._get_subfolders(self.base_folder)

    def get_activity_folders(self, subject_folder):
        subject_path = os.path.join(self.base_folder, subject_folder)
        return self._get_subfolders(subject_path)

    def get_trial_folders(self, subject_folder, activity_folder):
        activity_path = os.path.join(self.base_folder, subject_folder, activity_folder)
        return self._get_subfolders(activity_path)

    def get_camera_folders(self, subject_folder, activity_folder, trial_folder):
        trial_path = os.path.join(self.base_folder, subject_folder, activity_folder, trial_folder)
        return self._get_subfolders(trial_path)

    def _get_subfolders(self, folder):
        folders = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
        return sorted(folders, key=lambda x: int(re.search(r'\d+', x).group()))
    
class FrameLoader:
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder
        
    def get_video_folders(self):
        return[d for d in os.listdir(self.dataset_folder) if os.path.isdir(os.path.join(self.dataset_folder, d))]
        
    def load_frames_from_video(self, video_folder):
        image_folder = os.path.join(self.dataset_folder, video_folder)
        file_names = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')])
                
        return [(fn[:-4], cv2.imread(os.path.join(image_folder, fn), cv2.IMREAD_COLOR)) for fn in file_names]
               
class NumpyWriter:
    def __init__(self, output_folder):
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        
    def write_array(self, array, name):
        file_path = os.path.join(self.output_folder, f"{name}.npy")
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)
        np.save(file_path, array)

class YOLOv8HumanSegment:
    def __init__(self, model_name='yolov8l-seg.pt'):
        self.model = YOLO(model_name)
        
    def segment_humans(self, frame, count):
        results = self.model(frame)
        
        person_mask = None
        if results[0].masks:
            for mask, cls in zip(results[0].masks.data, results[0].boxes.cls):
                if int(cls) == 0:
                    person_mask = mask.cpu().numpy()
                    break
    
        return person_mask        
class OpticalFlowProcessor:
    def __init__(self, dataset_folder, output_folder, fps = 18, resize_dims = (38, 51)):
        self.frame_loader = FrameLoader(dataset_folder)
        self.numpy_writer = NumpyWriter(output_folder)
        self.fps = fps
        self.window_size = fps
        self.overlap = fps // 2
        self.segmenter = YOLOv8HumanSegment()
        self.missing_person_frames = []
        self.resize_dims = resize_dims
        
    def total_seconds_from_timestamp(timestamp: str) -> float:
        hours, minutes, seconds = map(float, timestamp.split('T')[1].split('_'))
        total_seconds = hours * 3600 + minutes * 60 + seconds
        return total_seconds
       
    def increment_timestamp(timestamp: str) -> str:
        date, time = timestamp.split('T')
        try:
            hours, minutes, remainder = time.split('_')
            seconds, ms = remainder.split('.')
        except ValueError:
            print(f"Error with timestamp: {time}")
            raise
        
        ms = int(ms)
        seconds = int(seconds)
        minutes = int(minutes)
        hours = int(hours)
        
        ms += 500000
        if ms >= 1000000:
            ms -= 1000000
            seconds += 1
        
        if seconds >= 60:
            seconds -= 60
            minutes += 1
        
        if minutes >= 60:
            minutes -= 60
            hours += 1

        time_str = f"{hours:02}_{minutes:02}_{seconds:02}.{ms:06}"
        return f"{date}T{time_str}"
    
    def process_video(self, video_folder):
        frames = self.frame_loader.load_frames_from_video(video_folder)
        i = 0
        count = 0
        num_frames_in_window = int(self.fps)
        overlap_frames = int(self.overlap)

        last_frame_time = frames[-1][0]
        last_frame_seconds = OpticalFlowProcessor.total_seconds_from_timestamp(last_frame_time)
        timestamp = frames[i][0]

        while i < len(frames) - num_frames_in_window:
            window_end = min(i + num_frames_in_window, len(frames))
            processed_window = []

            for j in range(i, window_end):
                frame_name = frames[j][0]
                frame = frames[j][1]
                segmentation_mask = self.segmenter.segment_humans(frame, count)
                
                if segmentation_mask is None:
                    self.missing_person_frames.append(frame_name)
                    segmentation_mask = np.zeros_like(frame[..., 0])  # Create a black mask
                
                masked_frame = cv2.bitwise_and(frame, frame, mask=segmentation_mask.astype(np.uint8))
                masked_frame = cv2.resize(masked_frame, self.resize_dims)
                processed_window.append(masked_frame)
                
            window_name = f"{video_folder}_{timestamp}"
            self.numpy_writer.write_array(processed_window, window_name)        
            timestamp = OpticalFlowProcessor.increment_timestamp(timestamp)
            next_increment_seconds = OpticalFlowProcessor.total_seconds_from_timestamp(timestamp)

            if last_frame_seconds - next_increment_seconds < 1.0:
                break

            i += (num_frames_in_window - overlap_frames)


    def run(self):
        dir_handler = DatasetDirectoryHandler(self.frame_loader.dataset_folder)
        
        for subject_folder in dir_handler.get_subject_folders():
            for activity_folder in dir_handler.get_activity_folders(subject_folder):
                for trial_folder in dir_handler.get_trial_folders(subject_folder, activity_folder):
                    for camera_folder in dir_handler.get_camera_folders(subject_folder, activity_folder, trial_folder):
                        print(f"Processing video: {camera_folder}")
                        self.process_video(os.path.join(subject_folder, activity_folder, trial_folder, camera_folder))
            
if __name__ ==  "__main__":
    dataset_folder = 'D:/UP-FallDataset'
    output_folder  = 'D:/raw_original_38x51'
    
    processor = OpticalFlowProcessor(dataset_folder, output_folder)
    processor.run()