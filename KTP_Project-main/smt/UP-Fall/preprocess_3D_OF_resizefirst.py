import cv2
import os
import numpy as np
import csv
import re

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
                
        return [(fn[:-4], cv2.imread(os.path.join(image_folder, fn), cv2.IMREAD_GRAYSCALE)) for fn in file_names]
    
class OpticalFlowComputer:
    def __init__(self):
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=30, detectShadows=True)
        self.fgbg.setShadowValue(0)
        self.fgbg.setShadowThreshold(0.5)
        self.learning_rate = -1
        
    def display_optical_flow(u, v, window_name):
        # Convert to polar coordinates
        magnitude, angle = cv2.cartToPolar(u, v, angleInDegrees=True)
        
        # Normalize magnitude for better visualization
        normalized_magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Create HSV image
        hsv_image = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.uint8)
        hsv_image[..., 0] = (angle / 2).astype(np.uint8)  # Hue
        hsv_image[..., 1] = 255  # Saturation
        hsv_image[..., 2] = normalized_magnitude  # Value
        
        # Convert to BGR and display
        bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        cv2.imshow(window_name, bgr_image)
        cv2.waitKey(1)
        
    def compute_optical_flow(self, prev_frame, current_frame):
        # prev_frame = cv2.normalize(src=prev_frame, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # current_frame = cv2.normalize(src=current_frame, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        prev_frame_resized = cv2.resize(prev_frame, (102, 76))
        current_frame_resized = cv2.resize(current_frame, (102, 76))
        
        # fgmask_prev = self.fgbg.apply(prev_frame_resized, learningRate = self.learning_rate)
        # fgmask_curr = self.fgbg.apply(current_frame_resized, learningRate = self.learning_rate)
        
        # kernel = np.ones((4,4), np.uint8)
        # fgmask_prev = cv2.morphologyEx(fgmask_prev, cv2.MORPH_CLOSE, kernel)
        # fgmask_prev = cv2.morphologyEx(fgmask_prev, cv2.MORPH_OPEN, kernel)
        # fgmask_curr = cv2.morphologyEx(fgmask_curr, cv2.MORPH_CLOSE, kernel)
        # fgmask_curr = cv2.morphologyEx(fgmask_curr, cv2.MORPH_OPEN, kernel)
        
        # prev_frame_masked = cv2.bitwise_and(prev_frame_resized, prev_frame_resized, mask=fgmask_prev)
        # current_frame_masked = cv2.bitwise_and(current_frame_resized, current_frame_resized, mask=fgmask_curr)
        
        concatenated_frames = cv2.hconcat([current_frame_resized])
        
        cv2.imshow('Original vs Preprocessed Frame', concatenated_frames)
        cv2.waitKey(1)
        
        # prev_frame = cv2.medianBlur(prev_frame_masked, 5)
        # current_frame = cv2.medianBlur(current_frame_masked, 5)
        
        # prev_blurred = cv2.GaussianBlur(prev_frame, (5, 5), 0)
        # curr_blurred = cv2.GaussianBlur(current_frame, (5, 5), 0)
        
        flow = cv2.calcOpticalFlowFarneback(prev_frame_resized, current_frame_resized, None, 0.5, 3, 5, 3, 5, 1.2, 0)
        flow2 = cv2.calcOpticalFlowFarneback(prev_frame_resized, current_frame_resized, None, 0.5, 3, 25, 3, 5, 1.2, 0)
        u_component = flow[..., 0]
        v_component = flow[..., 1]
        u_comp = flow2[..., 0]
        v_comp = flow2[..., 1]
        # Uncomment below to view optical flow as it runs
        OpticalFlowComputer.display_optical_flow(u_component, v_component, 'Flow 1')
        OpticalFlowComputer.display_optical_flow(u_comp, v_comp, 'Flow 2')
    
        return u_component, v_component
                   
class NumpyWriter:
    def __init__(self, output_folder):
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        
    def write_array(self, array, name):
        file_path = os.path.join(self.output_folder, f"{name}.npy")
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)
        np.save(file_path, array)
        
class OpticalFlowProcessor:
    def __init__(self, dataset_folder, output_folder, fps = 18):
        self.frame_loader = FrameLoader(dataset_folder)
        self.numpy_writer = NumpyWriter(output_folder)
        self.fps = fps
        self.window_size = fps
        self.overlap = fps // 2
        self.optical_flow_computer = OpticalFlowComputer()
        
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
        #print(f"First frame timestamp for {video_folder}: {frames[0][0]}")
        #print(f"Last frame timestamp for {video_folder}: {frames[-1][0]}")

        i = 0
        num_frames_in_window = int(self.fps)
        overlap_frames = int(self.overlap)

        last_frame_time = frames[-1][0]
        last_frame_seconds = OpticalFlowProcessor.total_seconds_from_timestamp(last_frame_time)
        timestamp = frames[i][0]

        while i < len(frames) - num_frames_in_window:
            window_end = min(i + num_frames_in_window, len(frames))
            optical_flows_u= []
            optical_flows_v = []

            for j in range(i, window_end - 1):
                try:
                    u_component, v_component = self.optical_flow_computer.compute_optical_flow(frames[j][1], frames[j + 1][1])
                    optical_flows_u.append(u_component)
                    optical_flows_v.append(v_component)
                except cv2.error as e:
                    print(f"Error processing frame {frames[j][0]} from video {video_folder}. Error: {e}")
                    continue
            # Commented out to test preprocess
            # optical_flows_u_array = np.stack(optical_flows_u, axis = 0)
            # optical_flows_v_array = np.stack(optical_flows_v, axis = 0)
            
            # combined_optical_flow = np.stack([optical_flows_u_array, optical_flows_v_array], axis=-1)
            # window_name = f"{video_folder}_{timestamp}"
            # self.numpy_writer.write_array(combined_optical_flow, window_name)

            timestamp = OpticalFlowProcessor.increment_timestamp(timestamp)
            #print(f"Incremented timestamp for {video_folder}: {timestamp}")
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
    output_folder  = 'preprocessed_dataset/nparray_uv_102x76_resizefirst'
    
    processor = OpticalFlowProcessor(dataset_folder, output_folder)
    processor.run()