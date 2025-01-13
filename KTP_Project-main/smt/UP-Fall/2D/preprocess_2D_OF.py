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
    @staticmethod
    def compute_optical_flow(prev_frame, current_frame):
        prev_blurred = cv2.GaussianBlur(prev_frame, (5, 5), 0)
        curr_blurred = cv2.GaussianBlur(current_frame, (5, 5), 0)
        
        flow = cv2.calcOpticalFlowFarneback(prev_blurred, curr_blurred, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        magnitude[magnitude < 1] = 0
        # hsv = np.zeros((prev_frame.shape[0], prev_frame.shape[1], 3), dtype=np.uint8)
        # hsv[..., 1] = 255
        # hsv[..., 0] = angle * 180 / np.pi / 2
        # hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        # rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # cv2.imshow('Optical Flow', rgb)
        # cv2.waitKey(1)
        resized_magnitude = cv2.resize(magnitude, (51, 38))
        
        return resized_magnitude
    
class CSVWriter:
    def __init__(self, output_csv):
        self.output_csv = output_csv
        self.header_written = self._check_header_written()
        
    def _check_header_written(self,):
        if os.path.exists(self.output_csv):
            with open(self.output_csv, 'r') as f:
                header = f.readline()
                return bool(header)
        
        return False
    
    def write_rows(self, rows):
        with open(self.output_csv, 'a', newline='') as file:
            writer = csv.writer(file)
            for row in rows:
                writer.writerow(row)
            
class OpticalFlowProcessor:
    def __init__(self, dataset_folder, output_csv, fps = 18):
        self.frame_loader = FrameLoader(dataset_folder)
        self.csv_writer = CSVWriter(output_csv)
        self.fps = fps
        self.window_size = fps
        self.overlap = fps // 2
        
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
            hours += 1  # Assuming you don't need to roll over days here

        time_str = f"{hours:02}_{minutes:02}_{seconds:02}.{ms:06}"
        return f"{date}T{time_str}"

    
    def process_video(self, video_folder):
        frames = self.frame_loader.load_frames_from_video(video_folder)
        #print(f"First frame timestamp for {video_folder}: {frames[0][0]}")
        #print(f"Last frame timestamp for {video_folder}: {frames[-1][0]}")

        i = 0
        rows = []
        num_frames_in_window = int(self.fps)
        overlap_frames = int(self.overlap)

        last_frame_time = frames[-1][0]
        last_frame_seconds = OpticalFlowProcessor.total_seconds_from_timestamp(last_frame_time)
        timestamp = frames[i][0]

        while i < len(frames) - num_frames_in_window:
            window_end = min(i + num_frames_in_window, len(frames))
            optical_flows = []

            for j in range(i, window_end - 1):
                try:
                    magnitude = OpticalFlowComputer.compute_optical_flow(frames[j][1], frames[j + 1][1])
                    optical_flows.append(magnitude)
                except cv2.error as e:
                    print(f"Error processing frame {frames[j][0]} from video {video_folder}. Error: {e}")
                    continue

            average_magnitude = np.mean(optical_flows, axis=0)
            
            csv_timestamp = timestamp.replace('_', ':')
            row = [csv_timestamp] + average_magnitude.flatten().tolist()
            rows.append(row)

            timestamp = OpticalFlowProcessor.increment_timestamp(timestamp)
            #print(f"Incremented timestamp for {video_folder}: {timestamp}")
            next_increment_seconds = OpticalFlowProcessor.total_seconds_from_timestamp(timestamp)

            if last_frame_seconds - next_increment_seconds < 1.0:
                return rows

            i += (num_frames_in_window - overlap_frames)

        return rows

    def run(self):
        if not self.csv_writer.header_written:
            header = [['Timestamp'] + [f'C({i};{j})' for i in range(38) for j in range(51)]]
            self.csv_writer.write_rows(header)

        dir_handler = DatasetDirectoryHandler(self.frame_loader.dataset_folder)
        
        for subject_folder in dir_handler.get_subject_folders():
            for activity_folder in dir_handler.get_activity_folders(subject_folder):
                for trial_folder in dir_handler.get_trial_folders(subject_folder, activity_folder):
                    for camera_folder in dir_handler.get_camera_folders(subject_folder, activity_folder, trial_folder):
                        print(f"Processing video: {camera_folder}")
                        rows = self.process_video(os.path.join(subject_folder, activity_folder, trial_folder, camera_folder))
                        self.csv_writer.write_rows(rows) 
            
if __name__ ==  "__main__":
    dataset_folder = '../dataset/UP-Fall'
    output_csv  = 'optical_flow_features2.csv'
    
    processor = OpticalFlowProcessor(dataset_folder, output_csv)
    processor.run()