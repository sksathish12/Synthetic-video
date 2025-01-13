import cv2
import os
import numpy as np
from multiprocessing import Pool

class DatasetDirectoryHandler:
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder

    def get_video_folders(self):
        return [
            f for f in os.listdir(self.dataset_folder)
            if os.path.isdir(os.path.join(self.dataset_folder, f)) and f.lower().startswith("fall")
        ]

    def get_videos(self, folder_full_path):
        video_extensions = ('.mp4', '.avi', '.mkv')
        return [
            f for f in os.listdir(folder_full_path)
            if os.path.isfile(os.path.join(folder_full_path, f)) and f.endswith(video_extensions)
        ]

    def get_fall_avatar_folders(self, fall_folder):
        return [
            f for f in os.listdir(fall_folder)
            if os.path.isdir(os.path.join(fall_folder, f)) and f.lower().startswith("avatar")
        ]

class FrameLoader:
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder

    def load_frames_from_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_index = 0

        if not cap.isOpened():
            return []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append((frame_index, gray_frame))
            frame_index += 1

        cap.release()
        return frames
class FrameDifferenceComputer:
    @staticmethod
    def compute_frame_difference(prev_frame, current_frame):
        prev_frame = cv2.normalize(src=prev_frame, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        current_frame = cv2.normalize(src=current_frame, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        frame_diff = cv2.absdiff(prev_frame, current_frame)

        frame_diff_test = cv2.GaussianBlur(frame_diff, (5, 5), 0)

        # Visualization: Concatenate the original frame difference and blurred version
        concatenated_frames = cv2.hconcat([frame_diff, frame_diff_test])

        # Resize the concatenated image to fit in the window
        resized_frames = cv2.resize(concatenated_frames, (800, 400))  # Resize as needed

        # Create a window (if not already created) and show the image
        cv2.imshow('Original vs Blurred Frame Difference', resized_frames)

        # Wait for a key press and close the window when done
        cv2.waitKey(1)

        return frame_diff_test


class OpticalFlowComputer:
    def __init__(self, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0, resize_dim=(38, 51)):
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.flags = flags
        self.resize_dim = resize_dim

    def compute_optical_flow(self, prev_frame, current_frame):
        flow = cv2.calcOpticalFlowFarneback(prev_frame, current_frame, None, self.pyr_scale, self.levels, self.winsize, self.iterations, self.poly_n, self.poly_sigma, self.flags)
        u_component = flow[..., 0]
        v_component = flow[..., 1]

        resized_u = cv2.resize(u_component, self.resize_dim)
        resized_v = cv2.resize(v_component, self.resize_dim)
        return resized_u, resized_v

class NumpyWriter:
    def __init__(self, output_folder):
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)

    def write_array(self, array, name, relative_path):
        file_path = os.path.join(self.output_folder, relative_path, f"{name}.npy")
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)
        np.save(file_path, array)

class OpticalFlowProcessor:
    def __init__(self, dataset_folder, output_folder, fps=18):
        self.frame_loader = FrameLoader(dataset_folder)
        self.numpy_writer = NumpyWriter(output_folder)
        self.fps = fps
        self.window_size = fps
        self.overlap = fps // 2
        self.optical_flow_computer = OpticalFlowComputer()

    def compute_frame_difference(self, frames, start, end):
        frame_differences = []
        for i in range(start, end - 1):
            try:
                frame_diff = FrameDifferenceComputer.compute_frame_difference(frames[i][1], frames[i + 1][1])
                frame_differences.append(frame_diff)
            except cv2.error:
                continue
        return frame_differences

    def compute_optical_flows(self, frame_differences):
        optical_flows_u = []
        optical_flows_v = []

        for x in range(len(frame_differences) - 1):
            try:
                u_component, v_component = self.optical_flow_computer.compute_optical_flow(frame_differences[x], frame_differences[x + 1])
                optical_flows_u.append(u_component)
                optical_flows_v.append(v_component)
            except cv2.error as e:
                print(f"Error processing frame {x}. Error: {e}")
                continue

        return optical_flows_u, optical_flows_v

    def process_video(self, video_path):
        try:
            # Check if the output file already exists
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            relative_path = os.path.relpath(os.path.dirname(video_path), self.frame_loader.dataset_folder)
            video_subfolder = os.path.join(relative_path, video_name)
            existing_files = os.path.join(self.numpy_writer.output_folder, video_subfolder)
            
            if os.path.exists(existing_files):
                print(f"[INFO] Skipping {video_path} as it has already been processed.")
                return

            # Process video if not already processed
            frames = self.frame_loader.load_frames_from_video(video_path)
            i = 0
            num_frames_in_window = self.fps
            overlap_frames = self.overlap

            while i < len(frames) - num_frames_in_window:
                window_end = min(i + num_frames_in_window, len(frames))
                frame_differences = self.compute_frame_difference(frames, i, window_end)
                optical_flows_u, optical_flows_v = self.compute_optical_flows(frame_differences)

                if optical_flows_u and optical_flows_v:
                    optical_flows_u_array = np.stack(optical_flows_u, axis=0)
                    optical_flows_v_array = np.stack(optical_flows_v, axis=0)

                    combined_optical_flow = np.stack([optical_flows_u_array, optical_flows_v_array], axis=-1)
                    print(f"Shape of optical_flows_u_array: {optical_flows_u_array.shape}")
                    print(f"Shape of optical_flows_v_array: {optical_flows_v_array.shape}")
                    print(f"Shape of combined_optical_flow: {combined_optical_flow.shape}")
                    window_name = f"window_{i}_{window_end}"
                    self.numpy_writer.write_array(combined_optical_flow, window_name, video_subfolder)

                i += (num_frames_in_window - overlap_frames)

            del frames  # Free memory only after the whole video has been processed
        except Exception as e:
            print(f"[ERROR] Failed to process {video_path}. Error: {e}")


    def run(self):
        dir_handler = DatasetDirectoryHandler(self.frame_loader.dataset_folder)
        fall_folders = dir_handler.get_video_folders()

        video_paths = []
        for fall_folder in fall_folders:
            fall_folder_path = os.path.join(self.frame_loader.dataset_folder, fall_folder)
            if not os.path.isdir(fall_folder_path):
                continue

            avatar_folders = dir_handler.get_fall_avatar_folders(fall_folder_path)
            for avatar_folder in avatar_folders:
                avatar_folder_path = os.path.join(fall_folder_path, avatar_folder)
                video_files = dir_handler.get_videos(avatar_folder_path)
                for video_file in video_files:
                    video_paths.append(os.path.join(avatar_folder_path, video_file))

        print(f"[INFO] Starting parallel processing with {os.cpu_count() // 2} cores...")
        with Pool(processes=os.cpu_count() // 2) as pool:
            pool.map(self.process_video, video_paths)

        print("[INFO] Finished processing all videos.")

if __name__ == "__main__":
    dataset_folder = 'D:/UNITY_FALL_DATASET'
    output_folder = 'D:/Preprocessed_Unity_Dataset'

    processor = OpticalFlowProcessor(dataset_folder, output_folder)
    processor.run()
