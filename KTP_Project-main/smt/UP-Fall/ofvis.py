import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

from dataset_prep_3d import OpticalFlow3DDataset
import time

def visualize_optical_flow(dataset, index, pause_time=0.5):
    data, _ = dataset[index]
    
    num_frames = data.shape[1]
    
    for frame_num in range(num_frames):
        # Extract optical flow components for the current frame
        u = data[0, frame_num, ...].numpy()
        v = data[1, frame_num, ...].numpy()

        # Convert optical flow to polar coordinates to get magnitude and angle
        magnitude, angle = cv2.cartToPolar(u, v)

        # Normalize magnitude for better visualization
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        # Set hue according to the optical flow direction
        hue = angle * 180 / np.pi / 2
        hsv = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = hue
        hsv[..., 1] = 255
        hsv[..., 2] = magnitude

        # Convert HSV to BGR for visualization
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        plt.imshow(bgr)
        plt.title(f'Colored Optical Flow - Frame {frame_num+1}/{num_frames}')
        plt.axis('off')
        plt.pause(pause_time)
        plt.clf()

        
    plt.close()



if __name__ == "__main__":
    features_path = 'preprocessed_dataset/nparray_uv_102x76_resizefirst'
    dataset = OpticalFlow3DDataset(features_path)
    
    for index in range(len(dataset)):
        # Visualize optical flow for the first example in the dataset
        visualize_optical_flow(dataset, index)
        input("Press Enter to continue")
