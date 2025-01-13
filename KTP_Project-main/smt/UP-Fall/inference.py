import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, Grayscale, ToTensor, Normalize

class FallDetectionCNN(nn.Module):
    def __init__(self):
        super(FallDetectionCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv3d(2, 64, (3, 3, 3), padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.conv2 = nn.Conv3d(64, 128, (3, 3, 3), padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.conv3 = nn.Conv3d(128, 256, (3, 3, 3), padding=1)
        self.bn3 = nn.BatchNorm3d(256)
        self.conv4 = nn.Conv3d(256, 256, (3, 3, 3), padding=1)
        self.bn4 = nn.BatchNorm3d(256)

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 2) 

    def forward(self, x):
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.max_pool3d(x, 2)
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.max_pool3d(x, 2)
        x = F.gelu(self.bn3(self.conv3(x)))
        x = F.max_pool3d(x, 2)
        x = F.gelu(self.bn4(self.conv4(x)))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = F.gelu(self.fc1(x))
        x = self.dropout1(x)
        x = F.gelu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

def load_model(model_path, device):
    model = FallDetectionCNN.to(device)
    model.load_state_dict(torch.load(model_path, map_location = device))
    model.eval()
    return model

def compute_frame_difference(prev_frame, current_frame): 
        prev_frame = cv2.normalize(src=prev_frame, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        current_frame = cv2.normalize(src=current_frame, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        frame_diff =  cv2.absdiff(prev_frame, current_frame)
    
        # frame_diff_test = cv2.GaussianBlur(frame_diff, (5, 5), 0)
        #Visualisation
        # concatenated_frames = cv2.hconcat([frame_diff, frame_diff_test])
        # cv2.imshow('Original vs Preprocessed', concatenated_frames)
        # cv2.waitKey(1)
        
        return frame_diff

def compute_optical_flow(prev_frame, current_frame, size = (102, 76)):
    flow = cv2.calcOpticalFlowFarneback(prev_frame, current_frame, None, 0.5, 3, 14, 3, 5, 1.2, 0)
    u_component, v_component = flow[..., 0], flow[..., 1]
    
    resized_u = cv2.resize(u_component, size)
    resized_v = cv2.resize(v_component, size)
    
    return resized_u, resized_v

def visualize_optical_flow(u, v):
    hsv = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(u, v)
    hsv[..., 0] = ang * 180 / np.pi / 2  # Hue (angle)
    hsv[..., 1] = 255  # Saturation
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value (magnitude)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('Optical Flow Visualization', rgb)
        
def process_video(video_path, model, device, window_size=18, overlap=9):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video file {video_path}")
    
    prev_frame = None
    optical_flow_window = []
    frame_difference = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("current frame", frame)
        cv2.waitKey(1)
    
        if prev_frame is not None:
            frame_diff = compute_frame_difference(prev_frame, frame)
            frame_difference.append(frame_diff)
            
            if (len(frame_difference) == 2):
                u, v = compute_optical_flow(frame_difference[0], frame_difference[1])
                visualize_optical_flow(u, v)  # Visualize optical flow
                optical_flow = np.stack([u, v], axis=0)
                optical_flow_window.append(optical_flow)

                if len(optical_flow_window) >= window_size:
                    input_tensor = np.array(optical_flow_window[-window_size:])[np.newaxis, ...].astype(np.float32)
                    input_tensor = torch.tensor(input_tensor).permute(0, 2, 1, 3, 4).to(device)

                    with torch.no_grad():
                        output = model(input_tensor)
                        probabilities = torch.nn.functional.softmax(output, dim=1)
                        pred_class = probabilities.argmax(dim=1)
                        print(f"Predicted Class: {pred_class.item()}, Probabilities: {probabilities.cpu().numpy()}")

                    optical_flow_window = optical_flow_window[overlap:]
                    
                frame_difference.pop(0)

        prev_frame = frame
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Break the loop if 'q' is pressed
            break

    cap.release()
    cv2.destroyAllWindows()
   
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FallDetectionCNN().to(device)
model.load_state_dict(torch.load('fall_detection_model_102x76_1to5_of.pth', map_location = device))

video_path = '../dataset/databrary-739/databrary739-Robinovitch-Falls_experienced_by_older_adult/sessions/50037-2614/2614_11182018.mp4'
process_video(video_path, model, device)