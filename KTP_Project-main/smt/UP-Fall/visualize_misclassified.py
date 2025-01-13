import torch
import numpy as np
import cv2

def visualize_optical_flow(u_component, v_component, upscale_factor=5):
    num_frames, height, width = u_component.shape
    rgb_sequence = np.zeros((num_frames, height * upscale_factor, width * upscale_factor, 3), dtype=np.uint8)

    for i in range(num_frames):
        magnitude, angle = cv2.cartToPolar(u_component[i], v_component[i])
        
        hsv = np.zeros((height, width, 3), dtype=np.uint8)
        hsv[..., 1] = 255
        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        rgb_upscaled = cv2.resize(rgb, (width * upscale_factor, height * upscale_factor), interpolation=cv2.INTER_LINEAR)
        rgb_sequence[i] = rgb_upscaled

    return rgb_sequence

def get_misclassified_samples_for_vis(model, dataloader, device):
    model.eval()
    misclassified_samples = []
    misclassified_indices = []

    with torch.no_grad():
        for i, (batch_features, batch_labels, batch_raw_paths) in enumerate(dataloader):
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                
            outputs = model(batch_features)
            _, predicted = torch.max(outputs.data, 1)
                
            misclassified_indices_batch = (predicted != batch_labels).nonzero(as_tuple=True)[0]
                
            for idx in misclassified_indices_batch:
                global_idx = i * dataloader.batch_size + idx.item()
                misclassified_indices.append(global_idx)
                
                data = batch_features[idx].cpu().numpy()
                true_label = batch_labels[idx].item()
                predicted_label = predicted[idx].item()
                raw_frame_path = batch_raw_paths[idx]
                    
                optical_flow_u = data[0, :, :, :]
                optical_flow_v = data[1, :, :, :]
                optical_flow_rgb = visualize_optical_flow(optical_flow_u, optical_flow_v)
                
                raw_frames_data = np.load(f"{raw_frame_path}", allow_pickle=True).item()['array']
                
                misclassified_samples.append((raw_frames_data, optical_flow_rgb, true_label, predicted_label))
                     
    return misclassified_samples, misclassified_indices

def visualize_misclassified_samples(misclassified_samples):
    for raw_frames, optical_flow_rgb, true_label, predicted_label in misclassified_samples:
        while True:
            for raw_frame, optical_frame in zip(raw_frames, optical_flow_rgb):
                raw_frame_rgb = cv2.cvtColor(raw_frame, cv2.COLOR_GRAY2RGB)
                raw_frame_rgb = cv2.resize(raw_frame_rgb, (640, 480))
                optical_frame = cv2.resize(optical_frame, (640, 480))
                cv2.imshow(f"Raw Frame (True: {true_label}, Predicted: {predicted_label})", raw_frame_rgb)
                cv2.imshow(f"Optical Flow (True: {true_label}, Predicted: {predicted_label})", optical_frame)
                key = cv2.waitKey(100)  # Display each frame for 100ms
                if key == ord('n'):
                    break
                
            if key == ord('n'):
                break

        cv2.destroyAllWindows()