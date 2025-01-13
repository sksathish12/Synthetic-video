import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2

# Folder where the numpy arrays are saved
folder_path = 'D:/nparray_balanced'

def load_arrays(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.npy'):
                yield os.path.join(root, file)

def visualize_optical_flow(array_path):
    data = np.load(array_path, allow_pickle=True)
    if 'array' in data.item():
        optical_flows = data.item()['array']
    else:
        print("The 'array' key is not found in the loaded data.")
        return

    # Convert the first frame of optical flow to a color representation for visualization
    u = optical_flows[0, :, :, 0]  # U component
    v = optical_flows[0, :, :, 1]  # V component
    hsv = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(u, v)
    hsv[..., 0] = ang * 180 / np.pi / 2  # Hue (angle)
    hsv[..., 1] = 255  # Saturation
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value (magnitude)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    fig, ax = plt.subplots()
    im = ax.imshow(rgb, animated=True)
    ax.axis('off')
    plt.title('Optical Flow Visualization')

    def update_fig(i):
        u = optical_flows[i, :, :, 0]
        v = optical_flows[i, :, :, 1]
        mag, ang = cv2.cartToPolar(u, v)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        im.set_array(rgb)
        return im,

    ani = animation.FuncAnimation(fig, update_fig, frames=optical_flows.shape[0], blit=True)
    plt.show()



if __name__ == "__main__":
    for array_path in load_arrays(folder_path):
        visualize_optical_flow(array_path)
