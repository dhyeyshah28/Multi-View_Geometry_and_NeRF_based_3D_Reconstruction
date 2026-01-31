import matplotlib.pyplot as plt
import numpy as np
import cv2

def plot_lines(lines, h, w):
    """Utility function to plot lines on an image."""
    for i in range(lines.shape[0]):
        if abs(lines[i, 0] / lines[i, 1]) < 1:
            # Case where the line is not too steep, we plot points based on width.
            y0 = -lines[i, 2] / lines[i, 1]
            yw = y0 - w * lines[i, 0] / lines[i, 1]
            plt.plot([0, w], [y0, yw], 'r')
        else:
            # Steep line case, we plot points based on height.
            x0 = -lines[i, 2] / lines[i, 0]
            xh = x0 - h * lines[i, 1] / lines[i, 0]
            plt.plot([x0, xh], [0, h], 'r')

def plot_epipolar_lines(image1, image2, uncalibrated_1, uncalibrated_2, E, K, plot=True):
    """Plots the epipolar lines on the images given matching points, Essential matrix E, and intrinsic matrix K."""
    

    # Compute the Fundamental matrix F from the Essential matrix E and the intrinsic matrix K
    K_inv = np.linalg.inv(K)
    F = K_inv.T @ E @ K_inv

    # Calculate epipolar lines for points in image 1 in image 2
    # epipolar_lines_in_2 represents lines in image2 for points in image1
    epipolar_lines_in_2 = (F @ uncalibrated_1)

    # Calculate epipolar lines for points in image 2 in image 1
    # epipolar_lines_in_1 represents lines in image1 for points in image2
    epipolar_lines_in_1 = (F.T @ uncalibrated_2)

    if plot:
        # Plotting the epipolar lines on the first image
        plt.figure(figsize=(12, 6))
        ax1 = plt.subplot(1, 2, 1)
        ax1.set_xlim([0, image1.shape[1]])
        ax1.set_ylim([image1.shape[0], 0])
        plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
        plot_lines(epipolar_lines_in_1, image1.shape[0], image1.shape[1])
        plt.scatter(uncalibrated_1[:, 0], uncalibrated_1[:, 1], c='yellow', s=15)

        # Plotting the epipolar lines on the second image
        ax2 = plt.subplot(1, 2, 2)
        ax2.set_xlim([0, image2.shape[1]])
        ax2.set_ylim([image2.shape[0], 0])
        plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
        plot_lines(epipolar_lines_in_2, image2.shape[0], image2.shape[1])
        plt.scatter(uncalibrated_2[:, 0], uncalibrated_2[:, 1], c='yellow', s=15)

        plt.show()
    else:
        return epipolar_lines_in_1, epipolar_lines_in_2
