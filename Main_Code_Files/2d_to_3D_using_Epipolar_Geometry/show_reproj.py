import numpy as np
import matplotlib.pyplot as plt

def show_reprojections(image1, image2, uncalibrated_1, uncalibrated_2, P1, P2, K, T, R, plot=True):
    """
    Reproject points from camera 1 to camera 2 and vice versa, then plot the results.

    Parameters:
    - image1, image2: Input images from two cameras.
    - uncalibrated_1, uncalibrated_2: The uncalibrated points in camera 1 and camera 2 coordinates.
    - P1, P2: Corresponding 3D points in camera 1 and camera 2 frames.
    - K: The intrinsic matrix for both cameras.
    - T: The translation vector from camera 1 to camera 2.
    - R: The rotation matrix from camera 1 to camera 2.
    - plot: Whether to plot the images and projections.

    Returns:
    - P1proj, P2proj: Reprojected points in camera 1 and camera 2 image coordinates.
    """

    # Reproject points from camera 1 to camera 2
    # Transform P1 to camera 2 frame: P2_cam_frame = R * P1 + T

    P1proj = np.zeros((P1.shape[0], 3))
    P2proj = np.zeros((P1.shape[0], 3))

    for i in range (P1.shape[0]):
        P1_transformed = (R @ P1[i] + T)  # Ensure that P1 is correctly transformed
        # Project to image plane using K (homogeneous coordinates)
        P1proj[i] = (K @ P1_transformed)
        # P2proj /= P2proj[:, 2].reshape(-1, 1)  # Normalize by the third coordinate

        # Reproject points from camera 2 to camera 1
        # Transform P2 to camera 1 frame: P1_cam_frame = R.T * (P2 - T)
        P2_transformed = (R.T @ P2[i] - R.T @ T)
        # Project to image plane using K (homogeneous coordinates)
        P2proj[i] = (K @ P2_transformed)
    
    if plot:
        # Plot the results
        plt.figure(figsize=(6.4*3, 4.8*3))

        # Plot for camera 1
        ax = plt.subplot(1, 2, 1)
        ax.set_xlim([0, image1.shape[1]])
        ax.set_ylim([image1.shape[0], 0])
        plt.imshow(image1[:, :, ::-1])  # Display image
        plt.plot(P2proj[:, 0], P2proj[:, 1], 'bs')  # Reprojected points in camera 1
        plt.plot(uncalibrated_1[0, :], uncalibrated_1[1, :], 'ro')  # Original points in camera 1

        # Plot for camera 2
        ax = plt.subplot(1, 2, 2)
        ax.set_xlim([0, image2.shape[1]])
        ax.set_ylim([image2.shape[0], 0])
        plt.imshow(image2[:, :, ::-1])  # Display image
        plt.plot(P1proj[:, 0], P1proj[:, 1], 'bs')  # Reprojected points in camera 2
        plt.plot(uncalibrated_2[0, :], uncalibrated_2[1, :], 'ro')  # Original points in camera 2

        plt.show()

    return P1proj, P2proj
