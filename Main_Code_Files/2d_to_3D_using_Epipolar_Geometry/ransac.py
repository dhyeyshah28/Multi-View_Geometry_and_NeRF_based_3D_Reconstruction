from cv2 import INTER_LINEAR
from lse import least_squares_estimation
import numpy as np

def ransac_estimator(X1, X2, num_iterations=60000):
    sample_size = 8
    eps = 1e-4  # Threshold for inliers
    best_num_inliers = -1
    best_inliers = None
    best_E = None

    # Define e3_cap as the skew-symmetric matrix of [0, 0, 1]^T
    e3_cap = np.array([[0, -1, 0],
                       [1,  0, 0],
                       [0,  0, 0]])

    for i in range(num_iterations):
        # Randomly sample 8 correspondences
        permuted_indices = np.random.RandomState(seed=(i*10)).permutation(np.arange(X1.shape[0]))
        sample_indices = permuted_indices[:sample_size]
        test_indices = permuted_indices[sample_size:]

        # Separate sampled and test points
        X1_sample = X1[sample_indices]
        X2_sample = X2[sample_indices]
        
        # Estimate E using the least-squares method
        E = least_squares_estimation(X1_sample, X2_sample)
        
        # Enforce the rank-2 constraint
        # U, S, Vt = np.linalg.svd(E)
        # S[2] = 0  # Set the smallest singular value to zero
        # E = U @ np.diag(S) @ Vt

        # Calculate residuals and count inliers
        inliers = sample_indices.tolist()
        # residuals = []
        for j in test_indices:
            x1 = X1[j]
            x2 = X2[j]
            
            # Calculate d1_num, d1_den, d2_num, d2_den
            d1_num = (x2.T @ E @ x1)**2
            d1_den = (np.linalg.norm(e3_cap @ E @ x1))**2
            
            d2_num = (x1.T @ E.T @ x2)**2
            d2_den = (np.linalg.norm(e3_cap @ E.T @ x2))**2
            
            # Total residual for this correspondence
            residual = (d1_num / d1_den) + (d2_num / d2_den)
            # residuals.append(residual)
            if residual < eps:
              inliers.append(j)
        
        # residuals = np.array(residuals)
        
        # # Identify inliers
        # inliers = inliers.append(np.where(residuals < eps)[0])
        inliers = np.array(inliers)

        # Update best E if current E has the most inliers
        if inliers.shape[0] > best_num_inliers:
            best_num_inliers = inliers.shape[0]
            best_E = E
            best_inliers = inliers

    return best_E, best_inliers
