import numpy as np

def least_squares_estimation(X1, X2):
    # Step 1: Build the A matrix from the correspondences
    N = X1.shape[0]
    A = np.zeros((N, 9))
    
    for i in range(N):
        x1, y1, _ = X1[i]
        x2, y2, _ = X2[i]
        A[i] = [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1]
    
    # Step 2: Solve for E by using SVD on A
    _, _, Vt = np.linalg.svd(A)
    E = Vt[-1].reshape(3, 3)
    
    # Step 3: Project E onto the space of essential matrices
    U, S, Vt = np.linalg.svd(E)
    S = np.diag([1, 1, 0])
    E = U @ S @ Vt
    
    return E
