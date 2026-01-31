import numpy as np

def pose_candidates_from_E(E):
    # Perform SVD on the essential matrix E
    U, _, Vt = np.linalg.svd(E)
    
    # Define Rz for +/- pi/2
    Rz_pos = np.array([[0, -1, 0],
                       [1, 0, 0],
                       [0, 0, 1]])
    Rz_neg = Rz_pos.T  # Equivalent to Rz(-pi/2)

    # Compute rotation matrices
    R1 = U @ Rz_pos.T @ Vt
    R2 = U @ Rz_neg.T @ Vt

    # Ensure R1 and R2 are proper rotations by checking determinant
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2

    # Compute translation vectors
    T1 = U[:, 2]       # third column of U
    T2 = -U[:, 2]      # opposite of the third column of U

    # Generate the four candidate solutions
    transform_candidates = [
        {"R": R1, "T": T1},
        {"R": R2, "T": T1},
        {"R": R1, "T": T2},
        {"R": R2, "T": T2}
    ]
    
    return transform_candidates
