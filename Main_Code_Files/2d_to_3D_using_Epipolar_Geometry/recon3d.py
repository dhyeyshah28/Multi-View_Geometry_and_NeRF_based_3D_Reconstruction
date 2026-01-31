import numpy as np

def reconstruct3D(transform_candidates, calibrated_1, calibrated_2):
  """This functions selects (T,R) among the 4 candidates transform_candidates
  such that all triangulated points are in front of both cameras.
  """

  best_num_front = -1
  best_candidate = None
  best_lambdas = None
  for candidate in transform_candidates:
    R = candidate['R']
    T = candidate['T']

    lambdas = np.zeros((2, calibrated_1.shape[0]))

    for i, (x1, x2) in enumerate(zip(calibrated_1, calibrated_2)):
            # Create the matrix A and vector b for least-squares solution
            A = np.zeros((3, 2))
            A[:, 0] = x2
            A[:, 1] = -R @ x1
            
            b = T

            # Solve for lambda1 and lambda2 using least squares
            lambda_vals, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            lambdas[0, i], lambdas[1, i] = lambda_vals
            
    num_front = np.sum(np.logical_and(lambdas[0]>0, lambdas[1]>0))

    if num_front > best_num_front:
      best_num_front = num_front
      best_candidate = candidate
      best_lambdas = lambdas
      print("best", num_front, best_lambdas[0].shape)
    else:
      print("not best", num_front)


  P1 = best_lambdas[1].reshape(-1, 1) * calibrated_1
  P2 = best_lambdas[0].reshape(-1, 1) * calibrated_2
  T = best_candidate['T']
  R = best_candidate['R']
  return P1, P2, T, R