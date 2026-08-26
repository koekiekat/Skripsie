import numpy as np
from scipy.spatial import distance as dist

def z_normalize_frames(seq):
    # seq shape: (n_time_frames, n_freq_bins)
    mean = seq.mean(axis=1, keepdims=True)
    std = seq.std(axis=1, keepdims=True) + 1e-8
    return (seq - mean) / std

def dtw_calc(template_Zxx, comparison_Zxx):
    x_seq = np.abs(template_Zxx).T      # shape: (n_time_frames, n_freq_bins)
    y_seq = np.abs(comparison_Zxx).T    # shape: (n_time_frames, n_freq_bins)

    dist_mat = dist.cdist(x_seq, y_seq, "cosine")
    path, cost_mat = dp(dist_mat)
    ali_cost = cost_mat[-1, -1]
    #print("Alignment cost: {:.4f}".format(ali_cost))

    M = x_seq.shape[0]
    N = y_seq.shape[0]
    norm_ali_cost = ali_cost / (M + N)
    #print("Normalized alignment cost: {:.4f}".format(norm_ali_cost))
    return norm_ali_cost

def dp(dist_mat):
    """
    Find minimum-cost path through matrix `dist_mat` using dynamic programming.

    The cost of a path is defined as the sum of the matrix entries on that
    path. See the following for details of the algorithm:

    - http://en.wikipedia.org/wiki/Dynamic_time_warping
    - https://www.ee.columbia.edu/~dpwe/resources/matlab/dtw/dp.m

    The notation in the first reference was followed, while Dan Ellis's code
    (second reference) was used to check for correctness. Returns a list of
    path indices and the cost matrix.
    """

    N, M = dist_mat.shape
    
    # Initialize the cost matrix
    cost_mat = np.zeros((N + 1, M + 1))
    for i in range(1, N + 1):
        cost_mat[i, 0] = np.inf
    for i in range(1, M + 1):
        cost_mat[0, i] = np.inf

    # Fill the cost matrix while keeping traceback information
    traceback_mat = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            penalty = [
                cost_mat[i, j],      # match (0)
                cost_mat[i, j + 1],  # insertion (1)
                cost_mat[i + 1, j]]  # deletion (2)
            i_penalty = np.argmin(penalty)
            cost_mat[i + 1, j + 1] = dist_mat[i, j] + penalty[i_penalty]
            traceback_mat[i, j] = i_penalty

    # Traceback from bottom right
    i = N - 1
    j = M - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        tb_type = traceback_mat[i, j]
        if tb_type == 0:
            # Match
            i = i - 1
            j = j - 1
        elif tb_type == 1:
            # Insertion
            i = i - 1
        elif tb_type == 2:
            # Deletion
            j = j - 1
        path.append((i, j))

    # Strip infinity edges from cost_mat before returning
    cost_mat = cost_mat[1:, 1:]
    return (path[::-1], cost_mat)