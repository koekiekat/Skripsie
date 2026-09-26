import numpy as np
from scipy.spatial import distance as dist
from numba import njit

def dtw_calc_new(template_feat, comparison_feat, metric="cosine"):
    x_seq = np.asarray(template_feat).T      # (n_frames, n_features)
    y_seq = np.asarray(comparison_feat).T

    dist_mat = dist.cdist(x_seq, y_seq, metric)
    cost_mat = dp(dist_mat)
    return cost_mat[-1, -1] / (x_seq.shape[0] + y_seq.shape[0])


def batched_dtw_costs_for_template_new(template_feat, feat_batch, metric="cosine"):
    x_seq = np.asarray(template_feat).T
    n_windows, n_feat, n_frames = feat_batch.shape

    y_all = feat_batch.transpose(0, 2, 1).reshape(-1, n_feat)
    dist_all = dist.cdist(x_seq, y_all, metric).reshape(x_seq.shape[0], n_windows, n_frames)

    costs = np.empty(n_windows)
    for w in range(n_windows):
        costs[w] = dp(dist_all[:, w, :])[-1, -1] / (x_seq.shape[0] + n_frames)
    return costs

@njit(cache=True)
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
    cost_mat = np.full((N + 1, M + 1), np.inf)
    cost_mat[0, 0] = 0.0

    # Fill the cost matrix while keeping traceback information
    #traceback_mat = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            m = min(cost_mat[i, j], cost_mat[i, j + 1], cost_mat[i + 1, j])
            cost_mat[i + 1, j + 1] = dist_mat[i, j] + m
            '''penalty = [
                cost_mat[i, j],      # match (0)
                cost_mat[i, j + 1],  # insertion (1)
                cost_mat[i + 1, j]]  # deletion (2)
            i_penalty = np.argmin(penalty)
            cost_mat[i + 1, j + 1] = dist_mat[i, j] + penalty[i_penalty]
            traceback_mat[i, j] = i_penalty'''

    # Traceback from bottom right
    '''i = N - 1
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
        path.append((i, j))'''

    # Strip infinity edges from cost_mat before returning
    cost_mat = cost_mat[1:, 1:]
    #return (path[::-1], cost_mat)
    return cost_mat