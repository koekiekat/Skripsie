import numpy as np
from scipy.spatial import distance as dist
from numba import njit
import librosa

def dtw_calc(template_feat, comparison_feat, metric="cosine"):
    x_seq = np.asarray(template_feat).T      # (n_frames, n_features)
    y_seq = np.asarray(comparison_feat).T

    dist_mat = dist.cdist(x_seq, y_seq, metric)
    cost_mat = dp(dist_mat)
    return cost_mat[-1, -1] / (x_seq.shape[0] + y_seq.shape[0])

def batched_dtw_costs_for_template(template_feat, feat_batch, metric="cosine"):
    x_seq = np.asarray(template_feat).T
    n_windows, n_feat, n_frames = feat_batch.shape

    y_all = feat_batch.transpose(0, 2, 1).reshape(-1, n_feat)
    dist_all = dist.cdist(x_seq, y_all, metric).reshape(x_seq.shape[0], n_windows, n_frames)

    costs = np.empty(n_windows)
    for w in range(n_windows):
        costs[w] = dp(dist_all[:, w, :])[-1, -1] / (x_seq.shape[0] + n_frames)
    return costs

def dtw_costs_vectorized(st_feats, mt_feats, bt_feats, window_feats, metric):
    st = [batched_dtw_costs_for_template(t, window_feats, metric) for t in st_feats]
    mt = [batched_dtw_costs_for_template(t, window_feats, metric) for t in mt_feats]
    bt = [batched_dtw_costs_for_template(t, window_feats, metric) for t in bt_feats]
    return st, mt, bt

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

def dp_new(dist_mat):
    """Global DTW cost matrix (same output as before), computed by librosa."""
    C = np.ascontiguousarray(dist_mat, dtype=np.float64)
    D = librosa.sequence.dtw(C=C, subseq=True, backtrack=False) #subseq = True / False
    return D

def dtw_calc_new(template_feat, comparison_feat, metric="cosine"):
    x_seq = np.asarray(template_feat).T      # (n_frames, n_features)
    y_seq = np.asarray(comparison_feat).T

    dist_mat = dist.cdist(x_seq, y_seq, metric)
    cost_mat = dp_new(dist_mat)
    return float(cost_mat[-1, :].min() / x_seq.shape[0])
    #return cost_mat[-1, -1] / (x_seq.shape[0] + y_seq.shape[0])

def batched_dtw_costs_for_template_new(template_feat, feat_batch, metric="cosine"):
    x_seq = np.asarray(template_feat).T
    n_windows, n_feat, n_frames = feat_batch.shape

    y_all = feat_batch.transpose(0, 2, 1).reshape(-1, n_feat)
    dist_all = dist.cdist(x_seq, y_all, metric).reshape(x_seq.shape[0], n_windows, n_frames)

    costs = np.empty(n_windows)
    for w in range(n_windows):
        costs[w] = dp_new(dist_all[:, w, :])[-1, -1] / (x_seq.shape[0] + n_frames)
    return costs

def dtw_costs_vectorized_new(st_feats, mt_feats, bt_feats, window_feats, metric):
    st = [batched_dtw_costs_for_template_new(t, window_feats, metric) for t in st_feats]
    mt = [batched_dtw_costs_for_template_new(t, window_feats, metric) for t in mt_feats]
    bt = [batched_dtw_costs_for_template_new(t, window_feats, metric) for t in bt_feats]
    return st, mt, bt

def dtw_calc_times(temp_feat, comp_feat, hop_sec, frame_sec, metric):
    temp_seq = np.asarray(temp_feat).T
    comp_seq = np.asarray(comp_feat).T

    dist_mat = dist.cdist(temp_seq, comp_seq, metric)

    #Takes single best end point. Column in last row D with lowest accumulated cost
    dtw_costs, warp_path = librosa.sequence.dtw(C = dist_mat, subseq = True, backtrack = True)

    #warp_path runs from END to START
    end_path = warp_path[0, 1] #lowest cost
    start_path = warp_path[-1, 1]

    cost = float(dtw_costs[-1, end_path] / temp_seq.shape[0])
    start_s = start_path * hop_sec - frame_sec / 2
    end_s = end_path * hop_sec + frame_sec / 2
    return cost, start_s, end_s

def dtw_curve(temp_feat, comp_feat, metric):
    T = np.asarray(temp_feat).T
    S = np.asarray(comp_feat).T
    C = np.ascontiguousarray(dist.cdist(T, S, metric))
    D = librosa.sequence.dtw(C=C, subseq=True, backtrack=False)
    return D[-1, :] / T.shape[0]