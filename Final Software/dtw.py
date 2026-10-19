import numpy as np
from scipy.spatial import distance as dist
#from numba import njit
import librosa

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
    temp = np.asarray(temp_feat).T
    stream = np.asarray(comp_feat).T
    dist_mat = np.ascontiguousarray(dist.cdist(temp, stream, metric))
    dtw_costs = librosa.sequence.dtw(C=dist_mat, subseq=True, backtrack=False)
    return dtw_costs[-1, :] / temp.shape[0]
    