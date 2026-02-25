"""
Project ：parametric_eit_nn
Directory: src/nn/cha_selection
File : utils_cha_selection.py
Author ：Patricia Fuchs
Date ：19.02.2026 14:42
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from joblib import Parallel, delayed
import json
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #


def resample_aorta(y:list, aorta_length:int):
    y = [signal.resample(sample, aorta_length) for sample in y]
    return y

def resample_eit(X: list, eit_length: int):
    num_cores = 8
    eit_frame_length = X[0].shape[1]

    def worker(eit_arr, length, eit_frame_length):
        return np.array(
            [signal.resample(eit_arr[:, j], length) for j in range(eit_frame_length)]
        ).T

    X = Parallel(n_jobs=num_cores)(
        delayed(worker)(eit_block, eit_length, eit_frame_length) for eit_block in X
    )
    return X




def norm_aorta(aorta_seg, goal_len):
    y = resample_aorta(aorta_seg, goal_len)
    y_norm = []
    for seg in y:
        mi = np.mean(seg)
        ma = np.std(seg)
        seg_norm = (seg-mi)/(ma)
        y_norm.append(seg_norm)
    return y_norm

def norm_eit(eit_seg, goal_len):
    y_res = resample_eit(eit_seg, goal_len)
    y_norm = []
    for seg in y_res:
        mi = np.mean(seg, axis=(0))
        ma = np.std(seg, axis=(0))
        seg_norm = (seg-mi)/(ma)
        y_norm.append(seg_norm)
    return y_norm


def get_highest_peaks(sig, num):
    peaks, _ = signal.find_peaks(sig, distance=10)
    vals = sig[peaks]
    zs = sorted(zip(vals, peaks), reverse=True)
    p2,p = map(list,zip(*zs))
    if len(peaks)< num:
        peaks_iden = peaks
    else:
        peaks_iden= p[:num]
    return peaks_iden


def autocorrelation(pfIn):
    # OR short term autocorrelation
    l = len(pfIn)
    pfSigConvLinear = np.convolve(pfIn, np.flip(pfIn))
    pfSigConvLinear = pfSigConvLinear[l-1:]
    return pfSigConvLinear



# ------------------------------------------------------------------ #
def short_term_zero_crossings(pfIn):
    """
    Determines the Short Term Zero Crossings (STZCR) of a given time frame.
    ==> Weighted average number of times the speech signal changes sign within the frame
    Typically 10-30 ms frames, frameshift is half of the frame
    @pfIn: Time-domain frame of length N
    @return Z = STZCR = 1/N * sum(ZCs in pfIn)
    """
    fZ = 0.0
    for k in range(1,len(pfIn)):
        fZ += np.abs(sign2(pfIn[k])-sign2(pfIn[k-1]))
    return fZ
# ------------------------------------------------------------------ #
def sign2(value):
    """
    Makes value >= 0 -> 1, x<0 -> -1
    """
    val = np.sign(value)
    if val == 0:
        return 1
    else:
        return val
# ------------------------------------------------------------------ #
def crosscorrelation(pfIn1, pfIn2):
    # OR short term autocorrelation
    l = len(pfIn1)
    pfSigConvLinear = np.convolve(pfIn1, np.flip(pfIn2))
    pfSigConvLinear = pfSigConvLinear[l-1:]
    pfSigConvLinear *= (1/l)
    return pfSigConvLinear



def save_index(smindex, numval, lim, path, filename, under, above):
    d= calc_index(smindex, numval, lim, u=under, a=above)
    with open(path+filename+".json", "w")as outfile:
        json.dump(d, outfile)

def calc_index(smindex, numvalues, limit, u=[0.2], a=[0.5,0.6,0.7,0.8]):
    b= {"numval":numvalues, "limit":limit}
    for uind in u:
        ind= []
        val = int(uind*numvalues)
        for k in range(len(smindex)):
            if smindex[k] < val:
                ind.append(k)
        b.update({"under"+str(int(100*uind)):ind})
    for aind in a:
        ind = []
        val = int(aind*numvalues)
        for k in range(len(smindex)):
            if smindex[k] > val:
                ind.append(k)
        b.update({"above"+str(int(100*aind)):ind})
    return b
