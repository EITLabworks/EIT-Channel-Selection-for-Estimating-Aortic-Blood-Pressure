"""
Project ：parametric_eit_nn
Directory: src/nn
File : create_cha_selection.py
Author ：Patricia Fuchs
Date ：18.02.2026 11:34
"""
from src.data import load_appended_pigs_dual

from src.data.eit_segments import calc_non_reziprok_ind, calc_non_injection_ind, calc_non_injection_ind_single, load_index
from src.cha_selection.utils_cha_selection import *
from src.data.loading_mat import load_data
import h5py
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

def run_fft(eit_seg, bSave=False):
    NFFT = 1024  # for 256, a limited resolution is already visible
    eit_seg = eit_seg / 1000000
    sr = 47.6837
    max_freq = sr / 2
    num_points = int(NFFT / 2 + 1)
    x_axis = np.linspace(0, max_freq, num_points)
    channels_valid = []
    for i in range(0, 1024):
        eit_seg_meanless = eit_seg[:, i] - np.mean(eit_seg[:, i])
        spec = np.fft.fft(eit_seg_meanless[:5 * 35], NFFT)
        peaks = get_highest_peaks(np.abs(spec[:num_points]), 3)
        peaks_freq = x_axis[peaks]
        if (peaks_freq[0] < 5 and peaks_freq[1] < 5) and peaks_freq[2] < 5:
            channels_valid.append(i)
    b = {"NFFT": NFFT, "FreqIndex": channels_valid}
    if bSave:
        with open("freq_index" + ".json", "w") as outfile:
            json.dump(b, outfile)
    return np.array(channels_valid)


# -------------------------------------------------------------------------------------------------------------------- #
# Crosscorrelation
def run_cc(eit_seg, aorta_seg, seg_len, goal_len, bSave=False):
    eit_seg = norm_eit(eit_seg, goal_len)
    aorta_seg = norm_aorta(aorta_seg, goal_len)
    cc_bigger = np.zeros((len(eit_seg), 1024))
    for i in range(len(aorta_seg)):
        for index in range(1024):
            c = crosscorrelation(aorta_seg[i], eit_seg[i][:, index])
            cc_bigger[i, index] = 1 if np.abs(c[seg_len]) > 0.6 else 0
    smindex = np.sum(cc_bigger, axis=0)
    d = calc_index(smindex, 100, 0.6, a=[0.6, 0, 7])
    if bSave:
        with open("cross_corr_index.json", "w") as outfile:
            json.dump(d, outfile)
    return np.array(d["above70"]), np.array(d["above60"])


# -------------------------------------------------------------------------------------------------------------------- #
# Autocorrlation
def run_autocorr(eit_seg, goal_len, bSave=False):
    eit_seg = norm_eit(eit_seg, goal_len)
    auto_zero = np.zeros((len(eit_seg), 1024))
    for i in range(len(eit_seg)):
        for index in range(1024):
            a = autocorrelation(eit_seg[i][:, index])
            zc = short_term_zero_crossings(a)
            auto_zero[i, index] = zc
    smindex = np.sum(auto_zero, axis=0)
    d = calc_index(smindex, 100, 6, u=[6, 8])
    if bSave:
        with open("autocorr_index.json", "w") as outfile:
            json.dump(d, outfile)
    return np.array(d["under600"]), np.array(d["under800"])


# -------------------------------------------------------------------------------------------------------------------- #
def firsthalf(N):
    return np.arange(0, N // 2)


# -------------------------------------------------------------------------------------------------------------------- #
# Main loop ------------------------------------------------------------------------------------------------------- #
psIndexlist = ["FirstHalf", "NonReziprok", "NonInjection", "NonInjectionAdjacent", "NonInjectionMid", "AortaIndex",
               "AortaIndexGuard", "CrossIndex", "CrossIndexGuard", "AutocorrIndex", "AutocorrIndexGuard", "FreqIndex",
               "VisualIndex", "VisualIndexGuard", "BolusIndex", "SingleBolusIndex"]
sIndexpath = ""
dIndex = {}
Pigs = ["P01", "P02"]
sDatapath = ""
sSegPath = ""
Nchannels = 1024
Nelectrodes = 32
bSaveIndex = False
eit_segs, aortas_segs = load_appended_pigs_dual(sDatapath, sSegPath, Pigs, blocklist=None, nameLoading="Aorta",
                                                iLeaveOut=1, bResampleData=False,
                                                iScaleFactor=1, bCreateTimeStamps=False, normData="none")
seglen = []
for k in range(len(eit_segs)):
    seglen.append(len(eit_segs[k]))

# Systemic
dIndex["FirstHalf"] = firsthalf(Nelectrodes)
dIndex["NonReziprok"] = calc_non_reziprok_ind(Nelectrodes)
idx_noninj, noninjadj = calc_non_injection_ind()
dIndex["NonInjection"] = idx_noninj
dIndex["NonInjectionAdjacent"] = noninjadj
dIndex["NonInjectionMid"] = calc_non_injection_ind_single("mid")

dIndex["AortaIndex"] = load_index(sIndexpath + "aorta_index.json", "aorta_index")
dIndex["AortaIndexGuard"] = load_index(sIndexpath + "aorta_index.json", "aorta_index_guard")

# Data-driven
cross, crossguard = run_cc(eit_segs, aortas_segs, seglen, 800, bSave=bSaveIndex)
dIndex["CrossIndex"] = cross
dIndex["CrossIndexGuard"] = crossguard

auto, autoguard = run_autocorr(eit_segs, 64, bSave=bSaveIndex)
dIndex["AutocorrIndex"] = auto
dIndex["AutocorrIndexGuard"] = autoguard

f= h5py.File("data/"+Pigs[0]+"_kalibrierteRuhepahsen", "r")
eit_data, = load_data(f, Pigs[0], "Block01", "EIT_Voltages" )

dIndex["FreqIndex"] = run_fft(eit_data, bSave=bSaveIndex)

dIndex["VisualIndex"] = load_index(sIndexpath + "Visual_index.json", "EITindex")
dIndex["VisualIndexGuard"] = load_index(sIndexpath + "Visual_index.json", "EITindexguard")
dIndex["BolusIndex"] = load_index(sIndexpath + "cha_selection.json", "BolusCombiIndex")
dIndex["SingleBolusIndex"] = load_index(sIndexpath + "cha_selection.json", "BolusPig07Index")

# Structured search
idx_xcdc = load_index(sIndexpath + "xcdc.json", "xcdc")
for i in [32, 64, 128, 256, 512]:
    dIndex["xcdc" + str(i)] = idx_xcdc[:i]
