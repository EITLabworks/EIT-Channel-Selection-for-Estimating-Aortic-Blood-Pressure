import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy import signal
from joblib import Parallel, delayed
import json


# ------------------------------------------------------------------------------------------------------------------- #
class EITSegments:
    def __init__(self, eitsegs, bResampled:bool = False, ppiSegLen= None, ppiSegIndex = None, ppsInfo = [None]):
        self.pppfEIT = eitsegs
        self.iN = len(self.pppfEIT)
        self.iNumCha= self.pppfEIT[0].shape[1]
        self.iEITLen = 64

        self.bArray= False if type(self.pppfEIT)==list else True
        self.bResampled = bResampled
        self.bNormalized= False
        self.ppiSegLen = ppiSegLen
        self.ppiSegIndex = ppiSegIndex
        self.ppsInfo = ppsInfo

    def return_data(self):
        return self.pppfEIT

    def set_data(self,ppiSegLen= None, ppiSegIndex = None, ppsInfo = [None]):
        self.ppiSegLen = ppiSegLen
        self.ppiSegIndex = ppiSegIndex
        self.ppsInfo = ppsInfo
    
    def check_quality(self, eit_max_len:int, eit_min_len:int = 15):
        idx = list()
        # quality checks for EIT data
        for n, eit in enumerate(self.pppfEIT):
            if eit.shape[0] > eit_max_len or eit.shape[0] <= eit_min_len:
                idx.append(n)
        return idx
    
    def normalize(self, norm_eit:str = "block", pigs: np.array = np.array([None])):
        """Normalize EIT signals (z-score normalization)

        norm_eit : str with 'global' or per 'block' or 'block2' with not channel independent standard deviation

        """
        if len(pigs) != 1:
            self.ppsInfo = pigs
        if norm_eit == 'global':
            mx = np.mean(self.pppfEIT, axis=(0, 1))
            sx = np.std(self.pppfEIT, axis=(0, 1))
            self.pppfEIT = (self.pppfEIT - mx) / sx
        elif norm_eit == 'block':
            le_p = LabelEncoder()
            le_b = LabelEncoder()
            le_p.fit(self.ppsInfo[:, 0])

            for p in le_p.classes_:
                idx_p = np.where(self.ppsInfo[:, 0] == p)
                le_b.fit(np.squeeze(self.ppsInfo[idx_p, 1]))
                for b in le_b.classes_:
                    idx = np.where((self.ppsInfo[:, 0] == p) & (self.ppsInfo[:, 1] == b))
                    mx = np.mean(self.pppfEIT[idx, :, :], axis=(0, 1))
                    sx = np.std(self.pppfEIT[idx, :, :], axis=(0, 1))
                    self.pppfEIT[idx, :, :] = (self.pppfEIT[idx, :, :] - mx) / sx

        self.bNormalized = True
    
    def resample(self, eit_length: int):
        """
        Resample EIT data to eit_length
        """
        num_cores = 8
        eit_frame_length = self.pppfEIT[0].shape[1]

        def worker(eit_arr, length, eit_frame_length):
            return np.array(
                [signal.resample(eit_arr[:, j], length) for j in range(eit_frame_length)]
            ).T

        self.pppfEIT = Parallel(n_jobs=num_cores)(
            delayed(worker)(eit_block, eit_length, eit_frame_length) for eit_block in self.pppfEIT
        )
        self.bResampled = True
        self.iEITLen = eit_length

    def leave_out(self, iLeaveOut):
        if iLeaveOut != 1:
            self.pppfEIT = self.pppfEIT[::iLeaveOut]
            if len(self.ppsInfo) != 1:
                self.ppsInfo = self.ppsInfo[::iLeaveOut]
            self.iN = len(self.pppfEIT)

    def get_min_max_len(self):
        L = self.get_time_length(fs=1)
        min_len = min(L)
        max_len = max(L)
        print("Minimum length of EIT segments " + str(min_len))
        print("Maximum length of EIT segments " + str(max_len))
    
    def remove_samples(self, rm_idx):
        for idx in sorted(rm_idx, reverse=True):
            del self.pppfEIT[idx]
            if len(self.ppsInfo) != 1:
                del self.ppsInfo[idx]
        self.iN = len(self.pppfEIT)


    def make_array(self):
        self.pppfEIT = np.array(self.pppfEIT)
        self.ppsInfo = np.array(self.ppsInfo)

    
    def make_even_amount(self, num:int = 8):
        if self.iN % num != 0:
            n = self.iN // num
            self.pppfEIT = self.pppfEIT[:int(num * n)]
            if len(self.ppsInfo)!=1:
                self.ppsInfo = self.ppsInfo[:int(num * n)]
            self.iN = int(num * n)
    
    def use_index(self, sUseIndex:str = "none"):
        """
        Use only certain of the 1024 EIT channels
        :param type: Describes the type, after which is decided which indices to use
        :param X: EIT input data [segments x time per segment x 1024]
        :return: X = EIT with reduced indices [segments x time per segment x numIndices <1024]
        """
        if sUseIndex != "none":
            print("Using index style " + str(sUseIndex))
            ind = get_indices(sUseIndex)
            print(ind)
            if ind.any() != None:
                self.pppfEIT = self.pppfEIT[:, :, ind]
            self.iNumCha = len(ind)
            self.piInd = ind
    
    def zero_pad(self, ieitlen:int = 64):
        self.iEITLen = ieitlen
        self.pppfEIT = [np.concatenate((sample, np.zeros((self.iEITLen- sample.shape[0], 1024))))
            for sample in self.pppfEIT]
    
    def add_axis(self):
        self.pppfEIT = self.pppfEIT[:, :, :, np.newaxis]

    def reshape(self):
        self.pppfEIT = self.pppfEIT.reshape((self.pppfEIT.shape[0], self.pppfEIT.shape[1], 32,32,1))

    def get_time_length(self, fs = 47.6837):
        """
        Determine time length of segments
        fs:     Sampling frequency
        """
        L= []
        for sample in self.pppfEIT:
            L.append(len(sample)/ fs)
        return L

    def preproc_eitsegs(self, useReziprok, sUseIndex, bHPeit, bLPeit, bZeropadding, eitlen, bReorderMea, normeit, bWeighting, bNewshape):
        # EIT Preproc
        # FOR MERGING THE EIT_LENGTH HAS TO BE DOUBLED FROM THE CONFIG FILE; BECAUSE THE SEGMENT LENGTH IS DOUBLED


        # append zeros to examples to equalize lengths
        if bZeropadding:
            self.zero_pad(eitlen)
        else:
            self.resample(eitlen)

        self.make_array()
        self.use_index(sUseIndex)

        if normeit != 'none':
            self.normalize()

        self.add_axis()
        if bNewshape:
            self.reshape()

    def scale(self, fScaleFactor):
        if fScaleFactor != 1:
            self.pppfEIT = [item / fScaleFactor for item in self.pppfEIT]

    def cast_float32(self):
        for j in range(len(self.pppfEIT)):
            self.pppfEIT[j] = self.pppfEIT[j].astype(np.float32)








# ------------------------------------------------------------------------------------------------------------------- #
def get_indices(type):
    """
    Gets index numbers to use for EIT data
    :param type: Which selection of indices should happen
    :return: The indices to use (python counting style beginning with zero)
    """
    indexpath= "C:/Users/pfuchs/Documents/uni-rostock/python_projects/parametric_eit_nn/results/nn/indices/"
    if type == "FirstHalf":
        ind = np.arange(0, 512)
    elif type == "NonReziprok":
        ind = calc_non_reziprok_ind(32)
    elif type == "AortaIndex":
        ind = load_index(indexpath + "aorta_index.json", "aorta_index")
    elif type == "AortaIndexGuard":
        ind = load_index(indexpath + "aorta_index.json", "aorta_index_guard")
    elif type == "CrossIndex":
        ind = load_index(indexpath + "crosscorr_index.json", "above70")
    elif type == "CrossIndexGuard":
        ind = load_index(indexpath + "crosscorr_index.json", "above60")
    elif type == "FreqIndex":
        ind = load_index(indexpath + "freq_index.json", "FreqIndex")
    elif type == "AutocorrIndex":
        ind = load_index(indexpath + "autocorr_index.json", "under600")
    elif type == "AutocorrIndexGuard":
        ind = load_index(indexpath + "autocorr_index.json", "under800")
    elif type == "NonInjection":
        ind2, ind = calc_non_injection_ind()
    elif type == "NonInjectionGuard":
        ind, ind2 = calc_non_injection_ind()
    elif type == "NonInjectionMid":
        ind = calc_non_injection_ind_single("mid")
    elif type == "VisualIndex":
        ind = load_index(indexpath + "visual_index.json", "EITindex")
    elif type == "VisualIndexGuard":
        ind = load_index(indexpath + "visual_index.json", "EITindexguard")
    elif type == "SingleBolusIndex":
        ind = load_index(indexpath + "cha_selection.json", "BolusPig07Index")
    elif type == "BolusIndex":
        ind = load_index(indexpath + "cha_selection.json", "BolusCombiIndex")
    elif type.startswith("xcdc"):
        ind = load_index(indexpath +"xcdc.json", "xcdc")
        usedchas= int(type[4:])
        ind = ind[usedchas]
        return np.array(sorted(ind))
    else:
        ind = np.array([None])
    return ind


# ------------------------------------------------------------------------------------------------------------------- #
def calc_non_reziprok_ind(n, bKeepInj=True):
    """
    Calc indices for only non-reziprok channels
    :param n: number of electrodes
    :return: array with indices
    """
    l = []
    if bKeepInj:
        start = 0
    else:
        start = 1
    for i in range(32):
        for k in range(start, n):
            number = int(i * 32 + k)
            l.append(number)
        start += 1
    l = np.array(l)
    return l


# ------------------------------------------------------------------------------------------------------------------- #
def calc_non_injection_ind():
    """
    Calc indices for only measurement=non-injeciton channels
    :return: array with indices, and array with guard indices
    """
    injection_ind = []
    injection_ind_guard = []
    for i in range(32):
        offset = i * 32
        for k in [i, i + 5, i - 5]:
            if k < 0:
                k += 32
            elif k > 31:
                k -= 32
            injection_ind.append(offset + k)
            injection_ind_guard.append((offset + k))
        for k in [i - 1, i + 1, i + 6, i + 4, i - 6, i - 4]:
            if k < 0:
                k += 32
            elif k > 31:
                k -= 32
            injection_ind_guard.append((offset + k))

    injection_ind = sorted(injection_ind, reverse=True)
    injection_ind_guard = sorted(injection_ind_guard, reverse=True)
    inj_ind = list(np.arange(0, 1024))
    inj_ind_guard = list(np.arange(0, 1024))
    for j in injection_ind:
        del inj_ind[j]
    for j in injection_ind_guard:
        del inj_ind_guard[j]

    return np.array(inj_ind), np.array(inj_ind_guard)
# ------------------------------------------------------------------------------------------------------------------- #
def calc_non_injection_ind_single(type="mid"):
    """
    Calc indices for only measurement=non-injeciton channels
    :return: array with indices, and array with guard indices
    """
    def get_i(i, type):
        if type=="mid":
            return i
        elif type=="left":
            return i-5
        else:
            return i+5
    injection_ind = []
    for i in range(32):
        offset = i * 32
        k = get_i(i, type)
        if k < 0:
            k += 32
        elif k > 31:
            k -= 32
        injection_ind.append(offset + k)
    injection_ind = sorted(injection_ind, reverse=True)
    inj_ind = list(np.arange(0, 1024))
    for j in injection_ind:
        del inj_ind[j]

    return np.array(inj_ind)

# ------------------------------------------------------------------------------------------------------------------- #
def load_index(path, name):
    """
    Load a file with stored EIT indices
    :param path: path to loaf
    :param name: FIle name
    :return: Array of indices
    """
    #try:
    print("loading index from " + str(path))
    with open(path, "r") as file:
            config = json.load(file)
    print("not failed")
    return np.array(config[name])
   # except:
     #   return np.array([None])



