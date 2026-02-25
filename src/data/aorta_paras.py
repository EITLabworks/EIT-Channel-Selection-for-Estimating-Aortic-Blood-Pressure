import numpy as np
from joblib import Parallel, delayed
from src.data.aorta_normalizer import AortaNormalizer


# ------------------------------------------------------------------------------------------------------------------- #
class AortaParameters:
    def __init__(self, aortaparas, paratype:str ,bResampled:bool=False, ppiSegLen= None, ppiSegIndex = None, ppsInfo = [None]):
        self.dTypes = {"Linear":True, "MinMeanMax": False}
        self.ppfParas = aortaparas
        self.bResampled = bResampled
        self.sParatype = paratype
        self.iN = len(self.ppfParas)
        if len(self.ppfParas) != 0:
            self.iParaLen= len(self.ppfParas[0])
        else:
            self.iParaLen = 0
        self.bArray = False if type(self.ppfParas)==list else True
        self.bNormalized = False
        self.sNormAorta = "none"
        self.cNormalizer = None
        self.ppiSegLen = ppiSegLen
        self.ppiSegIndex = ppiSegIndex
        self.ppsInfo = ppsInfo

    def return_data(self):
        return self.ppfParas


    def set_data(self,ppiSegLen= None, ppiSegIndex = None, ppsInfo = None):
        self.ppiSegLen = ppiSegLen
        self.ppiSegIndex = ppiSegIndex
        self.ppsInfo = ppsInfo

    def check_quality(self, para_maxlen):
        idx = list()
        # quality checks for EIT data
        for n, aorta in enumerate(self.ppfParas):
            if len(aorta)> para_maxlen:
                idx.append(n)
        return idx


    def normalize(self,normaorta, bGivenParas=False,cNormalizer=None):
        self.sNormAorta = normaorta
        if bGivenParas:
            self.cNormalizer = cNormalizer

        else:
            self.cNormalizer = AortaNormalizer(paratype=self.sParatype, mode=self.sNormAorta, lenparas=self.iParaLen)
            if self.sNormAorta != "fixed":
                [iFac, iOffset] = self.cNormalizer.get_factor_and_offset(self.ppfParas)
        if self.sNormAorta != "none":
            self.ppfParas= self.cNormalizer.normalize_forward(self.ppfParas)
            self.bNormalized = True


    def inverse_normalize(self):
        self.ppfParas = self.cNormalizer.inverse_normalize_forward(self.ppfParas)

    def leave_out(self, iLeaveOut):
        if iLeaveOut != 1:
            self.ppfParas = self.ppfParas[::iLeaveOut]
            if len(self.ppsInfo) != 1:
                self.ppsInfo = self.ppsInfo[::iLeaveOut]


    def remove_samples(self, rm_idx):
        for idx in sorted(rm_idx, reverse=True):
            del self.ppfParas[idx]
            if len(self.ppsInfo) != 1:
                del self.ppsInfo[idx]

        self.iN = len(self.ppfParas)

    def make_array(self):
        self.ppfParas = np.array(self.ppfParas)
        self.bArray = True

    def make_even_amount(self, num:int = 8):
        if self.iN % num != 0:
            n = self.iN // num
            self.ppfParas = self.ppfParas[:int(num * n)]
            if len(self.ppsInfo)!=1:
                self.ppsInfo = self.ppsInfo[:int(num * n)]
            self.iN = int(num * n)

    def resample(self, aorta_length: int):
        """
        Resample aorta curves represented by linear regression parameters
        :param y: list of parameter arrays
        :param aorta_length: Target length
        :return: y_resampled [num segs x aorta_length]
        """
        if self.dTypes[self.sParatype]:
            num_cores = 8
            resample_index = np.arange(0, self.iParaLen, 2)

            def worker(para_array, aorta_frame_length):
                l = para_array[-2]
                checklen = 2
                while l == 0:
                    l = para_array[-2 * checklen]
                    checklen += 1
                para_array[resample_index] = np.array(para_array[resample_index] * (aorta_frame_length - 1) / l, dtype=int)
                return para_array

            self.ppfParas = Parallel(n_jobs=num_cores)(
                delayed(worker)(para_vec, aorta_length) for para_vec in self.ppfParas
            )
            self.bResampled = True
            print("Aortic Parameters of type {0} were resampled.".format(self.sParatype))
        else:
            print("Aortic Parameters of type {0} were NOT resampled.".format(self.sParatype))


    def append_data(self, ppfAorta, ppiSegLen=[], ppiSegIndex=[], ppsInfo=[]):

        if self.bArray and type(ppfAorta) == np.ndarray:
            self.ppfParas = np.append(self.ppfParas, ppfAorta, axis=0)
        elif self.bArray and type(ppfAorta) == list:
            self.ppfParas = np.append(self.ppfParas, np.array(ppfAorta), axis=0)
        elif type(ppfAorta) == np.ndarray:
            self.ppfParas = self.ppfParas + list(ppfAorta)
        else:
            self.ppfParas = self.ppfParas+ ppfAorta
        if len(ppiSegLen)> 0:
            self.ppiSegLen = np.append(self.ppiSegLen, ppiSegLen)
        if len(ppiSegIndex)> 0:            self.ppiSegIndex = np.append(self.ppiSegIndex, ppiSegIndex)
        if len(ppsInfo)> 0:
            self.ppsInfo = self.ppsInfo + list(ppsInfo)
        self.iParaLen = len(self.ppfParas[0])

def resample_paras(ppfParas, iLen):
    num_cores = 8
    resample_index = np.arange(0, len(ppfParas[0]), 2)

    def worker(para_array, aorta_frame_length):
        l = para_array[-2]
        checklen = 2
        while l == 0:
            l = para_array[-2 * checklen]
            checklen += 1
        para_array[resample_index] = np.array(para_array[resample_index] * (aorta_frame_length - 1) / l, dtype=int)
        return para_array

    Paras = Parallel(n_jobs=num_cores)(
        delayed(worker)(para_vec, iLen) for para_vec in ppfParas
    )
    return Paras