import numpy as np
from scipy import signal
from src.data.aorta_normalizer import AortaNormalizer


# ------------------------------------------------------------------------------------------------------------------- #
class AortaSegments:
    def __init__(self, aortasegs, desc="Full", bResampled:bool=False, ppiSegLen= None, ppiSegIndex = None, ppsInfo = [None]):
        self.sDesc= desc
        self.ppfAorta = aortasegs
        self.bResampled = bResampled
        self.iN = len(self.ppfAorta)
        if len(self.ppfAorta) != 0:
            self.iParaLen = len(self.ppfAorta[0])
        else:
            self.iParaLen = len(self.ppfAorta)
        self.bArray = False if type(self.ppfAorta)==list else True
        self.bNormalized = False
        self.sNormAorta = "none"
        self.cNormalizer = None
        self.ppiSegLen = ppiSegLen
        self.ppiSegIndex = ppiSegIndex
        self.ppsInfo = ppsInfo

    def return_data(self):
        return self.ppfAorta


    def set_data(self,ppiSegLen= None, ppiSegIndex = None, ppsInfo = None):
        self.ppiSegLen = ppiSegLen
        self.ppiSegIndex = ppiSegIndex
        self.ppsInfo = ppsInfo

    def check_quality(self, aorta_maxlen):
        idx = list()
        # quality checks for EIT data
        for n, aorta in enumerate(self.ppfAorta):
            if len(aorta)> aorta_maxlen:
                idx.append(n)
        return idx


    def normalize(self,normaorta, bGivenParas=False,cNormalizer=None):
        self.sNormAorta = normaorta
        if bGivenParas:
            self.cNormalizer = cNormalizer
        else:
            self.cNormalizer = AortaNormalizer(paratype="Full", mode=self.sNormAorta, lenparas=self.iParaLen)
            if self.sNormAorta != "fixed":
                [iFac, iOffset] = self.cNormalizer.get_factor_and_offset(self.ppfAorta)
        if self.sNormAorta != "none":
            self.ppfAorta = self.cNormalizer.normalize_forward(self.ppfAorta)
            self.bNormalized = True

    #todo
    def inverse_normalize(self):
        self.ppfAorta = self.cNormalizer.inverse_normalize_forward(self.ppfAorta)


    def leave_out(self, iLeaveOut):
        if iLeaveOut != 1:
            self.ppfAorta = self.ppfAorta[::iLeaveOut]
            if len(self.ppsInfo) != 1:
                self.ppsInfo = self.ppsInfo[::iLeaveOut]


    def remove_samples(self, rm_idx):
        for idx in sorted(rm_idx, reverse=True):
            del self.ppfAorta[idx]
            if len(self.ppsInfo) != 1:
                del self.ppsInfo[idx]
        self.iN = len(self.ppfAorta)

    def make_array(self):
        self.ppfAorta = np.array(self.ppfAorta)
        self.bArray = True

    def make_even_amount(self, num:int = 8):
        if self.iN % num != 0:
            n = self.iN // num
            self.ppfAorta = self.ppfAorta[:int(num * n)]
            if len(self.ppsInfo)!=1:
                self.ppsInfo = self.ppsInfo[:int(num * n)]
            self.iN = int(num * n)

    def resample(self, aorta_length: int):
        """
        Resample aorta segments to aorta_length
        """
        self.ppfAorta = [signal.resample(sample, aorta_length) for sample in self.ppfAorta]
        self.bResampled = True



    def scale(self, fScaleFactor):
        if fScaleFactor != 1:
            self.ppfAorta= [item / fScaleFactor for item in self.ppfAorta]

    def cast_float32(self):
        for j in range(len(self.ppfAorta)):
            self.ppfAorta[j] = self.ppfAorta[j].astype(np.float32)

    def append_data(self, ppfAorta, ppiSegLen=[], ppiSegIndex=[], ppsInfo=[]):
        if self.bArray and type(ppfAorta) == np.ndarray:
            self.ppfAorta = np.append(self.ppfAorta, ppfAorta, axis=0)
        elif self.bArray and type(ppfAorta) == list:
            self.ppfAorta = np.append(self.ppfAorta, np.array(ppfAorta), axis=0)
        elif type(ppfAorta) == np.ndarray:
            self.ppfAorta = self.ppfAorta + list(ppfAorta)
        else:
            self.ppfAorta = self.ppfAorta+ ppfAorta
        if len(ppiSegLen)> 0:
            self.ppiSegLen = np.append(self.ppiSegLen, ppiSegLen)
        if len(ppiSegIndex)> 0:
            self.ppiSegIndex = np.append(self.ppiSegIndex, ppiSegIndex)
        if len(ppsInfo)> 0:
            self.ppsInfo = self.ppsInfo + list(ppsInfo)
        self.iParaLen=len(self.ppfAorta[0])

