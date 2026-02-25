import numpy as np
import json


class AortaNormalizer:
    def __init__(self, paratype="Linear", mode="fixed", factor=1, deduction=0, lenparas=0):
        self.sParatype = paratype
        self.sMode = mode
        self._make_standard_vector(10)
        self.dLengthParas = {"Linear":42, "Full": 1024}
        self.sFixLengthModes = ["positive1024","positive","bipolar", "bipolar1200", "standard"]
        self.psFormModes= ["normPos", "normtoone", "formnorm"]
        if paratype in self.dLengthParas:
            self._make_standard_vector(self.dLengthParas[paratype])
        if paratype == "Linear" and lenparas > 0:
            self._make_standard_vector(lenparas)
        if self.sMode == "fixed":
            if self.sParatype=="Linear":
                posIndex = np.arange(0,self.iL,2)
                self.pfFactor[posIndex] = 1024
                self.pfFactor[posIndex+1] = 20
                self.pfOffset[posIndex+1] = 85

            elif self.sParatype=="Full":
                self.pfFactor[:] = 20
                self.pfOffset[:] = 85
        elif self.sMode in self.psFormModes:
            posIndex = np.arange(0, self.iL, 2)
            self.pfFactor[posIndex] = 1024  #todo or 1023
        elif self.sMode in self.sFixLengthModes:
            pass
        else:
            self.pfFactor[:] = factor
            self.pfOffset[:] = deduction

        print(f"Init Aorta normalizer with Paratype={self.sParatype} and Mode={self.sMode}.")

    def _make_standard_vector(self, L):
        self.iL = L
        self.pfFactor = np.ones(self.iL)
        self.pfOffset = np.zeros(self.iL)

    def set_factor_and_offset(self, factor, deduction):
        if self.sMode == "fixed":
            pass
        elif self.sMode in self.sFixLengthModes:
            self.pfFactor[:] = factor
            self.pfOffset[:] = deduction


    def get_factor_and_offset(self, ppfAorta):
        factor, deduction = 1,0
        return [factor, deduction]

    def normalize_forward(self, ppfAorta):
        return np.divide(np.add(ppfAorta, -1*self.pfOffset),self.pfFactor)


    def normalize_inverse(self, ppfAorta):
        return np.add(np.multiply(ppfAorta, self.pfFactor), self.pfOffset)

    def reset_normalizer(self, L):
        self._make_standard_vector(L)

    def save_paras_to_json(self, path):
        d = {"Paratype": self.sParatype, "Mode": self.sMode, "iL": self.iL, "pFactor": list(self.pfFactor), "pOffset": list(self.pfOffset)}
        with open(f'{path}/aorta_normalizer.json', "w") as text_file:
            json.dump(d, text_file)
    def load_paras_from_json(self, path):
        with open(path+"/aorta_normalizer.json", "r") as text_file:
            d = json.load(text_file)
            self.iL = d["iL"]
            self.pFactor = np.array(d["pFactor"])
            self.pOffset = np.array(d["pOffset"])
            self.psParatype = d["Paratype"]
            self.sMode = d["Mode"]
        print(f"Aorta normalizer loaded from {path} with Paratype={self.sParatype} and Mode={self.sMode}.")




class NormFactors:
    def __init__(self, mode , deduction, factor):
        self.sMode= mode
        self.deduction = deduction
        self.factor = factor
        self.pTrainSpecs = [False, deduction, factor]
        self.pTestSpecs = [True, deduction, factor]
        self.InvSpecs = [True, deduction, factor]




