"""
Project ：parametric_eit_nn
Directory: src/nn
File : xcdc_channel_selection.py
Author ：Patricia Fuchs
Date ：09.02.2026 10:54
"""

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import time
import matplotlib as mpl
import json


# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, ( np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def crosscorr(x, y):
 #   L = max(len(x), len(y))
    w = np.convolve(x, np.flip(y))
    return w


def crosscorr2(x, y):
    L = max(len(x), len(y))
    w = np.convolve(x, y)
    return w[L - 1:]


def xc_similarity(x, y):
    return np.max(crosscorr(x, y))


class XCDC:
    """
    Implements the Cross-correlation based discriminant criterion for regression problems with a limited amount of regression values.
    Based on: "Cross-correlation based discriminant criterion for channel selection in motor imagery BCI systems", Yu, J. and Yu, Z.L.,
    Journal of Neural Engineering, 18, 2021.
    If the regression values are usually continuous values, they have to be transformed into integer values for a closed amount of possible values.
    """

    def __init__(self, iCha, sSavePath="C/"):
        self.iCha = iCha
        self.iN = 1000
        self.iBatchsize = self.iN
        self.pfRw = np.zeros(self.iCha)
        self.pfRb = np.zeros(self.iCha)
        self.ppiNumber = np.zeros(1)
        self.fLambda = 0.5
        self.fGamma = 1
        self.sSavePath = sSavePath
        self.pfD = np.zeros(self.iCha)
        self.piInd = np.zeros(self.iCha)

    def make_int(self, pfAorta):
        return np.array(pfAorta, dtype=int)

    def run_prenormalization(self, pppfData):
        pass

    def init_save_vectors(self):
        self.ppiNumber = np.zeros((self.iCha, self.iClasses))
        self.ppfCrossCorr = np.zeros((self.iCha, self.iClasses))

    def diff_aorta(self, val1, val2):
        return int(np.abs(val1 - val2))

    def do_frame_norm(self, pppfData):
        m = np.mean(pppfData, axis=1)
        s = np.std(pppfData, axis=1)
        for i in range(len(pppfData)):
            pppfData[i] = (pppfData[i] - m[i]) / s[i]
        return pppfData

    def save_update_current_results(self, iCounter):
        timestr = time.strftime("%Y%m%d-%H%M%S")

        title = "xcdc_"+str(int(self.fLambda*100))+"_"+ str(int(self.fGamma))
        np.savez(self.sSavePath + title+ '.npz',
                 timestamp=timestr,
                 pfRW=self.pfRw,
                 pfRb=self.pfRb,
                 ppiNumber=self.ppiNumber,
                 ppfCrossCorr=self.ppfCrossCorr,
                 iChaCurrent=iCounter,
                 fGamma=self.fGamma,
                 fLambda=self.fLambda,
                 pfD=self.pfD,
                 piInd=self.piInd, )

    def load_results(self, path,lambdaint, gammaint):
        title ="xcdc_"+str(lambdaint)+"_"+ str(gammaint)
        npz = np.load(path +title +".npz")
        self.pfRw = npz['pfRW']
        self.pfRb = npz['pfRb']
        self.ppiNumber = npz['ppiNumber']
        self.fGamma = npz['fGamma']
        self.fLambda = npz['fLambda']
        self.ppfCrossCorr = npz['ppfCrossCorr']
        self.iChaCurrent = npz['iChaCurrent']
        self.pfD = npz['pfD']
        timestamp = npz['timestamp']
        self.piInd = npz['piInd']
        self.iCha = self.ppfCrossCorr.shape[0]
        self.iClasses = self.ppfCrossCorr.shape[1]
        print(f"Results from timestamp {timestamp} at {path} was succesfully loaded. ")

    def calc_rb(self, gamma, iChaCurrent):
        return -1 * self.ppfCrossCorr[iChaCurrent, 1] / gamma * 1 / self.ppiNumber[iChaCurrent, 1]

    def calc_rb_old(self, gamma, iChaCurrent):
        idx_range = np.arange(1, self.iClasses)
        num = np.sum(self.ppiNumber[iChaCurrent, 1:])
        Rb = np.sum(self.ppfCrossCorr[iChaCurrent, 1:] * idx_range)
        return -1 * Rb / gamma * 1 / num

    def make_idx_range(self):
        idx_range = np.arange(self.iCha)
        drange, idx = zip(*sorted(zip(self.pfD, idx_range), reverse=True))
        self.piInd = np.array(idx)
        w = np.array(idx)[:5]
        print("sorted")
        print(type(sorted(w)))

    def save_json_index(self):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        d= {"xcdc": self.piInd,
            "fGamma": float(self.fGamma),
            "fLambda": float(self.fLambda),
            "timestamp":timestr}
        with open (self.sSavePath  + 'xcdc.json', 'w') as outfile:
            json.dump(d, outfile, cls=NumpyEncoder)


    def run_xcdc(self, pppfEIT, pfAorta, iStartIdx=0):
        """
        trial = frame
        """
        #todo pre norm ???
        starttime = time.time()
        self.iN = len(pppfEIT)  # number of input frames
        pfAortaInt = self.make_int(pfAorta)
        self.iClasses = np.max(pfAortaInt) - np.min(pfAortaInt) + 1
        print(f"iClasses = {self.iClasses}")
        self.init_save_vectors()
        pppfEIT = self.do_frame_norm(pppfEIT)
        for iCha in (tqdm(range(iStartIdx, self.iCha), total=self.iCha - iStartIdx)):
            for iF in range(self.iN):
                for iq in range(self.iN):
                    idx_diff = self.diff_aorta(pfAortaInt[iF,], pfAortaInt[iq])
                    S = xc_similarity(pppfEIT[iF, :, iCha], pppfEIT[iq, :, iCha])
                    self.ppiNumber[iCha, idx_diff] += 1
                    self.ppfCrossCorr[iCha, idx_diff] += S

            self.pfRw[iCha] = self.ppfCrossCorr[iCha, 0] / self.ppiNumber[iCha, 0]
            self.pfRb[iCha] = self.calc_rb_old(self.fGamma, iCha)
            self.save_update_current_results(iCha)
            self.iChaCurrent = iCha

        print(f"Finished time {time.time()-starttime}")
        self.pfD = self.pfRw * self.fLambda + (1 - self.fLambda) * self.pfRb
        self.make_idx_range()
        self.save_update_current_results(self.iCha)
        print(self.pfD)
        print("Running XCDC finished.")

    def run_xcdc_fast(self, pppfEIT, pfAorta, iStartIdx=0, iFFT= 128, iWindow= 5000):
        """
        trial = frame
        """
        # todo pre norm ???
        starttime = time.time()
        self.iN = len(pppfEIT)  # number of input frames
        pfAortaInt = self.make_int(pfAorta)
        self.iClasses = np.max(pfAortaInt) - np.min(pfAortaInt) + 1
        print(f"iClasses = {self.iClasses}")
        self.init_save_vectors()

        pppfEIT = self.do_frame_norm(pppfEIT)
        for iCha in (tqdm(range(iStartIdx, self.iCha), total=self.iCha - iStartIdx)):
            ppfSpectrumFlipped = np.fft.rfft(np.flip(pppfEIT[ :, :, iCha],axis=1), n=iFFT, axis=1)      # korrekte Achse
            for iF in range(self.iN):
                pfSpec= np.fft.rfft(pppfEIT[iF,:, iCha], n=iFFT)
                start= max(iF-iWindow, 0)
                end= min(iF+iWindow, self.iN)

                idx_diff_vec = np.abs( pfAorta[start:end]- pfAorta[iF]).astype(int)
                ppfMult = np.multiply(ppfSpectrumFlipped[start:end], pfSpec)
                ppfMult= np.max(np.fft.irfft(ppfMult, n=iFFT, axis=1),axis=1)
                idx_zero = np.where(idx_diff_vec==0)[0]
                self.ppiNumber[iCha, 0] += len(idx_zero)
                self.ppfCrossCorr[iCha, 0] +=np.sum( ppfMult[idx_zero])
                idx_nonzero = np.where(idx_diff_vec!=0)[0]
                self.ppiNumber[iCha, 1] += len(idx_nonzero)
                self.ppfCrossCorr[iCha, 1] +=np.sum( np.multiply(ppfMult[idx_nonzero], idx_diff_vec[idx_nonzero]))


            self.pfRw[iCha] = self.ppfCrossCorr[iCha, 0] / self.ppiNumber[iCha, 0]
            self.pfRb[iCha] = -1 * self.ppfCrossCorr[iCha,1] / self.fGamma * 1 / self.ppiNumber[iCha,1]
            self.save_update_current_results(iCha)
            self.iChaCurrent = iCha

        print(f"Finished time {time.time() - starttime}")
        self.pfD = self.pfRw * self.fLambda + (1 - self.fLambda) * self.pfRb
        self.make_idx_range()
        self.save_update_current_results(self.iCha)
        print(self.pfD)
        print("Running XCDC finished.")




    def plot_lambda_range(self, gamma, bShow=True, bSave=False, sSavepath="C/"):
        Rb = np.zeros(self.iCha)
        for i in range(self.iCha):
            Rb[i] = self.calc_rb(gamma, i)

        lambda_vals = np.linspace(0, 1, 11)
        res = np.zeros((lambda_vals.size, self.iCha))
        for l in range(lambda_vals.size):
            res[l, :] = self.pfRw * lambda_vals[l] + (1 - lambda_vals[l]) * Rb

        fig, ax = plt.subplots(figsize=(15, 12))
        xaxis = np.arange(1, self.iCha + 1)
        cmap = mpl.colormaps["Blues"]

        for i, l in enumerate(lambda_vals):
            ax.plot(xaxis, res[i, :], color=cmap(l), label="$\lambda$=" + str(round(l, 2)))
        fs = 18
        ax.grid(lw=0.6)
        ax.legend(fontsize=fs)
        ax.set_xlabel("EIT-Channel", fontsize=fs, loc="right")
        ax.set_ylabel("Discriminant score $D$", fontsize=fs, loc="top")
        ax.set_title("XCDC-based EIT channel selection for varying $lambda$", fontsize=24)
        plt.tight_layout()
        if bSave:
            fig.savefig(sSavepath + "lambda_range.png")
        if bShow:
            plt.show()

    def plot_gamma_range(self, lambdaval, bShow=True, bSave=False, sSavepath="C/"):
        pfGammaRange = np.arange(1, self.iClasses, 2)
        resRb = np.zeros((len(pfGammaRange), self.iCha))
        Dscore = np.zeros((pfGammaRange.size, self.iCha))
        for gidx, g in enumerate(pfGammaRange):
            for iCha in range(self.iCha):
                resRb[gidx, iCha] = self.calc_rb(g, iCha)
            Dscore[gidx] = self.pfRw * lambdaval + (1 - lambdaval) * resRb[gidx]

        fig, ax = plt.subplots(figsize=(15, 12), nrows=2)
        xaxis = np.arange(1, self.iCha + 1)
        cmap = mpl.colormaps["Blues"]

        for i, l in enumerate(pfGammaRange):
            cval = l / self.iClasses
            ax[0].scatter(xaxis, resRb[i, :], color=cmap(cval), label="$\gamma$=" + str(round(l, 2)))
            ax[1].plot(xaxis, Dscore[i, :], color=cmap(cval), label="$\gamma$=" + str(round(l, 2)))
        fs = 18
        ax[0].grid(lw=0.6)
        ax[0].legend(fontsize=fs)
        ax[0].set_xlabel("EIT-Channel", fontsize=fs, loc="right")
        ax[1].set_ylabel("Discriminant score $D$", fontsize=fs, loc="top")
        ax[1].grid(lw=0.6)
        ax[1].legend(fontsize=fs)
        ax[1].set_xlabel("EIT-Channel", fontsize=fs, loc="right")
        ax[0].set_ylabel("Between-class dissimilarity $R_b$", loc="top")

        ax[0].set_title("XCDC-based EIT channel selection for varying $lambda$", fontsize=24)
        plt.tight_layout()
        if bSave:
            fig.savefig(sSavepath + "gamma_range.png")
        if bShow:
            plt.show()





if __name__ == '__main__':

    idx= np.arange(5)
    pD = [5,2,6,1,4]
    drange, idx = zip(*sorted(zip(pD, idx), reverse=True))
    print(drange)
    print(idx)

    np.random.seed(42)
    ppfSim = np.random.normal(loc=50, scale=20, size=(10000, 12, 16),)
    pfAorta = np.random.normal(loc=100, scale=10, size=(10000))

    m = np.mean(ppfSim, axis=1)
    s = np.std(ppfSim, axis=1)
    for i in range(len(ppfSim)):
        ppfSim[i] = (ppfSim[i] - m[i]) / s[i]

    iCha=0
    colorS= ["blue", "red", "green", "orange", "brown"]
    iSpec = np.fft.rfft(ppfSim[0,:,iCha], n=24)
    for i in range(2):
        w = np.convolve(ppfSim[0, :, iCha], np.flip(ppfSim[i, :, iCha]))
        plt.plot(w, color= colorS[i])
        w2= np.fft.rfft(np.flip(ppfSim[i, :, iCha]), n=24)
        res= np.multiply(iSpec,w2)
        res= np.fft.irfft(res)
        plt.plot(w, color= colorS[i], ls="--", lw=3)
    plt.grid()
    plt.show()



    w= np.zeros(10)
    w2= np.array([1,5,3,1,2,1,2])
    w[w2] += 1
    print(w)

    t1= np.zeros(20)



    t2= np.zeros(20)
    t =20
    for i in range(3,15):
        t1[i] = 3
        t2[i] = t
        t-=1


    e = crosscorr2(t1,t2)
    f = crosscorr(t1, t2)
    plt.plot(e,color="blue", label= "faltung")
    plt.plot(f,color="red", label= "crosscorr")
    plt.show()



    f = [270532608]
    f2 = np.array(f, dtype=np.int16)

    w = np.zeros((5, 4))
    w2 = np.ones((6, 4))
    print(np.append(w, w2, axis=0))
    print(f2)
    xaxis = np.linspace(1, 10, 1000)
    w1 = 5 - xaxis / 1000
    w1[800] = 5
    w2 = 3 - xaxis[:750] / 750
    w3 = crosscorr(w1, w1)
    w4 = crosscorr(w2, w2)
    w5 = crosscorr(w1, w2)
    fig, ax = plt.subplots(figsize=(15, 12))
    plt.plot(w1, color="red")
    plt.plot(w2, color="blue")
    plt.plot(w3, color="green")
    plt.plot(w4, color="orange")
    plt.plot(w5, color="magenta")
    plt.grid(lw=0.6)
    plt.show()

    xclass = XCDC(16, "C:/Users/pfuchs/Documents/uni-rostock/python_projects/parametric_eit_nn/results/")
    xclass.fLambda = 0.5
    xclass.gamma = 1

 #   xclass.run_xcdc(ppfSim, pfAorta)
    xclass.run_xcdc_fast(ppfSim, pfAorta, iFFT= 24, iWindow=132)
    xclass.run_xcdc(ppfSim, pfAorta)
    xclass.save_json_index()
    xclass.make_idx_range()
    xclass.plot_lambda_range(1, )
    xclass.plot_gamma_range(0.5)
