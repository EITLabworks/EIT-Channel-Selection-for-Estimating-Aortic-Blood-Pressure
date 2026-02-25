import numpy as np
from os.path import join
from glob import glob
from scipy import signal
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder
import h5py
import json

np.random.seed(1)


# Loading npz
def load_minmeanmax(X: list, y: list, pigs: list, path: str,para_len:int,  vsig=[], ventkey="none"):
    """
    Load aorta pressure (and eit) signals from npz files in a given path

    X, y:      list to append the loaded data to
    path:      path to directory with npz files
    """
    print(f"Loading data from {path}")
    files =list(sorted( glob(join(path, "*.npz"), recursive=True)))
    if len(files) == 0:
        raise Exception("No npz files found in directory")


    for filepath in files:
        tmp = np.load(filepath)
        X.append(tmp["eit_v"])
        y.append([tmp["min_val"], tmp["mean_val"], tmp["max_val"]])
        pigs.append(tmp["data_info"])
        if ventkey != "none":
            vsig.append(tmp[ventkey])

    return X, y, pigs, vsig, "MinMeanMax"



# ------------------------------------------------------------------------------------------------------------------- #
def load_paras(X: list, y: list, pigs: list, path: str, para_len: int, vsig=[], ventkey="none"):
    """
    Load aorta pressure (and eit) signals from npz files in a given path

    X, y:      list to append the loaded data to
    path:      path to directory with npz files
    """
    print(f"Loading data from {path}")
    files = glob(join(path, "*.npz"), recursive=True)
    files = list(sorted(files))
    if len(files) == 0:
        raise Exception("No npz files found in directory")

    for filepath in files:
        tmp = np.load(filepath)
        X.append(tmp["eit_v"])
        y.append(tmp["aorta_para"])
        pigs.append(tmp["data_info"])
        if len(y[-1]) != para_len:
            if len(y[-1]) < para_len:
                p = np.append(y[-1], np.zeros(para_len - len(y[-1])))
                y[-1] = p
            else:
                y[-1] = y[-1][:para_len]
        paraType = str(tmp["para_type"])
        if ventkey != "none":
            vsig.append(tmp[ventkey])

    return X, y, pigs, vsig, paraType



def make_ventkey(venttype):
    if venttype == "none":
        return venttype
    elif venttype == "middle":
        return "vent_midseg"
    else:       # for start
        return "vent_startseg"




# ------------------------------------------------------------------------------------------------------------------- #
def reload_aorta_segs_from_piginfo(path_data, piglist, bRemovedMin=False,bResampled=False, iResampleLen=1024, sNormAorta="none"):
    """
    Reloads original aorta segments from data
    :param path_data: Path for data
    :param piglist: list containing with explicit segments should be loaded [[Pig, Block,Study,  Startindex, Segmentlength], ...]
    :param bRemovedMin:
    :param bResampled: If segements should be resampled
    :param iResampleLen: If Resampling, the desired length
    :return: Reloaded segments
    """
    segs = []
    for c in piglist:
        p = c[0]
        b = c[1]
        fpath = path_data + p + "_kalibrierteRuhephasen.mat"
        f = h5py.File(fpath, 'r')
        data = f.get('PigData')
        try:
            aorta = np.array(data[p][b]["Aorta"]).flatten()
            segs.append(aorta[int(c[3]):int(c[3]) + int(float(c[4]))])
        except:
            print("Segment not found: "+str(c))

           # segs.append(np.array([None]))

    if bResampled:
        segs= resample_aorta(segs, iResampleLen)

    if sNormAorta =="formnorm":
        for k in range(len(segs)):
            segs[k] = (segs[k]-np.min(segs[k])) / (np.max(segs[k]) - np.min(segs[k]))

    return segs



# ------------------------------------------------------------------------------------------------------------------- #
def resample_aorta(y: list, aorta_length: int):
    """
    Resample aorta segments to aorta_length
    """
    y = [signal.resample(sample, aorta_length) for sample in y]
    return y