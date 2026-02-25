""" Author = Patricia Fuchs"""
import numpy as np
from os.path import join

np.random.seed(1)
from src.data.aorta_paras import AortaParameters
from src.data.eit_segments import EITSegments
# ------------------------------------------------------------------------------------------------------------------- #
# Init
d = {"paras": load_paras, "minmeanmax":load_minmeanmax}
aorta_class = {"paras": AortaParameters,  "minmeanmax":AortaParameters}


# ------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
# Loading functions
def load_preprocess_segments(
        data_prefix: str,
        examples: list,
        para_len=60,
        raw=False,
        zero_padding=False,
        shuffle=True,
        eit_length=64,
        aorta_max_length=1024,
        aorta_len=1024,
        norm_aorta="none",
        norm_eit="none",
        resample_paras=False,
        sUseIndex="none",
        iLeaveOut=1,
        aorta_normalizer=None,
        loading_function="paras"
):
    """
    Loads the preprocessed NPZ files with data
    :return: pppfEITSegs = EIT data [num segments x eit_length=64 x 1024(or number indices) x 1],
             Aorta.dara =aorta data parameters [numsegments x para_len] or aorta curves [numsegments x aorta_len]
             psInfo = Pig info [numsegments x 6] with [Pig, block, index, len, study..]
             Vsig = Ventilation info if loaded (with optional Length signal) [numsegments x 1(or 2)]
             cNormalizer= Used Aortic normalizer
    """
    X, y, pigs, vsig = [], [], [], []

    # Load raw data from npz files
    loadFunction = d[loading_function]
    if loading_function == "valve":
        para_len = not zero_padding
    for folder in examples:
        X, y, pigs, vsig, para_type = loadFunction(X, y, pigs, join(data_prefix, folder), para_len, vsig=vsig, ventkey=make_ventkey(loadVent))


    # Init Data classes
    EITSegs = EITSegments(X, bResampled=False, ppsInfo=pigs)
    Aorta = aorta_class[loading_function](y, para_type, bResampled=False)    # AortaParas, AortaSegs or SingleValueSpec
    print("N = {0} data samples loaded.".format(EITSegs.iN))


    # If requested return raw data
    if raw:
        return EITSegs.pppfEIT, Aorta.return_data(), EITSegs.ppsInfo


    # Quality Checks and LeaveOut
    EITSegs.leave_out(iLeaveOut)
    Aorta.leave_out(iLeaveOut)
    EITSegs.get_min_max_len()

    rm_idx = EITSegs.check_quality(eit_max_len=eit_length, eit_min_len=15)
    rm_idx_para = Aorta.check_quality(aorta_max_length)
    rm_idx = sorted(rm_idx + rm_idx_para)
    EITSegs.remove_samples(rm_idx)
    Aorta.remove_samples(rm_idx)

    EITSegs.make_even_amount(8)
    Aorta.make_even_amount(8)

    print("After quality check and preprocessing, N = {0} data samples remain.".format(EITSegs.iN))

    #EIT PREPROC
    # Zeropadding or Resampling
    if zero_padding:
        EITSegs.zero_pad()
    else:
        EITSegs.resample(eit_length)

    EITSegs.make_array()
    EITSegs.use_index(sUseIndex)

    if norm_eit != 'none':
       EITSegs.normalize()

    EITSegs.add_axis()

    # Aorta Paras Preprocessing
    if resample_paras:
        Aorta.resample(aorta_len)
    Aorta.make_array()
    if norm_aorta != "none" and aorta_normalizer:
        Aorta.normalize(norm_aorta, bGivenParas=True, cNormalizer=aorta_normalizer)
    else:
        Aorta.normalize(norm_aorta, bGivenParas=False)

    # Create index for shuffling
    shuffle = create_shuffling_vector(shuffle, Aorta.iN, iMaxLen=68000)

    # Return EIT and aorta signals
    return EITSegs.pppfEIT[shuffle, ...], Aorta.return_data()[shuffle, ...],  EITSegs.ppsInfo[shuffle, ...], Aorta.cNormalizer



# ------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
# Help Function
def create_shuffling_vector(bShuffle, N, iMaxLen=0):
    shuffle = np.arange(N)
    if bShuffle:
        np.random.shuffle(shuffle)
    if iMaxLen > 0:
        shuffle = shuffle[:iMaxLen]
    return shuffle
