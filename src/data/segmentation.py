#  Author Patricia Fuchs
import numpy as np
import h5py

from src.data.loading_mat import load_data, load_segmentation_index
from src.data.aorta_segments import AortaSegments
from src.data.eit_segments import EITSegments

# ------------------------------------------------------------------------------------------------------------------- #
# Loading of data --------------------------------------------------------------------------------------------------- #




def load_appended_pigs_dual(pathData, pathSeg, piglist, blocklist=None, nameLoading="Aorta", iLeaveOut=1, bResampleData=False,
                            iScaleFactor=1, bCreateTimeStamps=False, normData="none"):
    eit_seglist = []
    sig2_seglist = []
    blocks_standard = ["01", "02", "03", "04", "05", "06", "07", "08", "09"]
    if blocklist is None:
        blocklist = blocks_standard
    starts = []
    timestamps = []
    metadata = []
    for pig in piglist:
        for (eit_data, signal, eit_index, eit_segs, eit_len, signal_index, signal_segs,
             signal_len, Nseg, block) in load_pig_dual(pathData, pathSeg,pig, blocks=blocklist,nametoLoad=nameLoading):
            eit_seglist = eit_seglist + eit_segs
            sig2_seglist =sig2_seglist + signal_segs
            metadata = metadata + Nseg*[pig+block]
        starts.append(len(sig2_seglist))

    starts = np.array(starts)

    EIT = EITSegments(eit_seglist, ppsInfo=metadata)

    A = AortaSegments(sig2_seglist)
    EIT.scale(iScaleFactor)

    if iLeaveOut>1:
        EIT.leave_out(iLeaveOut)
        A.leave_out(iLeaveOut)
        starts = starts//iLeaveOut

    if bResampleData:
        EIT.resample(64)
        A.resample(1024)

    if normData!= "none":
        EIT.make_array()
        EIT.normalize(normData)
        eit_seglist = np.array(eit_seglist)

    EIT.cast_float32()
    A.cast_float32()

    return EIT.pppfEIT, A.ppfAorta, starts



# ------------------------------------------------------------- #
def load_pig_dual(datapath, segpath, pnum, blocks=None, nametoLoad="Aorta", bOffset=False, bQCheck=True):
    fOrigin = "_kalibrierteRuhephasen.mat"
    standard_blocks = ["01", "02", "03", "04", "05", "06", "07", "08", "09"]
    if blocks == None:
        blocks = standard_blocks

    fpath = datapath + pnum+fOrigin
    print("Loading data from " +fpath)
    f = h5py.File(fpath, 'r')

    fSegname = segpath + pnum + "_Ruhephasen_HB_Segments.mat"
    fSeg = h5py.File(fSegname, "r")

    for b in blocks:
        block = "Block"+b
        eit_data, signal, eit_index, eit_segs, eit_len, Nseg, signal_index, signal_segs, signal_len, Nseg\
            = load_block_dual(pnum, block, f,fSeg,nameLoading=nametoLoad, bWithOffset=bOffset,bQCheck=bQCheck)
        if eit_index.any() != None:
            yield eit_data, signal, eit_index, eit_segs, eit_len, signal_index, signal_segs, signal_len, Nseg, block
        else:
            continue


# ------------------------------------------------------------- #
def load_block_dual(pnum, block, f, fSeg, bLP= False, bHP=False, bWithOffset= False, nameLoading="Aorta", bQCheck=True):
    # only one block ,load data dual  , evtl. preprocdata, segments data, use eitoffsets, qchecks, return
    signal, fs_ecg = load_data(f, pnum, block, nameLoading)
    signal = signal.flatten()
    b= check_if_data(signal, pnum+" - "+block)

    if not b:
        return np.array([None]), None, np.array([None]), None, None, None, None, None, None, None
    eit_data, fs_eit = load_data(f, pnum, block, "EIT_Voltages")


    # Load and apply segmentation
    signal_segs, signal_index, signal_len, eit_segs, eit_index, eit_len, Nseg = segment_block_dual(fSeg, eit_data, signal, pnum, block, 5, minlendata=150, bApplyQCheck=bQCheck)

    if signal_index.any() == None:
        return np.array([None]), None, np.array([None]), None, None, None, None, None, None, None

    return eit_data, signal, eit_index, eit_segs, eit_len, Nseg, signal_index, signal_segs, signal_len, Nseg


# ------------------------------------------------------------------------------------------------------------------- #
# Segmentation ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------- #
def segment_block_dual(fSeg, eitdata, data, pnum, block, minleneit=0, maxleneit=128,minlendata=0, maxlendata=1600, bApplyQCheck=False,):

    idx, lengths, eit_index, eit_len, rejected_index, aorta_rejected_len = load_segmentation_index(
        fSeg, pnum, block, True, False)

    b = check_if_data(idx, pnum+" - "+block)
    if not b:
        return [], np.array([None]), np.array([None]),[] ,np.array([None]), np.array([None]), 0

    eit_idx, eit_lengths= calc_eit_indexes(idx, lengths, data.shape[0], eitdata.shape[0])

    segments, Nsegments = apply_segmentation(data, idx, lengths)
    eit_segs, NsegEIT = apply_segmentation(eitdata, eit_idx, eit_lengths)

    if bApplyQCheck:
        [segments, eit_segs], [idx, eit_idx], [lengths, eit_lengths], Nsegments = apply_dual_quality_checks([segments, eit_segs], [idx, eit_idx],
                                                                      [lengths, eit_lengths], min_list=[minlendata, minleneit],
                                                                        max_list=[maxlendata, maxleneit])
    return segments, idx, lengths,eit_segs, eit_idx, eit_lengths, Nsegments






# ------------------------------------------------------------- #
def dual_quality_checks(signal_list, min_list, max_list):
    rm_idx = []

    for sig, minlen, maxlen in zip(signal_list, min_list, max_list):
        for segidx, seg in enumerate(sig):
            if seg.shape[0] < minlen or seg.shape[0] > maxlen:
                rm_idx.append(segidx)
    rm_mask = np.ones(len(signal_list[0]), dtype=bool)
    rm_mask[rm_idx] = False

    return rm_idx, rm_mask


# ------------------------------------------------------------- #
def apply_dual_quality_checks(signal_list,index_list, len_list, min_list, max_list):
    rm_idx, rm_mask = dual_quality_checks(signal_list, min_list, max_list)
    for idx in sorted(rm_idx, reverse=True):
        for sig in signal_list:
            del sig[idx]

    for i in range(len(index_list)):
        index_list[i] = index_list[i][rm_mask]
        len_list[i] = len_list[i][rm_mask]

    return signal_list, index_list, len_list, len(signal_list[0])



# ------------------------------------------------------------- #
def calc_eit_indexes(aorta_idx, aorta_len, data_len_aorta, data_len_eit, bSubtract1=False):
    if bSubtract1:
        aorta_idx = aorta_idx -1
    eit_idx = []
    eit_lengths = []

   # ds_r = data_len_aorta / data_eit.shape[0]
    ds_r = data_len_aorta / data_len_eit
    if aorta_idx.any() != None:
        for i in range(len(aorta_idx)):
                eit_ind = int(aorta_idx[i] / ds_r)
                eit_end = int((aorta_idx[i] + aorta_len[i]) / ds_r)
                eit_len = eit_end - eit_ind
                eit_idx.append(eit_ind)
                eit_lengths.append(eit_len)
        return np.array(eit_idx), np.array(eit_lengths)
    else:
        return np.array([None]), np.array([None])


# ------------------------------------------------------------- #
def apply_segmentation(data, indexes, lens):
    Nsegments = 0
    segments = []
    if indexes.any() == None:
        return segments, Nsegments
    for i in range(len(indexes)):
        s = data[indexes[i]:indexes[i] + lens[i]]
        segments.append(s)
        Nsegments += 1

    return segments, Nsegments


def check_if_data(sig, desc):
     # todo here a Bug in numpy, sig.any() == None does not work, but should
    if (sig[0] == None):
        print("\nFailed loading " +str(desc))
        return False
    else:
        print("\nStart loading "+ str(desc))
        return True


