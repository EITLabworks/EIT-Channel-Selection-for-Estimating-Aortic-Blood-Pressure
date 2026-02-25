import numpy as np
import matplotlib.pyplot as plt
import h5py



# Loading of data --------------------------------------------------------------------------------------------------- #
def load_data(file, pnum, block, name):
    data = file.get('PigData')
    fs = np.array(file.get('PigData/f_ADInstruments'))
    if fs == None:
        fs = 1000.0
    try:
        data = data[pnum][block]
        loaded_data = np.array(data[name])
        if loaded_data.shape[0] == 1 or loaded_data.shape[1] == 1:
            loaded_data = loaded_data.flatten()
    except:
        loaded_data = np.array([None])

    return loaded_data, fs



# ------------------------------------------------------------- #
def load_segmentation_index(file, pnum, block, bReduceIndexMatlab=False, iSyncToUncut=0):
    """
    Returns the pre-segmented indices and segment lengths for the EIT and Aorta data
    """
    data = file.get('PigData')
    try:
        data = data[pnum]
        data = data[block]
        aorta_index = np.array(data["Aorta_Segs"], dtype=int).flatten()
        aorta_len = np.array(data["Aorta_Seglen"], dtype=int).flatten()
        eit_index = np.array(data["Heartbeat_Segs"], dtype=int).flatten()
        eit_len = np.array(data["Heartbeat_Seglen"], dtype=int).flatten()
        aorta_rejected_index = np.array(data["Aorta_Segs_Rejected"], dtype=int).flatten()
        aorta_rejected_len = np.array(data["Aorta_Seglen_Rejected"], dtype=int).flatten()

        if iSyncToUncut != 0:
            eit_index = np.add(eit_index, -(iSyncToUncut - 1))

        if bReduceIndexMatlab:
            aorta_index = np.add(aorta_index, -1)
            eit_index = np.add(eit_index, -1)
            aorta_rejected_index = np.add(aorta_rejected_index, -1)
    except:
        aorta_index = np.array([None])
        aorta_len = np.array([None])
        eit_index = np.array([None])
        eit_len = np.array([None])
        aorta_rejected_index = np.array([None])
        aorta_rejected_len = np.array([None])

    return aorta_index, aorta_len, eit_index, eit_len, aorta_rejected_index, aorta_rejected_len
