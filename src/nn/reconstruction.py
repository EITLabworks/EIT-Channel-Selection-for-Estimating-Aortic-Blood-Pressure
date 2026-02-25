import numpy as np


# ------------------------------------------------------------------------------------------------------------------- #
# Overall reconstruction -------------------------------------------------------------------------------------------- #
def reconstruct_paras(paras, paratype, segLen=1024):
    if paratype == "Linear":
        return reconstruct_lin(paras)


# ------------------------------------------------------------- #
def recon_paras_block(parablock, paratype, bUncorrected=False, segLen=1024, AN=None,Denorm="none", bReorderLinParas=False):
    recon_block = []
    if AN != None:
        parablock = AN.normalize_inverse(np.array(parablock))
    else:
        parablock = np.array(parablock)

    for u in range(len(parablock)):
        p = parablock[u]
        re = reconstruct_paras(parablock[u], paratype, segLen=segLen)
        recon_block.append(re)
    return recon_block


# ------------------------------------------------------------------------------------------------------------------- #
# Specific reconstruction ------------------------------------------------------------------------------------------- #
def reconstruct_lin(p):
    p = list(p)
    p = _check_for_appended_zeros(p)
    try:
        aorta_seg = np.zeros(int(p[-2] + 1))
        for i in range(0, len(p) - 2, 2):
            aorta_seg[int(p[i])] = p[i + 1]
            xpoints = [int(np.abs(p[i])), int(np.abs(p[i + 2]))]
            ypoints = [p[i + 1], p[i + 3]]
            # if xpoints[1] >= xpoints[0]:

            x_interp = np.arange(xpoints[0] + 1, xpoints[1])
            if len(x_interp) != 0:
                y_interp = np.interp(x_interp, xpoints, ypoints)
                aorta_seg[x_interp] = y_interp
        aorta_seg[int(xpoints[1])] = ypoints[1]
    except:
        for i in range(len(p), 2):
            p[i] = int(p[i])
    return aorta_seg


# ------------------------------------------------------------------------------------------------------------------- #
# Help functions ---------------------------------------------------------------------------------------------------- #
def _check_for_appended_zeros(p, lim=1):
    q = 2
    p[0] = 0
    while q < len(p):
        if p[q] < p[q - 2]:
            p = p[:q] + p[q + 2:]
        else:
            q += 2
    return p
