"""
Project ：EIT-Channel-Selection-for-Estimating-Aortic-Blood-Pressure
Directory:
File : plot_results.py
Author ：Patricia Fuchs
Date ：22.01.2026 09:29
"""
import pandas as pd
from src.cha_selection.visu_cha_selection import plot_bars_relevance, plot_results_mae
import matplotlib.pyplot as plt
from src.data.eit_segments import get_indices
import numpy as np


# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
plt.rcParams["font.family"] = "Times New Roman"


names_valid = ["none", "FirstHalf", "NonReziprok", "NonInjection", "NonInjectionGuard", "NonInjectionMid", "AortaIndex",
               "AortaIndexGuard",
               "CrossIndex", "CrossIndexGuard",
               "AutocorrIndex", "AutocorrIndexGuard", "FreqIndex", "VisualIndex", "VisualIndexGuard", "BolusIndex",
               "SingleBolusIndex"]

plot_dict = {"none": "None", "FirstHalf": "FirstHalf", "NonReziprok": "NonReziprok", "NonInjection": "NonInjection",
             "NonInjectionGuard": "NonInjection\nAdjacent",
             "NonInjectionMid": "NonInjection\nMid", "AortaIndex": "AortaIndex", "AortaIndexGuard": "AortaIndex\nGuard",
             "CrossIndex": "Crosscorr", "CrossIndexGuard": "Crosscorr\nGuard",
             "AutocorrIndex": "Autocorr", "AutocorrIndexGuard": "Autocorr\nGuard", "FreqIndex": "FreqIndex",
             "VisualIndex": "VisualIndex", "VisualIndexGuard": "VisualIndex\nGuard",
             "BolusIndex": "BolusIndex", "SingleBolusIndex": "SinglePig\nBolus"}



# Help Functions ----------------------------------------------------------------------------------------------------- #
def check_valid_technique(name, namesvalid):
    valid = False
    for n in namesvalid:
        if name.endswith(n):
            return True
    return valid
# ----------------------------------------------- #
def calc_relevance_xcdc(dmaes, dindex):
    relevance_xcdc = np.zeros(1024)
    mae_current = dmaes["xcdc32"]
    name_current = "xcdc32"
    factor = 3
    for n, m in dmaes.items():
        if m < mae_current:
            print(n, m)
            idx_curr = dindex[name_current]
            idx = dindex[n]
            w = int(name_current[4:])
            for k in idx:
                if k not in idx_curr:
                    relevance_xcdc[k] = 1 * factor
            factor -= 1
        mae_current = m
        name_current = n
    return relevance_xcdc / 10


# Load Structured and Data-Driven Results ---------------------------------------------------------------------------- #
df = pd.read_excel("/results/Results_Channel_Selection.xlsx")

mae = []
dmae = {}
for idx, row in df.iterrows():
    if type(row["Technique"]) == str:
        if check_valid_technique(row["Technique"], names_valid):
            mae.append(row["MAE"])
            dmae[row["Technique"]] = row["MAE"]

print_names = {}
maes = [list(dmae.values())]
names = list(dmae.keys())
names_to_plot = []
for n in names:
    names_to_plot.append(plot_dict[n])


baselineidx = 0
idx_list = [[]]
idx_dict = {}
for k in range(1, len(names)):
    idx_list.append(get_indices(names[k]))
    idx_dict[names[k]] = idx_list[-1]


# Load XCDC Results and Indices -------------------------------------------------------------------------------------- #
dmae_xcdc = {}
dindex_xcdc = {}
for idx, row in df.iterrows():
    if type(row["Technique"]) == str:
        if row["Technique"].startswith("xcdc"):
            dmae_xcdc[row["Technique"]] = row["MAE"]
            dindex_xcdc[row["Technique"]] = get_indices(row["Technique"])

r = calc_relevance_xcdc(dmae_xcdc, dindex_xcdc)
nxcdc = list(dmae_xcdc.keys())
mxcdc = list(dmae_xcdc.values())
mxcdc = [maes[0][0]] + mxcdc
nxcdc = [names_to_plot[0]] + nxcdc

# Visualization ------------------------------------------------------------------------------------------------------ #
plot_bars_relevance(maes[0][0], maes[0][1:], names[1:], idx_dict, xcdc_r=r)
plot_results_mae(maes, names, ["MAE"], "MAE", bShow=True)

plot_results_mae([mxcdc], nxcdc, ["MAE"], "MAE", bShow=True)
