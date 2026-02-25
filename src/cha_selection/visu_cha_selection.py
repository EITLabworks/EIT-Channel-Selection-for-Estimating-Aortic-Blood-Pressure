"""
Project ：quali_effects_eit_measurements
Directory: src/nn
File : visu_cha_selection.py
Author ：Patricia Fuchs
Date ：19.01.2026 17:08
"""
import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #


def plot_results_mae(trials, names, trialnames, mea, bSave=False, bShow=False, sSavepath="C/"):
    fig, ax = plt.subplots(figsize=(15, 8.5))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 18
    })
    fs = 18
    plt.subplots_adjust(left=0.052, top=0.936, right=0.986, bottom=0.221)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2)
    #ax.set_facecolor("whitesmoke")
    x_axis = np.arange(len(names))

    colors = ["steelblue", "maroon", "goldenrod", "midnightblue"]
    for k, t in enumerate(trials):
        ax.plot(x_axis, t, marker="o", markersize=10, color=colors[k], label=trialnames[k])

    ax.set_xticks(x_axis, names, rotation=45, fontsize=fs)
    ax.legend(loc="best", fontsize=fs)
    ax.set_ylabel(mea, fontsize=fs, loc="top")
    #  ax.set_title("Results for different EIT Channel Selection Technqiues", fontsize=22)
    ax.tick_params(axis="both", which="major", labelsize=fs)
    ax.tick_params(axis="both", which="minor", labelsize=fs - 2)

    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)

    #  plt.tight_layout()
    if bSave:
        fig.savefig(sSavepath + mea + "EITNormResults.png")
    if bShow:
        plt.show()


def plot_bars_relevance(resultnone, maes, names, indexes, xcdc_r=[None], iCha=1024, fE=0, bShow=True, bSave=False,
                        sSavepath="C/"):
    xaxis = np.arange(iCha)
    plusvalues = np.zeros(iCha)
    negvalues = np.zeros(iCha)
    sumvalues = np.zeros(iCha)
    T = len(maes)

    for j in range(len(maes)):
        if maes[j] >= resultnone - fE:
            idx = indexes[names[j]]
            for x in xaxis:
                if x not in idx:
                    plusvalues[x] += 1
        else:
            idx = indexes[names[j]]
            for x in xaxis:
                if x not in idx:
                    negvalues[x] += 1

    negvalues = -1 * negvalues
    sumvalues = negvalues + plusvalues
    negvalues, plusvalues, sumvalues = negvalues / T, plusvalues / T, sumvalues / T

    fig, ax = plt.subplots(figsize=(14, 10))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 18
    })
    fs = 18
    plt.subplots_adjust(left=0.1, top=0.99, right=0.99, bottom=0.07)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2)
    ax.tick_params(axis="both", which="major", labelsize=fs)
    ax.tick_params(axis="both", which="minor", labelsize=fs - 2)
    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.9)
    if len(xcdc_r) > 1:
        ax.bar(xaxis, xcdc_r, color="maroon", label="XCDC", alpha=0.5)
    ax.bar(xaxis, plusvalues, color="steelblue", label="Positive", alpha=0.6)
    ax.bar(xaxis, negvalues, color="goldenrod", label="Negative", alpha=0.3)
    ax.bar(xaxis, sumvalues, color="midnightblue", label="Total")

    minor_ticks = np.arange(0, 1024, 5)
    ax.set_xticks(minor_ticks, minor=True)

    minor_ticks = np.arange(-0.8, 1, 19)
    ax.set_yticks(minor_ticks, minor=True)

    ax.legend(loc="lower right", fontsize=fs)
    ax.set_ylabel("Relevance [%]", fontsize=fs, loc="top")
    ax.set_xlabel("EIT Channels", fontsize=fs, loc="right")
    #   ax.set_title("Relevance of EIT Channels based on CNN Performances", fontsize =22)
    ax.grid(lw=0.6)
    if bSave:
        fig.savefig(f"{sSavepath}BarComparison.png", )
    if bShow:
        plt.show()
