import numpy as np
import matplotlib.pyplot as plt

from scipy.stats.stats import pearsonr

TITLE_SIZE = 24
FONT_SIZE = 18
LEGEND_SIZE = 16


# ------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------- #
# Plotter functions
# ------------------------------------------------------------------------------------------------------------- #
def plot_parameters(paras, parasori, title, Smax=100, bShow=True, bSave=False, fSavePath="C:/"):
    """
    Plots Smax  aorta parameters (representing the curve) overlaying over each other
    :param paras: Estimated aortic pressure parameters as list of segments [Num segments L x Para length]
    :param parasori: Original aortic parameters as list of segments [Num segments L x Para length]
    :param title: Plot title
    :param Smax: Max number of curves to plot
    :param bShow: if to show the plot
    :param bSave: if to save the plot
    :param fSavePath: Path to save the plot
    """
    Smax = Smax
    if len(paras) < Smax:
        Smax = len(paras)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(17, 9))
    plt.subplots_adjust(left=0.079, top=0.936, right=0.957, bottom=0.08)
    for axi in ax.flat:
        for axis in ["top", "bottom", "left", "right"]:
            axi.spines[axis].set_linewidth(2)
    for q in range(Smax):
        ax[1].plot(paras[q], "x", color="maroon", linewidth="0.8")
        ax[0].plot(parasori[q], "x", color="steelblue", linewidth="0.8")
    ax[0].grid(linewidth=0.8)
    ax[1].grid(linewidth=0.8)
    ax[1].set_xlabel("Parameter index", loc="right", fontsize=FONT_SIZE)
    ax[1].set_title("Estimated parameters " + title, fontsize=TITLE_SIZE)
    ax[0].set_title("Original parameters " + title, fontsize=TITLE_SIZE)
    if bSave:
        fig.savefig(fSavePath + title + "ParaEstimation.png")
    if bShow:
        plt.show()



# ------------------------------------------------------------------------------------------------------------- #
def plot_history(history, type, bShow=True, bSave=False, fSavePath="C:/"):
    """
    Plot Training history of the neural network
    :param history: History strucutre
    :param type: Measurement type to plot
    :param bShow: if to show the plot
    :param bSave: if to save the plot
    :param fSavePath: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(17, 9))
    plt.subplots_adjust(left=0.079, top=0.936, right=0.957, bottom=0.1)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2)
    ax.plot(history.history[type], "b", linewidth=2)
    ax.plot(history.history['val_' + type], "r", linewidth=2)
    ax.set_title('Model ' + type, fontsize=TITLE_SIZE)
    ax.set_ylabel(type, loc="top", fontsize=FONT_SIZE)
    ax.set_xlabel('Epoch', loc="right", fontsize=FONT_SIZE)
    ax.legend(['Training', 'Validation'], loc='upper left', fontsize=LEGEND_SIZE)
    ax.grid(linewidth=0.8)
    if bSave:
        fig.savefig(fSavePath + type + "TrainingHistory.png")
    if bShow:
        plt.show()


# ------------------------------------------------------------------------------------------------------------- #
def plot_4_recon_curves_paper(curves, curveoris, title, paratype, segment=[np.array([None])], recon_given=True,
                        ind=[5, 25, 40, 50], bShow=True, bSave=False, fSavePath="C:/"):
    """
    Plots 4 reconstructed aorta curves into on plot
    :param curves: Estimated aortic pressure curves as list of segments [Num segments L x individual segment len] or only paras
    :param curveoris: Original parametric aortic pressure curves [Num segments L x individual segment len] or only paras
    :param title: Plot title
    :param paratype: Type of paras for reconstruction
    :param segment: Original parametric aortic pressure curves [Num segments L x individual segment len]
    :param recon_given: If the reconstructed curves are already given /or if reconstruction should be performed
    :param ind: Indices from curves and curvesoris for which segments to use
    :param bShow: if to show the plot
    :param bSave: if to save the plot
    :param fSavePath: Path to save the plot
    """
    curves_used = []
    curves_used_ori = []
    for i in ind:
        curves_used.append(curves[i])
     #   curves_used_ori.append(curveoris[i])
 #   FONT_SIZE = 18
    # Für Graphik die beide columns überspannt 18, sonst 36
    FONT_SIZE = 18

    curve = curves_used
  #  curveori = curves_used_ori
    curveori = curveoris
    fontSize = 22
    title_Size = 24
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rc('xtick', labelsize=fontSize)
    plt.rc('ytick', labelsize=fontSize)
    # Figure
    fig, ax = plt.subplots(2, 2, figsize=(17, 9))

    plt.subplots_adjust(left=0.062, top=0.986, right=0.99, bottom=0.08, wspace=0.05, hspace=0.07)
    r = 0
    c = 0
    for axi in ax.flat:
        for axis in ["top", "bottom", "left", "right"]:
            axi.spines[axis].set_linewidth(2)
        axi.yaxis.set_tick_params(labelsize=FONT_SIZE)
        axi.xaxis.set_tick_params(labelsize=FONT_SIZE)
    for q in range(len(ind)):
        ax[r, c].plot(curve[q], color="maroon", linewidth=2, label="Estimated Curve")
        ax[r, c].plot(curveori[q], color="steelblue", linewidth=2, label="Parametric Curve")
        if segment[0].any() != None:
            ax[r, c].plot(segment[q], color="goldenrod", linewidth=2, label="Original Curve")
        ax[r, c].grid(linewidth=0.8)
        ax[r, c].legend(fontsize=FONT_SIZE)
    #    ax[r, c].set_title(title, fontsize=FONT_SIZE)
        c += 1
        if c == 2:
            c = 0
            r += 1
    ax[0,0].set_xticklabels([])
    ax[0,1].set_xticklabels([])
    ax[1,1].set_yticklabels([])
    ax[0,1].set_yticklabels([])

    ax[1, 1].set_xlabel("Samples", fontsize=FONT_SIZE, loc="right")
    ax[0, 0].set_ylabel("Pressure [mmHg]", fontsize=FONT_SIZE, loc="top")
    if bSave:
        fig.savefig(fSavePath + title + "4Plot.png")
        fig.savefig(fSavePath + title + "4Plot.svg")
      #  fig.savefig(fSavePath + title + "4Plot.pgf")
    #if bShow:
    plt.show()





# ------------------------------------------------------------------------------------------------------------- #
def plot_onepig_pressure(ppfTrue, ppfPredicted,  ppsInfos, title, name="mean", bShow=True, bSave=False, sSavePath="C:/"):
    if name == "mean":
        fct = np.mean
    elif name == "min":
        fct = np.min
    else:
        fct = np.max

    def calc_measure_from_segments(ppfData, func):
        if type(ppfData) is list:
            m = []
            for i in ppfData:
                m.append(func(i))
            m = np.array(m)
        else:
            m = func(ppfData, axis=1)
        return m

    m_true = calc_measure_from_segments(ppfTrue, fct)
    if len(ppfPredicted) > 0:
        m_pred = calc_measure_from_segments(ppfPredicted, fct)
    #if len(ppfEIT) >0:
    #    m_eit = calc_measure_from_segments(ppfEIT, fct)

    # Determine Blockmarks
    piBlockmarks = []
    currentblock = ppsInfos[0][1]
    for j in range(len(ppsInfos)):
        if currentblock != ppsInfos[j][1]:
            piBlockmarks.append(j - 1)
            currentblock = ppsInfos[j][1]

    fig, ax = plt.subplots(figsize=(18, 12))
    plt.subplots_adjust(left=0.064, top=0.936, right=0.94, bottom=0.06)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2)
    ax.grid(linewidth=1)
    #  ax.set_facecolor('whitesmoke')
    ax.plot(m_true, linewidth=2, color="steelblue", label="AP " + name)
    if len(ppfPredicted) > 0:
        ax.plot(m_pred, linewidth=2, color="maroon", label="Predicted AP " + name)
   # if len(ppfEIT) > 0:
   #     ax.plot(m_eit, linewidth=2, color="gold", label="EIT " + name)

    for i in piBlockmarks:
        ax.axvline(i, color="black", linewidth=1.5, linestyle="-.")
    ax.set_ylabel("M" +name[1:] + " [mm Hg]", fontsize=FONT_SIZE, loc="top")
    ax.set_xlabel("Segment Index", fontsize=FONT_SIZE, loc="right")
    ax.legend(fontsize=FONT_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)

    if bSave:
        fig.savefig(sSavePath + title + name + "Pig.png")
    if bShow:
        plt.show()


