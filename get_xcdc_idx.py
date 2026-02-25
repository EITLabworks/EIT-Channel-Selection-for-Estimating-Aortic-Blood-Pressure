"""
Project ：parametric_eit_nn
Directory: data
File : get_xcdc_idx.py
Author ：Patricia Fuchs
Date ：12.02.2026 14:58
"""

# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

import os

import warnings
import psutil

process = psutil.Process()
warnings.filterwarnings('ignore')
import gc
from src.cha_selection.xcdc_channel_selection import XCDC
# set_global_determinism(26)
import json
from src.data.util_paras import load_preprocess_segments
from src.nn.help_function_training import parse_arguments_new
import numpy as np
# Loading config ------------------------------------------------------------------------------------------ #
config_path = "/results/nn/configs/"
data_path = "/data/DataPulHypOriginal/"

sResultspath = "/results/nn/"


# Parsing of arguments ------------------------------------------------------------------------------------------ #


os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# Create Model Function


TP = parse_arguments_new(True, 300, 16)  # TrainingParameters

# config_file = "/config_modelTEST.json"
with open(config_path + TP.sConfigFile, "r") as file:
    config = json.load(file)
print("Starting with config:\n" + str(config))
TP.sParaType = config["para_type"]
TP.iNumParas = config["para_size"]
TP.iOutputDim = TP.iNumParas
TP.iOutputDim = 3


# for pop_pig in range(n_pigs):
TP.iNpigs = len(config["training_examples"])
print(f"{TP.iNpigs=}")
train_pigs = list(np.arange(TP.iNpigs))
train_pigs.pop(TP.iTestpig)
train_pigs = [config["training_examples"][sel] for sel in train_pigs]
test_pig = config["training_examples"][TP.iTestpig]
print(f"Exclude pig: {test_pig}\n\n")

print("Started loading data.")
X, y, pig_info, AortaNorm = load_preprocess_segments(
    config["data_prefix"],
    train_pigs,
    para_len=TP.iNumParas,
    zero_padding=False,     # resample eit
    shuffle=True,
    eit_length=config["eit_length"],
    aorta_max_length=config["aorta_max_length"],
    norm_eit="none",
    norm_aorta="none",
    resample_paras=False,
    sUseIndex="none",
    loading_function="minmeanmax"
)
print("Finished loading data.")
print(f"Xshape: {X.shape}")
del pig_info, AortaNorm
gc.collect()

# transform aorta into just the mean
y= y[:, 1]
print(f"y shape is {y.shape}")

X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))


iChannels= 1024
xclass= XCDC(iChannels, sResultspath)
xclass.fLambda= 0.5
xclass.fGamma= 1
#xclass.load_results("C:/Users/pfuchs/Documents/uni-rostock/python_projects/parametric_eit_nn/results/nn/", 50,1)
#xclass.make_idx_range()

xclass.run_xcdc_fast(X, y, iWindow=4092, iFFT= 128)
xclass.save_json_index()
xclass.plot_lambda_range(1,bSave=True, sSavepath=sResultspath )
xclass.plot_gamma_range(0.5, bSave=True, sSavepath=sResultspath)

