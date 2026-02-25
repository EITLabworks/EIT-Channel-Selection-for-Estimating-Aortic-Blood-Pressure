from numpy.random import seed
seed(1)
from tensorflow.random import set_seed
set_seed(2)
import json

from src.data.util_paras import load_preprocess_segments, reload_aorta_segs_from_piginfo
from src.nn.evaluation_fcts import *
from src.nn.reconstruction import *
import warnings
import psutil
process=psutil.Process()
warnings.filterwarnings('ignore')

from src.nn.eva_metrics import EvaMetrics
from src.nn.help_function_training import *

import gc
set_global_determinism(26)
bSaveModel = True
bParseArgs = False


# Loading config ------------------------------------------------------------------------------------------ #
config_path= "/results/nn/configs/"
data_path = "C:/Users\pfuchs\Documents/Data/EIT/PulHypStudie/DataOriginal/"
data_prefix = "data/npz/"
savePath= "/results/models/"
modelpath = "/results/models/"
modelname= "20260212-113936/"
# Parsing of arguments ------------------------------------------------------------------------------------------ #
TP = parse_arguments_new(False, 300,16)        # TrainingParameters


config_file = "/config_modelTEST.json"
with open(config_path+config_file, "r") as file:
    config = json.load(file)
print("Starting with config:\n"+str(config))

TP.iEpochs=150
TP.bResampleParas=True
TP.sNormEIT="block"
TP.sNormAorta="fixed"
TP.sLossfct ="mae"
TP.sParaType = "Linear"
TP.iNumParas = 22
#arguments
TP.iOutputDim=TP.iNumParas


test_pig = "PulHyp_09"
print(f"Exclude pig: {test_pig}\n\n")

bPlotGraphics = True
bSaveGraphics = True


# Load model-------------------------------------------------------------------------------------------- #
model = tf.keras.models.load_model(modelpath + modelname + "model.keras")
model.summary()


##############################################
# Load Testing Data
print(f"Test pig: {test_pig}")
X_test, y_test, p_test, AortaNorm = load_preprocess_segments(
    data_prefix,
    [test_pig],
    zero_padding=TP.bZeropadding,
    shuffle=False,
    eit_length=config["eit_length"],
    aorta_max_length=config["aorta_max_length"],
    para_len=TP.iNumParas,
    norm_eit=TP.sNormEIT,
    norm_aorta=TP.sNormAorta,
    resample_paras=TP.bResampleParas,
    sUseIndex=TP.sUseIndex,
    aorta_normalizer=None,
    loading_function="paras"
)

y_test_preds = model(X_test)
del X_test
gc.collect()


# Visualization ------------------------------------------------------------------------------------------------------ #


y_test_recon = recon_paras_block(y_test, TP.sParaType, AN=AortaNorm, Denorm=TP.sNormAorta, bReorderLinParas=TP.bReorderParas)
y_test_preds_recon = recon_paras_block(y_test_preds, TP.sParaType, AN=AortaNorm, Denorm=TP.sNormAorta, bReorderLinParas=TP.bReorderParas)


bRemovedMin=False

ind = [5,25,40,50]
pig_plot = []
for k in ind:
    pig_plot.append(p_test[k])
aorta_seg_test = reload_aorta_segs_from_piginfo(data_path, pig_plot, False, sNormAorta=TP.sNormAorta, bResampled=TP.bResampleParas)


plot_4_recon_curves_paper(y_test_preds_recon, y_test_recon, "Testing Reconstructed Signal", TP.sParaType,
                    segment=aorta_seg_test, recon_given=True,ind=ind,bShow=bPlotGraphics, bSave=bSaveGraphics,fSavePath=savePath)


ind = [100,125,140,150]
pig_plot = []
for k in ind:
    pig_plot.append(p_test[k])
aorta_seg_test = reload_aorta_segs_from_piginfo(data_path, pig_plot, bRemovedMin, bResampled=TP.bResampleParas, iResampleLen=1024, sNormAorta=TP.sNormAorta)
plot_4_recon_curves_paper(y_test_preds_recon, y_test_recon, "Testing Reconstructed Signal Part2", TP.sParaType,
                    segment=aorta_seg_test, recon_given=True,ind=ind,bShow=bPlotGraphics, bSave=bSaveGraphics,fSavePath=savePath)

del aorta_seg_test
gc.collect()


M = EvaMetrics(savePath)
M.bByParatype=False

y_test_recon_a, y_test_preds_recon_a = set_array_len(y_test_recon, y_test_preds_recon, len(y_test_recon[0]))

del y_test_recon, y_test_preds_recon
gc.collect()
y_test_real = np.array(reload_aorta_segs_from_piginfo(data_path, p_test, bRemovedMin,bResampled=True, iResampleLen=1024, sNormAorta=TP.sNormAorta))


M.calc_metrics(y_test_real, y_test_preds_recon_a, TP.sParaType, "TestCurve", pnames=p_test)
M.calc_metrics(y_test, y_test_preds.numpy(), TP.sParaType, "TestParas", bParas=True)
M.save_metrics()


plot_onepig_pressure(y_test_real, y_test_preds_recon_a, p_test, "Testing Minimum Prediction Course", name="min",bShow=bPlotGraphics, bSave=bSaveGraphics,sSavePath=savePath)
plot_onepig_pressure(y_test_real, y_test_preds_recon_a, p_test, "Testing Maximum Prediction Course", name="max",bShow=bPlotGraphics, bSave=bSaveGraphics,sSavePath=savePath)
plot_onepig_pressure(y_test_real, y_test_preds_recon_a, p_test, "Testing Mean Prediction Course", name="mean",bShow=bPlotGraphics, bSave=bSaveGraphics,sSavePath=savePath)

print("Testing finished.")