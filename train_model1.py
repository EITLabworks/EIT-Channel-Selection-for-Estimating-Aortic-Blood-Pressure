from numpy.random import seed

seed(1)
from tensorflow.random import set_seed

set_seed(2)
from tensorflow.keras.layers import Input, Activation, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
import json
from sklearn.model_selection import train_test_split

from src.data.util_paras import load_preprocess_segments, reload_aorta_segs_from_piginfo
from src.nn.evaluation_fcts import *
from tensorflow.keras.optimizers import Adam
from src.nn.reconstruction import *
import warnings

warnings.filterwarnings('ignore')

from src.nn.eva_metrics import EvaMetrics
from src.nn.help_function_training import *

import gc

set_global_determinism(26)
bSaveModel = True
bParseArgs = False

# Loading config ----------------------------------------------------------------------------------------------------- #
config_path = "/results/nn/configs/"
data_path = "/PulHypStudie/DataOriginal/"
mprefix = '/results/nn/models/'

# Parsing of arguments ----------------------------------------------------------------------------------------------- #
TP = parse_arguments_new(bParseArgs, 300, 16)  # TrainingParameters

with open(config_path + TP.sConfigFile, "r") as file:
    config = json.load(file)
print("Starting with config:\n" + str(config))

TP.iEpochs = 150
TP.bResampleParas = True
TP.sNormEIT = "block"
TP.sNormAorta = "fixed"
TP.sLossfct = "mae"
TP.sParaType = config["para_type"]
TP.iNumParas = config["para_size"]
#arguments
TP.iOutputDim = TP.iNumParas

TP.iNpigs = len(config["training_examples"])
print(f"{TP.iNpigs=}")
# for pop_pig in range(n_pigs):
train_pigs = list(np.arange(TP.iNpigs))
train_pigs.pop(TP.iTestpig)
train_pigs = [config["training_examples"][sel] for sel in train_pigs]
test_pig = config["training_examples"][TP.iTestpig]
print(f"Exclude pig: {test_pig}\n\n")

# Load of Training Data ---------------------------------------------------------------------------------------------- #
print("Started loading data.")
X, y, clrs_pig, AortaNorm = load_preprocess_segments(
    config["data_prefix"],
    train_pigs,
    para_len=TP.iNumParas,
    zero_padding=TP.bZeropadding,
    shuffle=True,
    eit_length=config["eit_length"],
    aorta_max_length=config["aorta_max_length"],
    norm_eit=TP.sNormEIT,
    norm_aorta=TP.sNormAorta,
    resample_paras=TP.bResampleParas,
    sUseIndex=TP.sUseIndex,
    loading_function="paras"
)
print("Finished loading data.")

X_train, X_valid, y_train, y_valid, clrs_train, clrs_valid = train_test_split(
    X, y, clrs_pig, test_size=0.1, random_state=42, shuffle=False)

del X, y
gc.collect()
print(f"{X_train.shape=},{X_valid.shape=},{y_train.shape=},{y_valid.shape=}")

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
iInputDim = len(X_valid[0][0])


# Model Initialization------------------------------------------------------------------------------------------------ #
def model(input_shape=(64, 1024, 1), latent_dim=42, ConfigParas=None):
    mapper_input = Input(shape=input_shape)
    x = mapper_input

    # convolutional layers
    x = Conv2D(7, kernel_size=[3, 8], strides=(2, 4), padding="same")(x)
    x = Activation("relu")(x)

    x = Conv2D(4, kernel_size=[1, 3], strides=(2, 2), padding="same")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Activation("elu")(x)
    x = Dropout(0.2)(x)

    x = Conv2D(4, kernel_size=[3, 8], strides=(2, 2), padding="same")(x)
    x = Activation("tanh")(x)

    x = Flatten()(x)
    x = Dense(3 * latent_dim, activation="elu")(x)
    x = Dense(2 * latent_dim, activation="elu")(x)  # elu
    mapper_output = Dense(latent_dim, activation="linear")(x)  ##linear
    return Model(mapper_input, mapper_output)


# Model Training ----------------------------------------------------------------------------------------------------- #
opt = Adam(learning_rate=TP.iLr)
eit_sample_len = len(X_valid[0][0])

model = model(input_shape=(config["eit_length"], eit_sample_len, 1), latent_dim=TP.iOutputDim, ConfigParas=TP)
model.compile(optimizer=opt, loss=TP.sLossfct, metrics=["accuracy", "mse"])

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_valid, y_valid),
    epochs=TP.iEpochs,
    batch_size=TP.iBatchsize)
print("Training finished")
del X_train, y_train

# Loading test data -------------------------------------------------------------------------------------------------- #
print(f"Test pig: {test_pig}")
X_test, y_test, p_test, AortaNorm = load_preprocess_segments(
    config["data_prefix"],
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
    aorta_normalizer=AortaNorm,
    loading_function="paras"
)

y_test_preds = model(X_test)
del X_test
gc.collect()
y_valid_preds = model(X_valid)
del X_valid
gc.collect()

# Save model --------------------------------------------------------------------------------------------------------- #
path = mprefix
if bSaveModel:
    path = save_model_info(mprefix, config_path, AortaNorm, model, history, TP)

# Visualization ------------------------------------------------------------------------------------------------------ #
bPlotGraphics = True
plot_history(history, "loss", bSave=bPlotGraphics, fSavePath=path)

y_test_recon = recon_paras_block(y_test, TP.sParaType, AN=AortaNorm, Denorm=TP.sNormAorta,
                                 bReorderLinParas=TP.bReorderParas)
y_test_preds_recon = recon_paras_block(y_test_preds, TP.sParaType, AN=AortaNorm, Denorm=TP.sNormAorta,
                                       bReorderLinParas=TP.bReorderParas)

y_valid_recon = recon_paras_block(y_valid, TP.sParaType, AN=AortaNorm, Denorm=TP.sNormAorta,
                                  bReorderLinParas=TP.bReorderParas)
y_valid_preds_recon = recon_paras_block(y_valid_preds, TP.sParaType, AN=AortaNorm, Denorm=TP.sNormAorta,
                                        bReorderLinParas=TP.bReorderParas)

bRemovedMin = False

ind = [5, 25, 40, 50]
pig_plot = []
for k in ind:
    pig_plot.append(clrs_valid[k])
aorta_seg_valid = reload_aorta_segs_from_piginfo(data_path, pig_plot, False, sNormAorta=TP.sNormAorta,
                                                 bResampled=TP.bResampleParas)
pig_plot = []
for k in ind:
    pig_plot.append(p_test[k])
aorta_seg_test = reload_aorta_segs_from_piginfo(data_path, pig_plot, False, sNormAorta=TP.sNormAorta,
                                                bResampled=TP.bResampleParas)

plot_4_recon_curves_paper(y_test_preds_recon, y_test_recon, "Testing Reconstructed Signal", TP.sParaType,
                          segment=aorta_seg_test, recon_given=True, ind=ind, bSave=bPlotGraphics, fSavePath=path)
plot_4_recon_curves_paper(y_test_preds_recon, y_test_recon, "Testing Reconstructed Signal", TP.sParaType,
                          segment=aorta_seg_test, recon_given=True, ind=ind, bSave=bPlotGraphics, fSavePath=path)
plot_4_recon_curves_paper(y_valid_preds_recon, y_valid_recon, "Validation Reconstructed Signal", TP.sParaType,
                          segment=aorta_seg_valid, recon_given=True, ind=ind, bSave=bPlotGraphics, fSavePath=path)

ind = [100, 125, 140, 150]
pig_plot = []
for k in ind:
    pig_plot.append(clrs_valid[k])
aorta_seg_valid = reload_aorta_segs_from_piginfo(data_path, pig_plot, bRemovedMin, bResampled=TP.bResampleParas,
                                                 iResampleLen=1024, sNormAorta=TP.sNormAorta)
pig_plot = []
for k in ind:
    pig_plot.append(p_test[k])
aorta_seg_test = reload_aorta_segs_from_piginfo(data_path, pig_plot, bRemovedMin, bResampled=TP.bResampleParas,
                                                iResampleLen=1024, sNormAorta=TP.sNormAorta)
plot_4_recon_curves_paper(y_test_preds_recon, y_test_recon, "Testing Reconstructed Signal Part2", TP.sParaType,
                          segment=aorta_seg_test, recon_given=True, ind=ind, bSave=bPlotGraphics, fSavePath=path)
plot_4_recon_curves_paper(y_valid_preds_recon, y_valid_recon, "Validation Reconstructed Signal Part2", TP.sParaType,
                          segment=aorta_seg_valid, recon_given=True, ind=ind, bSave=bPlotGraphics, fSavePath=path)

del aorta_seg_test, aorta_seg_valid
gc.collect()

M = EvaMetrics(path)
M.bByParatype = False
if TP.bResampleParas:
    y_test_recon_a, y_test_preds_recon_a = set_array_len(y_test_recon, y_test_preds_recon, len(y_test_recon[0]))
    y_valid_recon_a, y_valid_preds_recon_a = set_array_len(y_valid_recon, y_valid_preds_recon, len(y_valid_recon[0]))

else:
    y_test_recon_a, y_test_preds_recon_a = set_array_len_uneven(y_test_recon, y_test_preds_recon, 1200)
    y_valid_recon_a, y_valid_preds_recon_a = set_array_len_uneven(y_valid_recon, y_valid_preds_recon, 1200)

del y_valid_preds_recon, y_valid_recon, y_test_recon, y_test_preds_recon
gc.collect()
y_test_real = np.array(
    reload_aorta_segs_from_piginfo(data_path, p_test, bRemovedMin, bResampled=True, iResampleLen=1024,
                                   sNormAorta=TP.sNormAorta))
y_valid_real = np.array(
    reload_aorta_segs_from_piginfo(data_path, clrs_valid, bRemovedMin, bResampled=True, iResampleLen=1024,
                                   sNormAorta=TP.sNormAorta))

M.calc_metrics(y_test_real, y_test_preds_recon_a, TP.sParaType, "TestCurve", pnames=p_test)
M.calc_metrics(y_valid_real, y_valid_preds_recon_a, TP.sParaType, "ValiCurve")

M.calc_metrics(y_test, y_test_preds.numpy(), TP.sParaType, "TestParas", bParas=True)
M.calc_metrics(y_valid, y_valid_preds.numpy(), TP.sParaType, "ValiParas", bParas=True)

M.save_metrics()
plot_onepig_pressure(y_test_real, y_test_preds_recon_a, p_test, "Testing Minimum Prediction Course", name="min",
                     bSave=bPlotGraphics, sSavePath=path)
plot_onepig_pressure(y_test_real, y_test_preds_recon_a, p_test, "Testing Maximum Prediction Course", name="max",
                     bSave=bPlotGraphics, sSavePath=path)
plot_onepig_pressure(y_test_real, y_test_preds_recon_a, p_test, "Testing Mean Prediction Course", name="mean",
                     bSave=bPlotGraphics, sSavePath=path)
plot_onepig_pressure(y_valid_real, y_valid_preds_recon_a, clrs_valid, "Validation Mean Prediction Course", name="mean",
                     bSave=bPlotGraphics, sSavePath=path)
print("Testing finished.")
