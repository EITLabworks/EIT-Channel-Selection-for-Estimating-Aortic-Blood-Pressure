from tensorflow.keras.layers import Input, Dense, Dropout, Normalization, Activation, Conv2D, MaxPooling2D, Flatten
import argparse

import tensorflow as tf

import numpy as np
import warnings
import os
import time
from contextlib import redirect_stdout
import pickle
import shutil

warnings.filterwarnings('ignore')


# ------------------------------------------------------------------------------------------------------------------- #
def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


def set_global_determinism(seed):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tf.config.experimental.enable_op_determinism()
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    # see also:
    # https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism



class TrainingParas:
    def __init__(self, configepochs=150, configbatchsize=16):
        self.iTestpig = 2
        self.iLr = 1e-3
        self.sUseIndex = "none"

        self.bZeropadding = False
        self.sNormEIT = "block"
        self.bResampleParas = False
        self.sLossfct = "mse"
        self.sConfigFile = "/config_model1.json"
        self.sNormAorta= "none"
        self.iBatchsize = configbatchsize
        self.iEpochs = configepochs
        self.iOutputDim = 0
        self.sParaType = "none"
        self.iNpigs= 0
        self.iNumParas=0


# ------------------------------------------------------------------------------------------------------------------- #
    def write_configs(self, f):
        """
        Writes used parameter configuration to file f
        """
        f.write("Test_pig: " + str(self.iTestpig) + "\n")
        f.write("Epochs: " + str(self.iEpochs) + "\n")
        f.write("Batchsize: " + str(self.iBatchsize) + "\n")
        f.write("Learningrate: " + str(self.iLr) + "\n")
        f.write("Latent_dim = Num Paras: " + str(self.iOutputDim) + "\n")
        f.write("Activation function output layer: " + str(self.actiOutput) + "\n")
        f.write("bZeroPadding EIT signal: " + str(self.bZeropadding) + "\n")
        f.write("Resampling Parameters : " + str(self.bResampleParas) + "\n")
        f.write("Normalization strategy: " + str(self.sNormEIT) + "\n")
        f.write("Lossfunction: " + str(self.sLossfct) + "\n")
        f.write("sUseIndex: " + str(self.sUseIndex) + "\n")
        f.write("normAorta: " + str(self.sNormAorta) + "\n")


    def check_normAorta(self):
        if self.sNormAorta == "standard" or (self.sNormAorta == "bipolar" or self.sNormAorta == "bipolar1200"):
            self.actiOutput = "linear"

def parse_arguments_new(bParseArgs: bool, configepochs, configbatchsize):
    if not bParseArgs:
        return TrainingParas(configepochs, configbatchsize)
    else:
        parser = argparse.ArgumentParser(description="Crossvalidation")
        parser.add_argument("-n", "--number", type=int, help="number between 0 and 20 for excluding a single pig",
                            required=True)
        parser.add_argument("-b", "--batchsize", type=int, help="batchsize", default=16)
        parser.add_argument("-e", "--epochs", type=int, help="epochs", default=300)
        parser.add_argument("-lr", "--learn", type=float, help="learningrate", default=1e-3)
        parser.add_argument("-bz", "--bzero", type=bool, help="bZeropadding", default=False)
        parser.add_argument("-no", "--norm", type=str, help="Normalization type", default="block")
        parser.add_argument("-rp", "--resampleparas", type=bool, help="bResample Parameters (Lin)", default=False)
        parser.add_argument("-lf", "--loss", type=str, help="loss function to use", default="mse")
        parser.add_argument("-rz", "--sUseIndex", type=str, help="Only use certain indices (e.g. nonreziprok",
                            default="none")
        parser.add_argument("-cf", "--config", type=str, help="Config file", default="/config_model1.json")
        parser.add_argument("-na", "--normaorta", type=str, help="normalize Aorta", default="none")



        args = parser.parse_args()
        print(args)
        P  = TrainingParas(configepochs, configbatchsize)
        # pop_pig = [2, 3, 4, 5, 7, 8, 9, 18, 19][args.number]
        P.iTestpig = args.number
        P.iBatchsize = args.batchsize
        P.iLr = args.learn
        P.bResampleParas = args.resampleparas
        P.sLossfct = args.loss
        P.sUseIndex = args.sUseIndex
        P.iEpochs = args.epochs
        P.bZeropadding = args.bzero
        P.sNormEIT = args.norm
        P.sConfigFile = args.config
        P.sNormAorta= args.normaorta

        loss_fcts = {"mse": 0, "mae": 0, "msle": 0, "huber": 0}

        print("Used Loss function " + str(P.sLossfct))
        P.check_normAorta()

        return P


# ------------------------------------------------------------------------------------------------------------------- #
# Array / List modifications
# ------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
def set_array_len(y_truelen, y_change, l:int):
    """
    Changes the length of each subarray to length l and transform another list into an array
    :param y_truelen: This will be transformed into an array [num segs x l]
    :param y_change: The subarrays will be shortened/padded to length l   [num segs x individual lengths]
    :param l: Desired length
    :return: np.array(y_truelen) [num segs x l], np.array(y_change) [num segs x l]
    """
    for k in range(len(y_change)):
        if len(y_change[k]) < l:
            y_change[k] = pad_array(y_change[k], l)
        if len(y_change[k]) > l:
            y_change[k] = y_change[k][:l]
    return np.array(y_truelen), np.array(y_change)


# ------------------------------------------------------------------------------------------------------------------- #
def set_array_len_uneven(y_truelen, y_change, maxlen:int):
    """
    Pad/truncate the segments within two array to the min(max len of y_true, maxlen)
    :param y_truelen: Array with maximal length with segments to be truncated/padded [num segs x individual lengths]
    :param y_change: Array with segments to be truncated/padded [num segs x individual lengths]
    :param maxlen: Maximal length of each segment
    :return: np.array(y_truelen) [num segs x l], np.array(y_change) [num segs x l] with l=min(max_len(y_true_segs), maxlen)
    """
    l = len(y_truelen[0])           # Max len of y_true segments
    for j in range(len(y_truelen)):
        if len(y_truelen[j]) > l:
            l = len(y_truelen[j])

    l = min(maxlen, l)
    for k in range(len(y_truelen)):
        if len(y_truelen[k]) < l:
            y_truelen[k] = pad_array(y_truelen[k], l)
        if len(y_truelen[k]) > l:
            y_truelen[k] = y_truelen[k][:l]

    for k in range(len(y_change)):
        if len(y_change[k]) < l:
            y_change[k] = pad_array(y_change[k], l)
        if len(y_change[k]) > l:
            y_change[k] = y_change[k][:l]

    return np.array(y_truelen), np.array(y_change)



# ------------------------------------------------------------------------------------------------------------------- #
def pad_array(some_array, target_len:int):
    """
    Pad an array to target len
    :param some_array: Array to be padded [individual length]
    :param target_len: Target  length
    :return: some_array [target_len]
    """
    return np.append(some_array, np.ones(target_len - len(some_array))*some_array[-1])



# -------------------------------------------------------------------------------------------------------------- #
def save_model_info(mprefix, config_path, Aortanorm, model, history=None, Configs=None, filename=None):

    timestr = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(mprefix, timestr)
    os.mkdir(path)

    # save weights
    #   model.save_weights(f'{path}/weights', overwrite=True)
    Aortanorm.save_paras_to_json(path)

    # save architecture
    with open(f'{path}/model_config.json', "w") as text_file:
        text_file.write(model.to_json())

    with open(f'{path}/model_summary.txt', "w") as text_file:
        # text_file.write("Test_pig: "+str(test_pig)+"\n\n\n")
        if filename!=None:
            text_file.write("Filename: "+ filename + "\n\n")
        Configs.write_configs(text_file)
        text_file.write("\n\n\n")
        with redirect_stdout(text_file):
            model.summary()
    if history != None:
        with open(f'{path}/history.pickle', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
    model.save(f'{path}/model.keras')
    path = path + "/"
    shutil.copyfile(config_path + Configs.sConfigFile, path + Configs.sConfigFile)
    return path


def get_trainable_parameters(model):
    return np.sum([np.prod(v.shape.as_list()) for v in model.trainable_variables])

def get_small_nnw(tuner, maxtrials=5, size=5000):
    best_hps= tuner.get_best_hyperparameters(maxtrials)
    tp= []
    model = tuner.hypermodel.build(best_hps[0])
    train_paras= get_trainable_parameters(model)
    tp.append(train_paras)
    if train_paras > size:
        print(f"Optimal model has {train_paras} trainable parameters. That exceed max. size.")
    else:
        print(f"Optimal model has {train_paras} trainable parameters. ")
        return model, best_hps[0]
    for i in range(1,maxtrials):
        model = tuner.hypermodel.build(best_hps[i])
        train_paras = get_trainable_parameters(model)
        tp.append(train_paras)
        if train_paras < size:
            print(f"Optimal model (iteration {i} has {train_paras} trainable parameters.")
            return model, best_hps[i]
        else:
            continue
    w= np.argmin(tp)
    model = tuner.hypermodel.build(best_hps[w])
    train_paras = get_trainable_parameters(model)
    print(f"No model underneath the minimum parameter size is found. \nModel (iteration {w} has {train_paras} trainable parameters and is returned.")
    return model, best_hps[w]
