import pandas as pd
import numpy as np
from random import uniform, randint
from sklearn.model_selection import ParameterGrid, ParameterSampler


def svm_hyper_param_space(est_class):
    svm_hps = {
               'C': np.arange(1, 1001, 99),
               }
    # if est_class == 'reg':
    #     svm_hps['epsilon'] = [0.3, 0.5, 0.7, 0.9]
    if est_class == 'clf':
        svm_hps['class_weight'] = ['balanced']
    return svm_hps


def xgb_hyper_param_space():
    return {  #learning task parameters : objective, eval, ...
              #'objective': ,
              #'eval_meas': ,
              #booster tree parameters
              #1.set init values, comp less expensive for initial look
              #for highly imbalanced classes
              'scale_pos_weight': [1],
              #2.most impact these two
              'min_child_weight': [1, 2, 3, 4, 5, 6],  #default:1
              "max_depth": [3, 4, 5, 6, 7, 8], # default:6
              #3.carry on with gamma
              "min_split_loss": [0, 0.1, 0.2, 0.3],  # alias:gamma, default:0 should be tuned according to loss function
              #4.these two around 0.8
              "subsample": [0.5, 0.6, 0.7, 0.8, 0.9],  # default
              'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
              #5.regularization parameters : model complexity and performance and under/over fitting?
              'reg_lambda': [1], #alias:lambda (L2), default:1
              'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05], #alias:alpha (L1), default:0
              #6.decrease learning rate and increase number of trees
              "learning_rate": [0.1, 0.2],  # alias:eta, default:0.3
              'n_estimators': np.arange(100, 701, 100),  # default:100
            }


def init_globals():

    global normType_list
    normType_list = ['std', 'minMax', 'robust']
    global FS_feats
    FS_feats = []
    global clin_demog_feats
    clin_demog_feats = ['gender_num', 'age', 'duration', 'med']

    global param_grid_lsvc
    param_grid_lsvc = list(ParameterGrid(svm_hyper_param_space('clf')))
    global param_grid_lsvr
    param_grid_lsvr = list(ParameterGrid(svm_hyper_param_space('reg')))
    global param_grid_xgb
    param_grid_xgb = list(ParameterSampler(xgb_hyper_param_space(), n_iter=20))
    # param_grid_xgbr = list(ParameterSampler(xgb_hyper_param_space(), n_iter=20))

    # Hoexter et al 2013 (CSTC)
        # volumetric data:
        # right rostral anteriorcingulate
        # left rostral anteriorcingulate
        # right thalamus
        # left thalamus
        # right medial orbitofrontal
        # right lateral orbitofrontal
        # left medial orbitofrontal
        # left lateral orbitofrontal
        # right accumbens area (?)
        # right pallidum
        # right putamen
        # right caudate
        # left accumbens area (?)
        # left pallidum
        # left putamen
        # left caudate
    global hoexter_feats_FS
    hoexter_feats_FS = [
        'lh_rostralanteriorcingulate_volume**aparc',
        'rh_rostralanteriorcingulate_volume**aparc',

        'Right-Thalamus-Proper**volume',
        'Left-Thalamus-Proper**volume',

        'rh_medialorbitofrontal_volume**aparc',
        'lh_medialorbitofrontal_volume**aparc',

        'rh_lateralorbitofrontal_volume**aparc',
        'lh_lateralorbitofrontal_volume**aparc',

        'Left-Accumbens-area**volume',
        'Right-Accumbens-area**volume',

        'Left-Pallidum**volume',
        'Right-Pallidum**volume',

        'Right-Putamen**volume',
        'Left-Putamen**volume',

        'Left-Caudate**volume',
        'Right-Caudate**volume'
    ]

    # Boedhoe et al 2016 (Pallidum, Hippocampus)
    global boedhoe_feats_FS
    boedhoe_feats_FS = [
        'Left-Pallidum**volume',
        'Right-Pallidum**volume',
        'Left-Hippocampus**volume',
        'Right-Hippocampus**volume'
    ]

    return
