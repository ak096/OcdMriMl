import os
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.feature_selection import VarianceThreshold

from FreeSurfer_read import FreeSurfer_data_collect
from gdrive import get_pat_stats

pd.options.mode.use_inf_as_na = True


def lsvm_hypparam_space(est_class):
    svm_hps = {
               'C': [0.001, 0.01, 0.1, 1, 10, 50, 100, 250, 400, 500, 600, 800, 1000, 2000, 5000],
               }
    if est_class == 'reg':
        svm_hps['epsilon'] = [0.3, 0.5, 0.7, 0.9]
    if est_class == 'clf':
        svm_hps['class_weight'] = ['balanced']
    return svm_hps


def xgb_hypparam_space():
    return {  #learning task parameters : objective, eval, ...
              #'objective': ,
              #'eval_meas': ,
              #booster tree parameters
              #1.set init values, comp less expensive for initial look
              #for highly imbalanced classes
              'scale_pos_weight': [1],

              #2.most impact these two
              'min_child_weight': [1, 2, 3, 4, 5, 6],  #default:1
              'max_depth': [3, 4, 5, 6, 7, 8], # default:6
              #3.carry on with gamma
              "min_split_loss": [0, 0.1, 0.2, 0.3, 1, 5, 15, 50],  # alias:gamma, default:0 should be tuned according to loss function
              #4.these two around 0.8
              "subsample": [0.5, 0.6, 0.7, 0.8, 0.9],  # default
              'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
              #5.regularization parameters : model complexity and performance and under/over fitting?
              'reg_lambda': [1], #alias:lambda (L2), default:1
              'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05], #alias:alpha (L1), default:0
              #6.decrease learning rate and increase number of trees
              "learning_rate": [0.1, 0.2, 0.3, 0.4],  # alias:eta, default:0.3
              'n_estimators': np.arange(100, 701, 100),  # default:100
            }


def gbe_hypparam_space():
    return {'n_estimators': np.arange(100, 701, 100),
            'max_depth': [3, 4, 5, 6, 7, 8],
            'learning_rate': np.arange(0.1, 1.1, 0.1),
            'subsample': np.arange(0.5, 1.0, 0.1),  # default
            }


def _add_rand_feats(frame):
    # add some rand feat (randomly permuted features) to original data frame before train/test

    rand_feats = list(np.random.choice(frame.columns.tolist(), n_rand_feat, replace=False))
    for i in np.arange(n_rand_feat):
        rand_feat_name = 'RANDOM_' + str(i) + '_' + rand_feats[i]
        frame[rand_feat_name] = np.random.permutation(frame.loc[:, rand_feats[i]])
    return frame
# def init_globals():


normType_list = ['std', 'minMax', 'robust']

clin_demog_feats_names = ['gender_num', 'age', 'duration', 'med']

linear_ = 'linear'
non_linear_ = 'non_linear'

clf = 'clf'
reg = 'reg'

with open('lsvc_hypparam_grid.pickle', 'rb') as handle:
    lsvc_hypparam_grid = pickle.load(handle)


with open('lsvr_hypparam_grid.pickle', 'rb') as handle:
    lsvr_hypparam_grid = pickle.load(handle)

with open('xgb_hypparam_grid.pickle', 'rb') as handle:
    xgb_hypparam_grid = pickle.load(handle)


with open('gbe_hypparam_grid.pickle', 'rb') as handle:
    gbe_hypparam_grid = pickle.load(handle)

# param_grid_xgbr = list(ParameterSampler(xgb_hyper_param_space(), n_iter=20))

grid_space_size = len(gbe_hypparam_grid)

## Hoexter et al 2013 (CSTC)
## volumetric data x lh/rh:
## cortical
# rostral anteriorcingulate
# medial orbitofrontal
# lateral orbitofrontal
## subcortical
# thalamus
# accumbens area (no vol. available)
# pallidum
# putamen
# caudate

_hoexter_subcortical_feats_names = [
    # LEFT
    'Left-Thalamus-Proper_volume',
    'Left-Accumbens-area',
    'Left-Pallidum_volume',
    'Left-Putamen_volume',
    'Left-Caudate_volume',
    # RIGHT
    'Right-Thalamus-Proper_volume',
    'Right-Accumbens-area',
    'Right-Pallidum_volume',
    'Right-Putamen_volume',
    'Right-Caudate_volume',
]

hoexter_Desikan_feats_names = [
    # from Desikan cortical parcellation atlas (gyri include sulci border limits)
    # LEFT
    'lh_rostralanteriorcingulate_volume**Desi.',
    'lh_medialorbitofrontal_volume**Desi.',
    'lh_lateralorbitofrontal_volume**Desi.',
    # RIGHT
    'rh_rostralanteriorcingulate_volume**Desi.',
    'rh_medialorbitofrontal_volume**Desi.',
    'rh_lateralorbitofrontal_volume**Desi.',

]
#_hoexter_Dest09s_parts = ['cingulate', 'cingul', 'orbital', 'orbit']
hoexter_Destrieux_feats_names = [
    # from Destrieux cortical parcellation atlas (sulci separate features from gyri)
    # LEFT
    'lh_G&S_cingul-Ant_volume**Dest.09s',
    'lh_G_front_inf-Orbital_volume**Dest.09s',
    'lh_S_orbital_lateral_volume**Dest.09s',
    #'lh_G_orbital_volume**Dest.09s',
    # RIGHT
    'rh_G&S_cingul-Ant_volume**Dest.09s',
    'rh_G_front_inf-Orbital_volume**Dest.09s',
    'rh_S_orbital_lateral_volume**Dest.09s',
    #'rh_G_orbital_volume**Dest.09s'
]

## Boedhoe et al 2016, 2018
## x lh/rh
## cortical
# transverse temporal area
# inferior parietal thickness
## subcortical
# pallidum volume
# hippocampus volume

_boedhoe_subcortical_feats_names = [
    # LEFT
    'Left-Pallidum_volume',
    'Left-Hippocampus_volume',
    # RIGHT
    'Right-Pallidum_volume',
    'Right-Hippocampus_volume',
]

boedhoe_Desikan_feats_names = [
    # Desikan cortical parcellation atlas (gyri include sulci border limits)
    # LEFT
    'lh_transversetemporal_area**Desi.',
    'lh_inferiorparietal_thickness**Desi.',
    # RIGHT
    'rh_transversetemporal_area**Desi.',
    'rh_inferiorparietal_thickness**Desi.',
]

#_boedhoe_Dest09s_parts = ['transverse', 'transv', 'temporal', 'temp', 'parietal', 'pariet']
boedhoe_Destrieux_feats_names = [
    # from Destrieux cortical parcellation atlas (sulci separate features from gyri)
    # LEFT
    'lh_G_temp_sup-G_T_transv_area**Dest.09s',
    'lh_S_temporal_transverse_area**Dest.09s',
    'lh_S_subparietal_thickness**Dest.09s',
    'lh_G_pariet_inf-Angular_thickness**Dest.09s',
    'lh_G_pariet_inf-Supramar_thickness**Dest.09s',
    # RIGHT
    'rh_G_temp_sup-G_T_transv_area**Dest.09s',
    'rh_S_temporal_transverse_area**Dest.09s',
    'rh_S_subparietal_thickness**Dest.09s',
    'rh_G_pariet_inf-Angular_thickness**Dest.09s',
    'rh_G_pariet_inf-Supramar_thickness**Dest.09s',
]


# get data from FreeSurfer stats
path_base = os.path.abspath('Desktop/FS_SUBJ_ALL').replace('PycharmProjects/OcdMriMl/', '')
print('PreProc: FreeSurfer read: pat and con')
pat_frame = FreeSurfer_data_collect('pat', path_base)
con_frame = FreeSurfer_data_collect('con', path_base)

if not pat_frame.columns.tolist() == con_frame.columns.tolist():
    print('PreProc: FreeSurfer read: pat and con frame not equal!')
    exit()

# remove low variance features
before = pat_frame.shape[1]
threshold = 0.01
sel = VarianceThreshold(threshold=threshold)
sel.fit(pat_frame)
retained_mask = sel.get_support(indices=False)
pat_frame = pat_frame.loc[:, retained_mask]
after = pat_frame.shape[1]
print('PreProc: %d feats. removed: under %.2f var: %d to %d' % (before - after, threshold, before, after))

# remove features less than 90% populated
before = pat_frame.shape[1]
ratio = 1.00
threshold = round(ratio * pat_frame.shape[0])
pat_frame.dropna(axis='columns', thresh=threshold, inplace=True)
after = pat_frame.shape[1]
print('PreProc: %d feats. removed: less than %.2f percent filled: %d to %d' %
      (before - after, (threshold / pat_frame.shape[0]) * 100, before, after))

# add some random features (permuted original features)
before = pat_frame.shape[1]
n_rand_feat = 3
pat_frame = _add_rand_feats(pat_frame)
after = pat_frame.shape[1]
print('PreProc: %d feats. (rand.) added: %d to %d' % (n_rand_feat, before, after))

FreeSurfer_feats_names = pat_frame.columns.tolist()

# add clin demo features
pat_frame_stats = get_pat_stats()
pat_frame.sort_index(inplace=True)
pat_frame_stats.sort_index(inplace=True)
if pat_frame_stats.index.tolist() != pat_frame.index.tolist():
    exit("PreProc: feature and target pats not same!")
pat_frame = pd.concat([pat_frame, pat_frame_stats.loc[:, clin_demog_feats_names]], axis=1, sort=False)

all_feat_names = pat_frame.columns.tolist()

pat_frame.columns = np.arange(len(pat_frame.columns.tolist()))

_hoexter_subcortical_feats = [all_feat_names.index(f) for f in _hoexter_subcortical_feats_names]
hoexter_Desikan_feats = [all_feat_names.index(f) for f in hoexter_Desikan_feats_names]
hoexter_Destrieux_feats = [all_feat_names.index(f) for f in hoexter_Destrieux_feats_names]

_boedhoe_subcortical_feats = [all_feat_names.index(f) for f in _boedhoe_subcortical_feats_names]
boedhoe_Desikan_feats = [all_feat_names.index(f) for f in boedhoe_Desikan_feats_names]
boedhoe_Destrieux_feats = [all_feat_names.index(f) for f in boedhoe_Destrieux_feats_names]

FreeSurfer_feats = [all_feat_names.index(f) for f in FreeSurfer_feats_names]
clin_demog_feats = [all_feat_names.index(f) for f in clin_demog_feats_names]

#get std of YBOCS
YBOCS_reg_std = np.std(pat_frame_stats.loc[:, 'YBOCS_reg'])

fpis_clf = {}
fpis_reg = {}

h_b_expert_fsets = {'Desikan': [hoexter_Desikan_feats + _hoexter_subcortical_feats,
                                boedhoe_Desikan_feats + _boedhoe_subcortical_feats],
                    'Destrieux': [hoexter_Destrieux_feats + _hoexter_subcortical_feats,
                                  boedhoe_Destrieux_feats + _boedhoe_subcortical_feats],
                    'Both': [hoexter_Desikan_feats + hoexter_Destrieux_feats + _hoexter_subcortical_feats,
                             boedhoe_Desikan_feats + boedhoe_Destrieux_feats + _boedhoe_subcortical_feats]
                    }
atlas_dict = {'Desikan': ['**Desi.'], 'Destrieux': ['**Dest.09s'], 'Both': []}

# note: convention for fset naming: ...'_feats_names' for strings else just ...'_feats' for indices