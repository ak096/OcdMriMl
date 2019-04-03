from hoexter_features import hoexter_FSfeats
from boedhoe_features import boedhoe_FSfeats
import glob


def get_feats(key, bm):
    print('getting features for %s' % key)
    if 'hoexter' in key:
        print('returning hoexter features')
        feats = hoexter_FSfeats
    elif 'boedhoe' in key:
        print('returning boedhoe features')
        feats = boedhoe_FSfeats
    elif 't_' in key:
        print('returning tfeatures')
        feats = glob.t_frame_perNorm_list[glob.normType_list.index(bm['normType_train'])].columns[0:bm['num_feats']].values

    return feats
