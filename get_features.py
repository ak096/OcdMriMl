from hoexter_features import hoexter_FSfeats
from boedhoe_features import boedhoe_FSfeats
import glob


def get_feats(key, bm):
    print('getting features for %s' % key)
    if 'hoexter' in key:
        print('returning hoexter_features')
        feats = hoexter_FSfeats
    elif 'boedhoe' in key:
        print('returning boedhoe_features')
        feats = boedhoe_FSfeats
    elif 't_' in key:
        print('returning t_features')
        feats = glob.t_frame_perNorm_list[bm['normIdx_train']].columns[0:bm['num_feats']-4].tolist()

    return feats
