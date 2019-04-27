from hoexter_features import hoexter_FSfeats
from boedhoe_features import boedhoe_FSfeats
import glob


def get_feats(key, bm):

    if 'hoexter' in key:
        feats = hoexter_FSfeats
    elif 'boedhoe' in key:
        feats = boedhoe_FSfeats
    elif 't_' in key:
        feats = glob.t_frame_perNorm_list[bm['normIdx_train']].columns[0:bm['num_feats']-4].tolist()

    return feats
