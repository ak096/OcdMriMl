import numpy as np
import gbl
from scorers_ import ClfScorer, RegScorer


class FeatSetResults:
    def __init__(self, feat_set_list):
        self.data = {
                    'feat_set_list': feat_set_list,
                    'pred_frames': [],
                    'pred_scores': [],
                    'scoring': '',
                    'conf_interval': [],
                    'feat_imp_frames': [],
                    'est5': [],
                    'train_scores': []
                    }

    def sort_prune_pred(self, pred_score_thresh=0.5):

        #if self.data['scoring'] in list(ClfScorer().__dict__.keys()):

        enum = [[i, v] for i, v in enumerate(self.data['pred_scores'])]
        idx = [i[0] for i in sorted(enum, key=lambda k:k[1], reverse=True) if i[1] > pred_score_thresh]
        print('sorting pruning pred results, only left')
        print(idx)
        self.data['pred_frames'] = [self.data['pred_frames'][i] for i in idx]
        self.data['pred_scores'] = [self.data['pred_scores'][i] for i in idx]
        self.data['feat_imp_frames'] = [self.data['feat_imp_frames'][i] for i in idx]
        self.data['est5'] = [self.data['est5'][i] for i in idx]
        self.data['train_scores'] = [self.data['train_scores'][i] for i in idx]

