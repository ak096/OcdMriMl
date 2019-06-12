import numpy as np


class Results2:
    def __init__(self):
        self.est5 = {
                    'feat_sel': [],
                    'pred_frames': [],
                    'pred_scores': [],
                    'conf_interval': [],
                    'feat_imp_frames': [],
                    'est5': [],
                    'train_scores': []
                    }

    def sort_prune_pred(self):
        enum = [[i,v] for i,v in enumerate(self.est5['pred_scores'])]
        idx = [i[0] for i in sorted(enum, key=lambda k:k[1], reverse=True) if i[1] > 0.5]

        self.est5['pred_frames'] = [self.est5['pred_frames'][i] for i in idx]
        self.est5['pred_scores'] = [self.est5['pred_scores'][i] for i in idx]
        self.est5['feat_imp_frames'] = [self.est5['feat_imp_frames'][i] for i in idx]
        self.est5['est5'] = [self.est5['est5'][i] for i in idx]
        self.est5['train_scores'] = [self.est5['train_scores'][i] for i in idx]


class Results1:
    def __init__(self):
        self.est_type = {
                         'lsvm': Results2(),
                         'xgb': Results2()
                        }


class TargetResults:
    def __init__(self):
        self.targets = {}

    def add_target(self, tgt_name):
        self.targets[tgt_name] = {
                                 't_frame': None,
                                 'f_frame': None,
                                 'mi_frame': None,
                                 'feat_pool_counter': None,
                                 'learned': Results1(),
                                 'hoexter': Results1(),
                                 'boedhoe': Results1()
                                }


class LearnedFeatureSets:
    def __init__(self):
        self.linear = []
        self.non_linear = []
        self.all = []

    def combine(self):
        self.all = self.linear + self.non_linear