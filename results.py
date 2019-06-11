

class Results2():
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

class Results1():
    def __init__(self):
        self.est_type = {
                         'lsvm': Results2(),
                         'xgb': Results2()
                        }


class TargetResults():
    def __init__(self):
        self.targets = {}

    def add_target(self, tgt_name):
        self.target[tgt_name] = {
                                 't_frame': None,
                                 'f_frame': None,
                                 'mi_frame': None,
                                 'feat_pool_counter': None,
                                 'learned': Results1(),
                                 'hoexter': Results1(),
                                 'boedhoe': Results1()
                                }


