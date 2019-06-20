import numpy as np
import gbl
from scorers_ import ClfScorer, RegScorer


class FeatSetResults():
    def __init__(self, fset_list):
        self.data = {
                    'fset_list': fset_list,
                    'pred_frames': [],
                    'pred_scores': [],
                    'scoring': '',
                    'conf_interval': [],
                    'feat_imp_frames': [],
                    'ests': [],
                    'train_scores': []
                    }

    def sort_prune_pred(self, pred_score_thresh=0.5):

        #if self.data['scoring'] in list(ClfScorer().__dict__.keys()):

        enum = [[i, v] for i, v in enumerate(self.data['pred_scores'])]
        idx = [i[0] for i in sorted(enum, key=lambda k:k[1], reverse=True) if i[1] > pred_score_thresh]

        self.data['pred_frames'] = [self.data['pred_frames'][i] for i in idx]
        self.data['pred_scores'] = [self.data['pred_scores'][i] for i in idx]
        self.data['feat_imp_frames'] = [self.data['feat_imp_frames'][i] for i in idx]
        self.data['ests'] = [self.data['ests'][i] for i in idx]
        self.data['train_scores'] = [self.data['train_scores'][i] for i in idx]


def update_results(tgt_name, est_type, fsets_count, fset, fset_results, all_fsets_results_frame, all_fsets_results_dict):
    if fset_results.data['pred_scores']:
        print('%s/%s/%s/%d: updating results:' % (tgt_name, est_type, fset, fsets_count))
        exists = False
        for k, v in all_fsets_results_dict.items():
            print('%s/%s/%s/%d: comparing to:' % (tgt_name, est_type, fset, fsets_count), k)
            # print('%s : ' % (k), set(v['fset_list']))
            # print('%s : ' % (fset), set(fset_results.data['fset_list']))
            if set(fset_results.data['fset_list']) == set(v['fset_list']):
                print('%s/%s/%s/%d: update existing:' % (tgt_name, est_type, fset, fsets_count), k)
                all_fsets_results_frame.loc['freq', k] += 1
                all_fsets_results_dict[k]['preds'].append(fset_results.data['pred_scores'][0])
                if fset_results.data['pred_scores'][0] > all_fsets_results_frame.loc['pred_best', k]:
                    all_fsets_results_frame.loc['pred_best', k] = fset_results.data['pred_scores'][0]
                    all_fsets_results_frame.loc['tgt_best', k] = tgt_name
                    all_fsets_results_frame.loc['est_type_best', k] = est_type

                    all_fsets_results_dict[k].update({'est_best': fset_results.data['ests'][0],
                                                      'pred_frame_best': fset_results.data['pred_frames'][0]
                                                      })
                exists = True
                break
        if not exists:  # add fset into results
            print('%s/%s/%s/%d: create new fset entry:' % (tgt_name, est_type, fset, fsets_count), fset)
            all_fsets_results_frame[fset] = [1, fset_results.data['pred_scores'][0], tgt_name, est_type,
                                             -1, -1]
            all_fsets_results_dict[fset] = {'fset_list': fset_results.data['fset_list'],
                                            'est_best': fset_results.data['ests'][0],
                                            'pred_frame_best': fset_results.data['pred_frames'][0],
                                            'preds': [fset_results.data['pred_scores'][0]]}
                
    return all_fsets_results_frame, all_fsets_results_dict

