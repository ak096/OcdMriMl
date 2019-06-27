import gbl
import pandas as pd
from train_predict import conf_interval
# class FeatSetResults():
#     def __init__(self, fset_list):
#         self.data = {
#                     'fset_list': fset_list,
#                     #'pred_frames': [],
#                     'pred_scores': [],
#                     'scoring': '',
#                     'conf_interval': [],
#                     'feat_imp_frames': [],
#                     'ests': [],
#                     'train_scores': []
#                     }
#
#     def sort_prune_pred(self, pred_score_thresh=0.5):
#
#         #if self.data['scoring'] in list(ClfScorer().__dict__.keys()):
#
#         enum = [[i, v] for i, v in enumerate(self.data['pred_scores'])]
#         idx = [i[0] for i in sorted(enum, key=lambda k:k[1], reverse=True) if i[1] > pred_score_thresh]
#
#         #self.data['pred_frames'] = [self.data['pred_frames'][i] for i in idx]
#         self.data['pred_scores'] = [self.data['pred_scores'][i] for i in idx]
#         self.data['feat_imp_frames'] = [self.data['feat_imp_frames'][i] for i in idx]
#         self.data['ests'] = [self.data['ests'][i] for i in idx]
#         self.data['train_scores'] = [self.data['train_scores'][i] for i in idx]


def update_results(tgt_name, est_type, fsets_count, curr_fset, curr_fset_results, fsets_results_frame,
                   fsets_results_dict, fsets_names_frame):

    print('%s/%s/%s/%d: above thresh, adding to results:' % (tgt_name, est_type, curr_fset, fsets_count))
    exists = False
    for k, v in fsets_results_dict.items():
        #print('%s/%s/%s/%d: comparing to:' % (tgt_name, est_type, curr_fset, fsets_count), k)
        if set(v['fset_list']) == set(curr_fset_results['fset_list']) :
            exists = True
            fsets_results_frame.loc['freq', k] += 1

            print('%s/%s/%s/%d: update existing:%s freq:%d' % (tgt_name, est_type, curr_fset, fsets_count, k,
                                                                fsets_results_frame.loc['freq', k]))
            fsets_results_dict[k]['preds'].append(curr_fset_results['pred_score_best'])
            if fsets_results_frame.loc['pred_best', k] < curr_fset_results['pred_score_best']:
                fsets_results_frame.loc['pred_best', k] = curr_fset_results['pred_score_best']
                fsets_results_frame.loc['tgt_best', k] = tgt_name
                fsets_results_frame.loc['est_type_best', k] = est_type

                fsets_results_dict[k].update({'est_best': curr_fset_results['est_best'],
                                              'pred_frame_best': curr_fset_results['pred_frame_best']
                                             })
            break
    if not exists:  # add fset into results
        print('%s/%s/%s/%d: create new fset entry:' % (tgt_name, est_type, curr_fset, fsets_count), curr_fset)
        fsets_results_frame[curr_fset] = [1, curr_fset_results['pred_score_best'], tgt_name, est_type,
                                          -1, -1]
        fsets_results_dict[curr_fset] = {'fset_list': curr_fset_results['fset_list'],
                                         'fset_names_list': [gbl.all_feat_names[i] for i in
                                                             curr_fset_results['fset_list']],
                                         'est_best': curr_fset_results['est_best'],
                                         'pred_frame_best': curr_fset_results['pred_frame_best'],
                                         'preds': [curr_fset_results['pred_score_best']]
                                         }
        fsets_names_frame = pd.concat([fsets_names_frame,
                                       pd.DataFrame({curr_fset: sorted(fsets_results_dict[curr_fset]['fset_names_list'])})],
                                      axis=1)
    return fsets_results_frame, fsets_results_dict, fsets_names_frame


def compute_results(fsets_results_frame, fsets_results_dict):
    for column in fsets_results_frame:
        fsets_results_frame.loc['pred_avg', column] = \
            sum(fsets_results_dict[column]['preds']) / len(fsets_results_dict[column]['preds'])

        fsets_results_frame.loc['pred_ci', column] = conf_interval(fsets_results_dict[column]['preds'])[1]
    return fsets_results_frame
