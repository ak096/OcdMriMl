import pandas as pd
import numpy as np
from collections import Counter
from copy import deepcopy

import gbl
from train_predict import conf_interval
from feat_selection_ml import compute_fqis_apriori_frame, compute_fqis_pyfpgrowth_dict, compute_fqis_orange_fpgrowth_list, compute_fqis_lcs_dict


def update_fset_results(tgt_name, est_type, fsets_count, curr_fset, curr_fset_results, fsets_results_frame,
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


def compute_fset_results_frame(fsets_results_frame, fsets_results_dict):
    for column in fsets_results_frame:
        fsets_results_frame.loc['pred_avg', column] = round(conf_interval(fsets_results_dict[column]['preds'])[0], 3)
        fsets_results_frame.loc['pred_ci', column] = round(conf_interval(fsets_results_dict[column]['preds'])[1], 3)
    return fsets_results_frame


def compute_fpi_results_dict(fpis):
    fpi_result = {}
    for k, v in fpis.items():
        fpi_result[gbl.all_feat_names[k]] = {'pi_freq': len(fpis[k]),
                                             'pi_avg': round(conf_interval(fpis[k])[0], 3),
                                             'pi_high': np.max(fpis[k]),
                                             'pi_low': np.min(fpis[k]),
                                             'pi_ci': round(conf_interval(fpis[k])[1], 3)
                                             }

    return fpi_result


def cust_func(s):
    t = sum(s.freq)
    if type(s.freq) is not pd.core.series.Series: # is than non-iterable int or float
        s.freq = [s.freq]
    if type(s.perm_imp_avg) is not pd.core.series.Series: # is than non-iterable int or float
        s.perm_imp_avg = [s.perm_imp_avg]
    pi_freq = t
    pi_avg = round(sum([(i[0]/t)*i[1] for i in list(zip(s.freq, s.perm_imp_avg))])/t, 3)
    pi_high = round(np.max(s.perm_imp_high), 3)
    pi_low = round(np.min(s.perm_imp_low), 3)

    return pd.Series({'pi_freq': pi_freq, 'pi_avg': pi_avg, 'pi_high': pi_high, 'pi_low': pi_low})


def combine_fpi_frames(args):
    fpi_all_frame = pd.concat([a for a in args], sort=False)
    fpi_all_frame.apply(cust_func)
    return fpi_all_frame # .sort_values(by='pi_avg', axis=1, ascending=False, inplace=True) # without ci


def combine_dicts(*args):
    super_dict = {}
    for dict in args:
        for k, v in dict.items():
            super_dict.setdefault(k, []).extend(v)
    return super_dict


def compute_fpi_all_results_frame(*args): #super_dict):#
    super_dict = combine_dicts(*args)
    fpi_all_results_dict = compute_fpi_results_dict(super_dict)
    fpi_all_results_frame = pd.DataFrame.from_dict(fpi_all_results_dict)
    return fpi_all_results_frame.sort_values(by='pi_avg', axis=1, ascending=False, inplace=True)


def compute_hb_idx(fsets_results_frame):
    fsrf = fsets_results_frame
    hb_idx = [True if any(f in c for f in ['hoexter', 'boedhoe']) else False for c in fsrf]
    non_hb_all_idx = [not i for i in hb_idx]
    ph = fsrf.loc['pred_best', hb_idx].min()
    non_hb_geq = [c for c in fsrf.loc[:, non_hb_all_idx] if fsrf.loc['pred_best', c] >= ph]
    return non_hb_geq, hb_idx, non_hb_all_idx


def construct_fqis_super_isets(fsets_results_frame, fsets_names_frame): # for fsets >= pred_best of min hoexter/boedhoe
    non_hb, _, _ = compute_hb_idx(fsets_results_frame)
    fsnf = fsets_names_frame
    super_isets = [[f for f in fsnf[c].tolist() if str(f) != 'nan'] for c in non_hb]
    return super_isets


def compute_hb_fcounts_frame(fsets_results_frame, fsets_names_frame):
    non_hb_geq, _, _ = compute_hb_idx(fsets_results_frame)
    fsnf = fsets_names_frame
    non_hb_geq_f_all = []
    for c in fsnf.loc[:, non_hb_geq]:
        non_hb_geq_f_all.extend(list(set(fsnf[c].tolist())))
    non_hb_fcounts_frame = pd.DataFrame(index=['count'], data=dict(Counter(non_hb_geq_f_all)))
    non_hb_fcounts_frame.sort_values(by='count', axis=1, ascending=False, inplace=True)
    nhbfcf = non_hb_fcounts_frame
    non_hb_hb_f = [c for c in nhbfcf if c in set(gbl.h_b_expert_fsets['DesiDest'][0] + gbl.h_b_expert_fsets['DesiDest'][1])]
    non_hb_hb_fcounts_frame = non_hb_fcounts_frame.loc[:, non_hb_hb_f]
    return non_hb_fcounts_frame, non_hb_hb_fcounts_frame


def compute_store_results():
    min_sup = 0.50
    fqis_algo = 'lcs'
    e_writer = pd.ExcelWriter('all_fqis{}{}_fsup.xlsx'.format(min_sup, fqis_algo))

    for task in [gbl.clf, gbl.reg]:
        print(task)
        fpi_results_frames = []
        for atlas in gbl.atlas_dict.keys():
            print(atlas)
            fsets_results_frame = pd.read_excel(io='atlas_{}_maxgridpoints_30_minsupport_0.9.xlsx'.format(atlas),
                                                sheet_name='fsets_results_{}'.format(task), index_col=0)

            fsets_names_frame = pd.read_excel(io='atlas_{}_maxgridpoints_30_minsupport_0.9.xlsx'.format(atlas),
                                              sheet_name='fsets_names_{}'.format(task), index_col=0, dtype=str)

            fpi_results_frames.append(pd.read_excel(io='atlas_{}_maxgridpoints_30_minsupport_0.9.xlsx'.format(atlas),
                                      sheet_name='fimps_{}'.format(task), index_col=0))


            super_isets_names_list = construct_fqis_super_isets(fsets_results_frame, fsets_names_frame)
            super_isets_index_list = [[gbl.all_feat_names.index(f) for f in s] for s in super_isets_names_list]
            drop=False
            if fqis_algo == 'apriori':
                # fqis with apriori
                #fqis_frame = compute_fqis_apriori_frame(super_isets_names_list, min_sup)
                pass
            elif fqis_algo == 'pyfpgrowth':
                ## fqis with fpgrowth
                fqis_dict = compute_fqis_pyfpgrowth_dict(super_isets_index_list, min_sup)
                fqis_frame = pd.DataFrame({'itemsets': [{gbl.all_feat_names[f] for f in s} for s in fqis_dict.keys()],
                                           'support': [round(v/len(super_isets_index_list), 3) for v in fqis_dict.values()]},
                                           index=np.arange(len(fqis_dict.keys())))
                drop=True
            elif fqis_algo == 'orange':
                ## fqis with orange
                print('fqis orange')
                fqis_list = compute_fqis_orange_fpgrowth_list(super_isets_index_list, min_sup)
                fqis_frame = pd.DataFrame({'itemsets': [{gbl.all_feat_names[f] for f in s[0]} for s in fqis_list],
                                           'support': [round(v[1]/len(super_isets_index_list), 3) for v in fqis_list]},
                                           index=np.arange(len(fqis_list)))
                drop=True
            elif fqis_algo == 'lcs':
                ## fqis with largest common subsets
                fqis_dict = compute_fqis_lcs_dict(super_isets_index_list, min_sup)
                fqis_frame = pd.DataFrame({'itemsets': [{gbl.all_feat_names[f] for f in s} for s in fqis_dict.keys()],
                                           'support': fqis_dict.values()},
                                           index=np.arange(len(fqis_dict.keys())))
                drop=False

            fqis_frame.sort_values(by='support', axis=0, ascending=False, inplace=True)
            if drop:
                drop_list = []
                for i, its_i in enumerate(fqis_frame.loc[:, 'itemsets']):
                    print('processing set {} from {}'.format(i + 1, len(fqis_frame.loc[:, 'itemsets'])))
                    if len(its_i) == 1:
                        drop_list.append(i)
                        continue
                    elif i+1 == len(fqis_frame.loc[:, 'itemsets']):
                        continue
                    else:
                        for its_j in fqis_frame.loc[i+1:, 'itemsets']:
                            if set(its_i).issubset(set(its_j)):
                                drop_list.append(i)
                                break

            print('dropping:', drop_list)
            fqis_frame.drop(drop_list, inplace=True)

            fqis_frame.to_excel(e_writer, '{}_{}_fqis'.format(atlas, task))

            # fcounts and fsup
            non_hb_geq, _, _ = compute_hb_idx(fsets_results_frame)
            non_hb_fcounts_frame, hb_fcounts_frame = compute_hb_fcounts_frame(fsets_results_frame, fsets_names_frame)

            non_hb_fcounts_frame = non_hb_fcounts_frame.transpose()
            hb_fcounts_frame = hb_fcounts_frame.transpose()
            non_hb_fcounts_frame['support'] = non_hb_fcounts_frame['count']/len(non_hb_geq)
            hb_fcounts_frame['support'] = hb_fcounts_frame['count']/len(non_hb_geq)

            non_hb_fcounts_frame.to_excel(e_writer, '{}_{}_non_hb_fsup'.format(atlas, task))
            hb_fcounts_frame.to_excel(e_writer, '{}_{}_hb_fsup'.format(atlas, task))

        # fpi_all_frame = combine_fpi_frames(fpi_results_frames)
        # fpi_all_frame.to_excel(e_writer, 'fimps_{}'.format(task))

    e_writer.save()
