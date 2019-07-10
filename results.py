from collections import Counter
from copy import deepcopy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns ; sns.set()
import pickle

import gbl
from train_predict import conf_interval
from feat_selection_ml import compute_fqis_apriori_frame, compute_fqis_fpgrowth_dict, compute_fqis_fpgrowth_orange3_list, compute_lcs_dict


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
    if type(s.freq) is not pd.core.series.Series: # is than non-iterable int or float
        s.freq = [s.freq]
    if type(s.perm_imp_avg) is not pd.core.series.Series: # is than non-iterable int or float
        s.perm_imp_avg = [s.perm_imp_avg]
    freq = [n for n in s.freq if str(n) != 'nan']
    perm_imp_avg = [n for n in s.perm_imp_avg if str(n) != 'nan']
    perm_imp_high = [n for n in s.perm_imp_high if str(n) != 'nan']
    perm_imp_low = [n for n in s.perm_imp_low if str(n) != 'nan']

    t = sum(freq)
    pi_freq = t
    pi_avg = round(sum([f*perm_imp_avg[i] for i, f in enumerate(freq)])/t, 3)
    pi_high = round(np.max(perm_imp_high), 3)
    pi_low = round(np.min(perm_imp_low), 3)

    return pd.Series({'freq': pi_freq, 'perm_imp_avg': pi_avg, 'perm_imp_high': pi_high, 'perm_imp_low': pi_low})


def combine_fpi_frames(args):
    fpi_task_all_results_frame = pd.concat([a for a in args], sort=False)
    fpi_task_results_frame = fpi_task_all_results_frame.apply(cust_func)
    fpi_task_results_frame.sort_values(by='perm_imp_avg', axis=1, ascending=False, inplace=True)
    return fpi_task_results_frame  # without ci


def combine_dicts(*args):
    super_dict = {}
    for dict in args:
        for k, v in dict.items():
            super_dict.setdefault(k, []).extend(v)
    return super_dict


def compute_fpi_all_results_frame(*args):
    fpi_task_dict = combine_dicts(*args)
    fpi_task_results_dict = compute_fpi_results_dict(fpi_task_dict)
    fpi_task_results_frame = pd.DataFrame.from_dict(fpi_task_results_dict)
    fpi_task_results_frame.sort_values(by='pi_avg', axis=1, ascending=False, inplace=True)
    return fpi_task_results_frame


def compute_hb_idx(fsets_results_frame):
    fsrf = fsets_results_frame
    hb_idx = [True if any(f in c for f in ['hoexter', 'boedhoe']) else False for c in fsrf]
    non_hb_idx = [not i for i in hb_idx]
    ph = fsrf.loc['pred_best', hb_idx].min()
    non_hb_geq = [c for c in fsrf.loc[:, non_hb_idx] if fsrf.loc['pred_best', c] >= ph]
    non_hb_geq_idx = [True if c in non_hb_geq else False for c in fsrf]
    return non_hb_geq, hb_idx, non_hb_idx, non_hb_geq_idx


def construct_fqis_super_isets(fsets_results_frame, fsets_names_frame): # for fsets >= pred_best of min hoexter/boedhoe
    non_hb, _, _, _ = compute_hb_idx(fsets_results_frame)
    fsnf = fsets_names_frame
    super_isets = [[f for f in fsnf[c].tolist() if str(f) != 'nan'] for c in non_hb]
    return super_isets


def compute_hb_fcounts_frame(fsets_results_frame, fsets_names_frame):
    non_hb, _, _, _ = compute_hb_idx(fsets_results_frame)
    fsnf = fsets_names_frame
    non_hb_f_all = []
    for c in fsnf.loc[:, non_hb]:
        non_hb_f_all.extend([f for f in fsnf[c].tolist() if str(f) != 'nan'])
    non_hb_fcounts_frame = pd.DataFrame(index=['count'], data=dict(Counter(non_hb_f_all)))
    non_hb_fcounts_frame.sort_values(by='count', axis=1, ascending=False, inplace=True)
    nhbfcf = non_hb_fcounts_frame
    non_hb_hb_f = [c for c in nhbfcf if c in set(gbl.h_b_expert_fsets['DesiDest'][0] + gbl.h_b_expert_fsets['DesiDest'][1])]
    non_hb_hb_fcounts_frame = non_hb_fcounts_frame.loc[:, non_hb_hb_f]
    return non_hb_fcounts_frame, non_hb_hb_fcounts_frame


def compute_store_results():
    min_sup = 0.50
    fqi_algo = 'lcs'
    xlsx_name = 'all_fs_gp30mins0.9_geqhb_fq{}{}_fc_fp.xlsx'.format(min_sup, fqi_algo)
    e_writer = pd.ExcelWriter(xlsx_name)

    for task in [gbl.clf, gbl.reg]:
        print(task)
        fpi_results_frames = []
        non_hb_geq_f = []
        for atlas in ['Desikan', 'Destrieux', 'DesiDest']:
            print(atlas)
            fsets_results_frame = pd.read_excel(io='atlas_{}_maxgridpoints_30_minsupport_0.9.xlsx'.format(atlas),
                                                sheet_name='fsets_results_{}'.format(task), index_col=0)

            fsets_names_frame = pd.read_excel(io='atlas_{}_maxgridpoints_30_minsupport_0.9.xlsx'.format(atlas),
                                              sheet_name='fsets_names_{}'.format(task), index_col=0, dtype=str)

            fpi_results_frames.append(pd.read_excel(io='atlas_{}_maxgridpoints_30_minsupport_0.9.xlsx'.format(atlas),
                                      sheet_name='fimps_{}'.format(task), index_col=0))

            _, hb_idx, _, non_hb_geq_idx = compute_hb_idx(fsets_results_frame)
            fsrf = fsets_results_frame.loc[:, np.array(hb_idx) | np.array(non_hb_geq_idx)].copy()
            fsnf = fsets_names_frame.loc[:, np.array(hb_idx) | np.array(non_hb_geq_idx)].copy()

            # FQIS
            super_isets_names_list = construct_fqis_super_isets(fsets_results_frame, fsets_names_frame)
            super_isets_index_list = [[gbl.all_feat_names.index(f) for f in s] for s in super_isets_names_list]
            # for s in super_isets_names_list:
            #     print(s)
            #     print(len(s))
            # print(len(super_isets_index_list))

            if fqi_algo == 'apriori':
                # apriori
                fqis_frame = compute_fqis_apriori_frame(super_isets_names_list, min_sup=min_sup)
            elif fqi_algo == 'fpgrowth':
                # fpgrowth
                fqis_dict = compute_fqis_fpgrowth_dict(super_isets_index_list, min_sup=min_sup)
                fqis_frame = pd.DataFrame({'itemset': [{gbl.all_feat_names[f] for f in s} for s in fqis_dict.keys()],
                                           'support': [round(l/len(super_isets_index_list), 3) for l in fqis_dict.values()]})
            elif fqi_algo == 'orange':
                # orange fpgrowth
                fqis_list = compute_fqis_fpgrowth_orange3_list(super_isets_index_list, min_sup=min_sup)
                fqis_frame = pd.DataFrame({'itemset': [{gbl.all_feat_names[f] for f in l[0]} for l in fqis_list],
                                           'support': [round(l[1]/len(super_isets_index_list), 3) for l in fqis_list]})
            elif fqi_algo == 'lcs':
                # largest common subsets with support
                fqis_dict = compute_lcs_dict(super_isets_index_list, min_sup=min_sup)
                fqis_frame = pd.DataFrame({'itemset': [{gbl.all_feat_names[f] for f in s} for s in fqis_dict.keys()],
                                           'support': [round(l / len(super_isets_index_list), 3) for l in
                                                       fqis_dict.values()]})

            fqis_frame.to_excel(e_writer, '{}_{}_fqis'.format(atlas, task))

            # FCOUNTS
            non_hb_geq_fcounts_frame, non_hb_geq_hb_fcounts_frame = compute_hb_fcounts_frame(fsets_results_frame, fsets_names_frame)
            non_hb_geq_f.extend(non_hb_geq_fcounts_frame.columns.tolist())
            non_hb_geq_fcounts_frame.transpose().to_excel(e_writer, '{}_{}_fcounts'.format(atlas, task))
            non_hb_geq_hb_fcounts_frame.transpose().to_excel(e_writer, '{}_{}_hb_fcounts'.format(atlas, task))

            # save transposed versions of fset results for plotting
            #fsets_results_frame.transpose().to_excel(e_writer, '{}_{}_fsets_results'.format(atlas, task))
            fsrf.transpose().to_excel(e_writer, '{}_{}_fsets_results'.format(atlas, task))
            #fsets_names_frame.transpose().to_excel(e_writer, '{}_{}_fsets_names'.format(atlas, task))
            fsnf.transpose().to_excel(e_writer, '{}_{}_fsets_names'.format(atlas, task))

            fpi_results_frames[-1].transpose().to_excel(e_writer, '{}_{}_fpi'.format(atlas, task))
            #fpi_results_frames[-1][[c for c in fpi_results_frames[-1] if c in non_hb_geq_fcounts_frame]]\
            #                        .transpose().to_excel(e_writer, '{}_{}_fpi_non_hb_geq'.format(atlas, task))

        # FPIS
        fpi_task_results_frame = combine_fpi_frames(fpi_results_frames)
        fpi_task_results_frame.transpose().to_excel(e_writer, '{}_fpi'.format(task))
        #fpi_task_results_frame[[c for c in fpi_task_results_frame if c in set(non_hb_geq_f)]]\
        #                       .transpose().to_excel(e_writer, '{}_fpi_non_hb_geq'.format(task))

    e_writer.save()
    print('SAVED %s' % xlsx_name)


def scatterplot_fset_results():
    for task in [gbl.clf, gbl.reg]:
        for atlas in gbl.atlas_dict.keys():
            fsets_results_frame = pd.read_excel(io='atlas_{}_maxgridpoints_30_minsupport_0.9.xlsx'.format(atlas),
                                                sheet_name='fsets_results_{}'.format(task), index_col=0)
            fsets_results_frame.sort_values(by='pred_best', axis=1, ascending=True, inplace=True)
            _, hb_idx, _, non_hb_geq_idx = compute_hb_idx(fsets_results_frame)
            fsrf = fsets_results_frame.loc[:, np.array(hb_idx) | np.array(non_hb_geq_idx)].copy()

            fsrft = fsrf.transpose()

            fsrft['sampling'] = ['OverSamp' if 'SMOTE' in tb or 'ROS' in tb else 'NonSamp' for tb in fsrft.tgt_best]
            fsrft['tgt'] = ['2-clf' if '2' in tb else '3-clf' if '3' in tb else '4-clf' if '4' in tb else 'reg'
                            for tb in fsrft.tgt_best]
            fsrft['annot'] = ['S' if 'SMOTE' in tb else 'R' if 'ROS' in tb else '' for tb in fsrft.tgt_best]
            fsrft['est_type'] = fsrft['est_type_best']
            fsrft['idx'] = fsrft.index.tolist()
            print(fsrft)

            dpi = 100
            h=max(0.22*len(fsrft.index), 14)
            fig, ax = plt.subplots(figsize=(8, h), dpi=dpi)

            markers = {'2-clf': 'X', '3-clf': '^', '4-clf': 'd', 'reg': 'o'}
            sizes = {'OverSamp': 14**2, 'NonSamp': 9**2}
            # seaborn
            sns.scatterplot(x='pred_best', y='idx',  data=fsrft, ax=ax,
                            hue='est_type', hue_order=[gbl.linear_, gbl.non_linear_],
                            size='sampling', sizes=sizes, size_order=['NonSamp', 'OverSamp'],
                            style='tgt', markers=markers)

            for i, txt in enumerate(fsrft.annot):
                ax.annotate(txt, (fsrft.pred_best[i], fsrft.index[i]), fontsize='small')
            ax.legend(title='Legend', loc='lower right')#, markerscale=0.8)# fontsize='medium')
            ax.yaxis.set_major_locator(plt.LinearLocator())
            ax.xaxis.set_major_locator(plt.MaxNLocator())

            ax.autoscale(enable=True)
            ax.set_aspect('auto')

            plt.title('{}'.format(atlas))  # , fontsize='x-large')
            plt.ylabel('fset')

            if task==gbl.clf:
                xlabel = 'recall-weighted'
            if task == gbl.reg:
                xlabel = 'mean_abs_error'
            plt.xlabel(xlabel)

            plt.yticks(fsrft.index, weight='bold')#, rotation='vertical')#, fontsize='xx-small',)
            #plt.tick_params(width=2, grid_alpha=0.7)
            plt.grid(True)

            plt.tight_layout()
            plt.savefig('{}_{}_fsr.png'.format(atlas, task))


# def boxplot_fpi_results():
#     for task in [gbl.clf, gbl.reg]:
#         atlas_fpi_dicts = []
#         for atlas in gbl.atlas_dict.keys():
#             with open('{}_fpi_{}_dict.pickle'.format(atlas, task), 'rb') as handle:
#                 atlas_fpi_dicts.append(pickle.load(handle))
#
#             atlas_fpi_frame = pd.DataFrame.from_dict(atlas_fpi_dicts[-1])
#             #atlas_fpi_frame_T = atlas_fpi_frame.transpose(copy=True)
#             atlas_fpi_frame.boxplot(vert=False)
#             plt.title('{} {} Feature Importance'.format(atlas, task))
#             plt.xlabel('mean permutation importance')
#             plt.savefig('{}_{}_fpi'.format(atlas, task))
#
#         all_atlas_fpi_dict = combine_dicts(atlas_fpi_dicts)
#         all_atlas_fpi_frame = pd.DataFrame.from_dict(all_atlas_fpi_dict)
#         #all_atlas_fpi_frame_T = all_atlas_fpi_frame.transpose(copy=True)
#         all_atlas_fpi_frame.boxplot(vert=False)
#         plt.title('{} {} Feature Importance'.format(atlas, task))
#         plt.xlabel('mean permutation importance')
#         plt.savefig('all_{}_fpi'.format(atlas, task))


def scatterplot_fpi_results_helper(frame, atlas, task, w=50, h=30):

    dpi = 100
    h = max(0.22*len(frame.index), 14)
    #h = max(0.20 * np.unique(frame.))
    fig, ax = plt.subplots(figsize=(8, h), dpi=dpi)

    sns.scatterplot('perm_imp_avg', frame.index, ax=ax, data=frame, size=frame.freq, size_norm=(5**2, 20**2),
                    label='pi_avg', markers='o')
    sns.scatterplot('perm_imp_high', frame.index, ax=ax, data=frame, label='pi_high', markers='o')

    ax.legend(title='Legend', loc='lower right')#, markerscale=0.8)#, fontsize='xx-large')

    ax.yaxis.set_major_locator(plt.LinearLocator())

    if task == gbl.clf:
        ax.xaxis.set_major_locator(plt.MultipleLocator(base=0.05))
    elif task == gbl.reg:
        ax.xaxis.set_major_locator(plt.MaxNLocator())

    ax.autoscale(enable=True)
    ax.set_aspect('auto')

    plt.title('{}'.format(str(atlas)))  # , fontsize='large')
    plt.ylabel('feature')#, fontsize='large')
    plt.xlabel('permutation importance')#, fontsize='large')

    plt.yticks(frame.index, weight='bold')#, rotation='vertical')#, fontsize='x-small')
    #plt.tick_params(axis='x', width=2.0, grid_alpha=0.7)
    plt.grid(True)

    plt.tight_layout() # prevent clipping
    plt.savefig('{}_{}_fpi.png'.format(atlas, task))


def scatterplot_fpi_results():
    for task in [gbl.clf, gbl.reg]:
        fpi_results_frames = []
        for atlas in gbl.atlas_dict.keys():
            fpi_results_frames.append(pd.read_excel(io='atlas_{}_maxgridpoints_30_minsupport_0.9.xlsx'.format(atlas),
                                                    sheet_name='fimps_{}'.format(task), index_col=0))

            fpi_results_frames[-1].sort_values(by='perm_imp_avg', axis=1, ascending=True, inplace=True)
            fpi_results_frames[-1].drop(fpi_results_frames[-1].columns[fpi_results_frames[-1].loc['perm_imp_avg', :] <= 0.0], axis=1, inplace=True)
            fpirfT = fpi_results_frames[-1].transpose(copy=True)
            scatterplot_fpi_results_helper(fpirfT, atlas, task)

        fpi_task_results_frame = combine_fpi_frames(fpi_results_frames) # already sorted descending perm_imp_avg
        fpi_task_results_frame.sort_values(by='perm_imp_avg', axis=1, ascending=True, inplace=True)
        fpi_task_results_frame.drop(fpi_task_results_frame.columns[fpi_task_results_frame.loc['perm_imp_avg', :] <= 0.0], axis=1, inplace=True)
        fpitrfT = fpi_task_results_frame.transpose(copy=True)
        if task == 'clf':
            scatterplot_fpi_results_helper(fpitrfT, 'all', task, w=60)
        else:
            scatterplot_fpi_results_helper(fpitrfT, 'all', task)


def transpose_excel(xl_name): # inplace
    xl = pd.ExcelFile(xl_name)
    ewriter = pd.ExcelWriter(xl_name)
    for s in xl.sheet_names:
        frame = pd.read_excel(io=xl_name, sheet_name=s, index_col=0)
        frame.transpose().to_excel(ewriter, s)
    ewriter.save()