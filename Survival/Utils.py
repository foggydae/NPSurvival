from Survival.FeatureEngineer import FeatureEngineer
from Survival.IPEC import IPEC

import pandas as pd
import numpy as np
from fancyimpute import SoftImpute, BiScaler
import lifelines
import datetime


def corr_feature_filter(df, corr_thd=1, method='pearson'):
    corr_df = df.corr(method=method)
    dup_feature = []
    for index, feature1 in enumerate(corr_df.columns):
        for feature2 in corr_df[feature1].index[index + 1:]:
            if np.abs(corr_df[feature1][feature2]) >= corr_thd and \
                            feature1 != feature2:
                dup_feature.append(feature2)
    return dup_feature


def model_prepare(patient_dict, feature_set, train_id_list, test_id_list, verbose=False):
    patient_df = \
        pd.DataFrame.from_dict(patient_dict, orient='index')[feature_set]

    if verbose:
        print("[LOG]Impute missing value.")
    # missing value imputation using SoftImpute
    # X_incomplete_normalized = BiScaler(verbose=False).fit_transform(
    #     patient_df.drop(columns=["LOS", "OUT"]).values)
    imputed_result = SoftImpute(verbose=False).complete(
        patient_df.drop(columns=["LOS", "OUT"]).values)
    imputed_patient_df = pd.DataFrame(imputed_result,
        columns=patient_df.drop(columns=["LOS", "OUT"]).columns).set_index(
        patient_df.index.values)
    imputed_patient_df["LOS"] = patient_df["LOS"]
    imputed_patient_df["OUT"] = patient_df["OUT"]

    # if verbose:
    #     print("[LOG]Remove correlated features.")
    # # remove correlated features
    # dup_feature = corr_feature_filter(imputed_patient_df, corr_thd=1)
    # if verbose:
    #     print("    Number of dumplicated featues:", len(dup_feature))
    # imputed_patient_df.drop(columns=dup_feature, inplace=True)

    # if verbose:
    #     print("[LOG]Remove rows that has infinite LoS.")
    # # remove rows that has infinite LoS
    # inf_id_set = \
    #     set(imputed_patient_df[imputed_patient_df["LOS"] == np.inf].index)
    # train_id_list = list(set(train_id_list) - inf_id_set)
    # test_id_list = list(set(test_id_list) - inf_id_set)

    # feature matrix preparation, train & test splitting
    train_df = imputed_patient_df.loc[train_id_list]
    test_df = imputed_patient_df.loc[test_id_list]
    feature_list = np.array(train_df.drop(columns=["LOS", "OUT"]).columns)

    return train_df, test_df, feature_list


def calculate_dataset_size(dataset_idx, patient_dict, feature_set,
                           train_id_list, test_id_list, sources):
    print("Dataset name:", sources[dataset_idx])
    print("patient number:", len(patient_dict.keys()))
    print("train:", len(train_id_list), "; test:", len(test_id_list))
    print("feature number:", len(feature_set) - 2)


def calc_concordance(test_time_median_pred, test_df, print_result=False):
    test_time_true = np.array(test_df["LOS"])
    test_out_value = np.array(test_df["OUT"])
    concordance_value = lifelines.utils.concordance_index(test_time_true,
        test_time_median_pred, test_out_value)
    if print_result:
        print("concordance:", concordance_value)
    return concordance_value


def load_val_data(dataset_idxs, verbose=False, data_path=None):
    fe = FeatureEngineer(verbose=verbose, data_path=data_path)

    fold_nums = [10, 1, 1]
    low_freq_event_thds = [0.02, 0.01, 0.003]
    low_freq_value_thds = [0.01, 0.005, 0.001]
    names_ref = fe.get_diseases_list()

    train_dfs = {}
    test_dfs = {}
    unique_times = {}
    dataset_names = []

    for dataset_idx in dataset_idxs:
        dataset_name = names_ref[dataset_idx]
        train_dfs[dataset_name] = []
        test_dfs[dataset_name] = []
        unique_times[dataset_name] = []
        if verbose:
            print("current dataset:", dataset_name)
        dataset_names.append(dataset_name)
        for i in range(fold_nums[dataset_idx] * 5):
            if verbose:
                print("---------------------------------------------")
                print("fold", i)
                print(datetime.datetime.now().strftime("%m%d %H:%M:%S"))
            patient_dict, feature_set, train_id_list, test_id_list = \
                fe.load_data_as_dict(src_idx=dataset_idx, 
                    file_prefix="cross_val/"+str(fold_nums[dataset_idx])+"-5fold_"+str(i)+"_", 
                    low_freq_event_thd=low_freq_event_thds[dataset_idx], 
                    low_freq_value_thd=low_freq_value_thds[dataset_idx])
            train_df, test_df, feature_list = \
                model_prepare(patient_dict, feature_set, train_id_list, test_id_list, verbose=verbose)
            unique_times[dataset_name].extend(list(train_df["LOS"].unique()))
            train_dfs[dataset_name].append(train_df)
            test_dfs[dataset_name].append(test_df)
        unique_times[dataset_name] = sorted(list(set(unique_times[dataset_name])))

    return train_dfs, test_dfs, unique_times, dataset_names


def load_score_containers(dataset_names, parameters):
    dimension = tuple([len(parameter) for parameter in parameters])
    concordances = {dataset_name:np.zeros(dimension) for dataset_name in dataset_names}
    ipecs = {dataset_name:np.zeros(dimension) for dataset_name in dataset_names}
    return concordances, ipecs


def calc_scores(model, cur_test, sorted_unique_time, print_result=False, verbose_calc_time=False):
    # init = datetime.datetime.now()
    # start = datetime.datetime.now()

    # ipec = IPEC(cur_train, g_type="All_One", t_thd=0.8, 
    #     t_step="obs", time_col='LOS', death_identifier='OUT', verbose=False)
    # if verbose_calc_time:
    #     print("init ipec class:", datetime.datetime.now() - start)
    #     start = datetime.datetime.now()

    # test_time_median_pred = model.pred_median_time(cur_test)
    # if verbose_calc_time:
    #     print("calc median time:", datetime.datetime.now() - start)
    #     start = datetime.datetime.now()

    # proba_matrix = \
    #     model.pred_proba(cur_test, time=ipec.get_check_points())
    # if verbose_calc_time:
    #     print("calc survival matrix:", datetime.datetime.now() - start)
    #     start = datetime.datetime.now()

    # concordance = calc_concordance(test_time_median_pred, 
    #     cur_test, print_result=print_result, show_time=False)
    # if verbose_calc_time:
    #     print("calc concordance:", datetime.datetime.now() - start)
    #     start = datetime.datetime.now()

    # ipec_score = ipec.calc_ipec(proba_matrix, 
    #     list(cur_test["LOS"]), list(cur_test["OUT"]), print_result=print_result)
    # if verbose_calc_time:
    #     print("calc ipec:", datetime.datetime.now() - start)
    #     print("total:", datetime.datetime.now() - init)

    # return concordance, ipec_score

    init = datetime.datetime.now()
    start = datetime.datetime.now()

    test_time_median_pred, proba_matrix = \
        model.predict(cur_test, time_list=sorted_unique_time)
    if verbose_calc_time:
        print("calc median time & survival matrix:", datetime.datetime.now() - start)
        start = datetime.datetime.now()

    concordance = calc_concordance(test_time_median_pred, 
        cur_test, print_result=print_result)
    if verbose_calc_time:
        print("calc concordance:", datetime.datetime.now() - start)
        start = datetime.datetime.now()

    ipec_score_list = calc_ipec_list(proba_matrix, 
        cur_test, sorted_unique_time)
    if verbose_calc_time:
        print("calc ipec:", datetime.datetime.now() - start)
        print("total:", datetime.datetime.now() - init)

    return concordance, ipec_score_list

def filename_generator(model_name, pca_flag, dataset_idxs):
    now = datetime.datetime.now()
    filename = model_name + "_results/" + model_name + "_"
    if pca_flag:
        filename += "P_"
    for dataset_idx in dataset_idxs:
        filename += str(dataset_idx)
    filename += "_" + now.strftime("%m%d_%H%M") + ".pickle"
    return filename


def calc_ipec_list(prob_matrix, test_df, check_points):

    obs_test_times = list(test_df["LOS"])
    obs_test_events = list(test_df["OUT"])
    # _G_func = lambda time: 1
    # _G_values = [1] * len(obs_times)
    check_points = [0] + check_points

    ipec_matrix = np.zeros((len(obs_test_times), len(check_points)-1))
    for obs_j in range(len(obs_test_times)):
        t_j = obs_test_times[obs_j]
        d_j = obs_test_events[obs_j]
        g_j = 1
        tmp_ipec = 0

        for i in range(1, len(check_points)):
            t_i = check_points[i]
            t_i_1 = check_points[i - 1]
            g_i = 1
            s_i = prob_matrix[obs_j][i - 1] # check points list has an addition 0 at the beginning, so the i-1 is the true ith observation
            obs_gt_cur = int(t_j > t_i)
            obs_lte_cur = 1 - obs_gt_cur

            cur_tmp_ipec = \
                (t_i - t_i_1) * \
                ((d_j * obs_lte_cur / g_j) + (obs_gt_cur / g_i)) * \
                ((obs_gt_cur - s_i) ** 2)

            tmp_ipec += cur_tmp_ipec
            ipec_matrix[obs_j][i-1] = tmp_ipec

    return np.average(ipec_matrix, axis=0)



