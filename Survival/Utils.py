from Survival.FeatureEngineer import FeatureEngineer
from Survival.IPEC import IPEC

import pandas as pd
import numpy as np
from fancyimpute import SoftImpute
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


def model_prepare(patient_dict, feature_set, train_id_list, test_id_list):
    patient_df = \
        pd.DataFrame.from_dict(patient_dict, orient='index')[feature_set]

    # missing value imputation using SoftImpute
    imputed_result = SoftImpute(verbose=False).complete(
        patient_df.drop(columns=["LOS", "OUT"]).values)
    imputed_patient_df = pd.DataFrame(imputed_result,
        columns=patient_df.drop(columns=["LOS", "OUT"]).columns).set_index(
        patient_df.index.values)
    imputed_patient_df["LOS"] = patient_df["LOS"]
    imputed_patient_df["OUT"] = patient_df["OUT"]

    # remove correlated features
    dup_feature = corr_feature_filter(imputed_patient_df, corr_thd=1)
    #     print("Number of dumplicated featues:", len(dup_feature))
    imputed_patient_df.drop(columns=dup_feature, inplace=True)

    # remove rows that has infinite LoS
    inf_id_set = \
        set(imputed_patient_df[imputed_patient_df["LOS"] == np.inf].index)
    train_id_list = list(set(train_id_list) - inf_id_set)
    test_id_list = list(set(test_id_list) - inf_id_set)

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


def evaluate_predict_result(test_time_median_pred, test_df, test_time_true=None,
                            print_result=False, show_time=False):
    if test_time_true is None:
        test_time_true = np.array(test_df["LOS"])
    test_out_value = np.array(test_df["OUT"])
    test_out_flag = np.array(list(test_df["OUT"].apply(lambda x: x == 1)))
    pred_out_flag = test_time_median_pred != np.inf
    out_flag = test_out_flag == pred_out_flag
    concordance_value = lifelines.utils.concordance_index(test_time_true,
        test_time_median_pred, test_out_value)

    if print_result:
        print("concordance:", concordance_value)
    if show_time:
        print("----------")
        for index, value in enumerate(test_time_median_pred):
            print("pred:", value, "; true:", test_time_true[index])

    return concordance_value


def load_val_data(dataset_idxs, verbose=False):
    fe = FeatureEngineer(verbose=verbose)

    fold_nums = [10, 1, 1]
    low_freq_event_thds = [0.02, 0.01, 0.003]
    low_freq_value_thds = [0.01, 0.005, 0.001]
    names_ref = fe.get_diseases_list()

    train_dfs = {}
    test_dfs = {}
    dataset_names = []

    for dataset_idx in dataset_idxs:
        dataset_name = names_ref[dataset_idx]
        train_dfs[dataset_name] = []
        test_dfs[dataset_name] = []
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
                model_prepare(patient_dict, feature_set, train_id_list, test_id_list)
            train_dfs[dataset_name].append(train_df)
            test_dfs[dataset_name].append(test_df)

    return train_dfs, test_dfs, dataset_names


def load_score_containers(dataset_names, parameters):
    dimension = tuple([len(parameter) for parameter in parameters])
    concordances = {dataset_name:np.zeros(dimension) for dataset_name in dataset_names}
    ipecs = {dataset_name:np.zeros(dimension) for dataset_name in dataset_names}
    return concordances, ipecs


def calc_scores(model, cur_train, cur_test):
    ipec = IPEC(cur_train, g_type="All_One", t_thd=0.8, 
        t_step="obs", time_col='LOS', death_identifier='OUT')
    test_time_median_pred = model.pred_median_time(cur_test)
    proba_matrix = \
        model.pred_proba(cur_test, time=ipec.get_check_points())
    concordance = evaluate_predict_result(test_time_median_pred, 
        cur_test, print_result=False)
    ipec_score = ipec.calc_ipec(proba_matrix, 
        list(cur_test["LOS"]), list(cur_test["OUT"]))
    return concordance, ipec_score


def filename_generator(model_name, pca_flag, dataset_idxs):
    now = datetime.datetime.now()
    filename = model_name + "_results/" + model_name + "_"
    if pca_flag:
        filename += "P_"
    for dataset_idx in dataset_idxs:
        filename += str(dataset_idx)
    filename += "_" + now.strftime("%m%d_%H%M") + ".pickle"
    return filename



