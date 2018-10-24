import pandas as pd
import numpy as np
from fancyimpute import SoftImpute
import lifelines


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
