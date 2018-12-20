from Survival.FeatureEngineer import FeatureEngineer
from lifelines.utils import concordance_index
import numpy as np
import datetime


def load_whole_data(dataset_idxs, verbose=False, data_path=None):
    """
    Load complete train and test dataset with arbitrarily selected hyper-params

    :param dataset_idxs: list of any combination of 0, 1, 2
    :param verbose
    :param data_path
    :return train_dfs: dict, dataset_name -> train_dataset [DataFrame]
    :return test_dfs: dict, dataset_name -> test_dataset [DataFrame]
    :return unique_times: dict, dataset_name -> LOS checkpoints, from train
    :return dataset_names
    """
    fe = FeatureEngineer(verbose=verbose, data_path=data_path)

    # arbitrarily selected params for feature engineering part (based on cross
    # validation before).
    low_freq_event_thds = [0.02, 0.01, 0.003]
    low_freq_value_thds = [0.01, 0.005, 0.001]

    # ["pancreatitis", "ich", "sepsis"]
    names_ref = fe.get_diseases_list()

    train_dfs = {}
    test_dfs = {}
    unique_times = {}
    dataset_names = []

    for dataset_idx in dataset_idxs:
        dataset_name = names_ref[dataset_idx]
        if verbose:
            print("current dataset:", dataset_name)

        # All data processing logic is in the Feature Engineer Class
        train_df, test_df, feature_list = \
            fe.load_data(src_idx=dataset_idx, file_prefix="",
                         low_freq_event_thd=low_freq_event_thds[dataset_idx],
                         low_freq_value_thd=low_freq_value_thds[dataset_idx])

        train_dfs[dataset_name] = train_df
        test_dfs[dataset_name] = test_df
        unique_times[dataset_name] = sorted(list(train_df["LOS"].unique()))
        dataset_names.append(dataset_name)

    return train_dfs, test_dfs, unique_times, dataset_names


def load_val_data(dataset_idxs, verbose=False, data_path=None):
    """
    Load validation train and test dataset.

    :param dataset_idxs: list of any combination of 0, 1, 2
    :param verbose:
    :param data_path:
    :return train_dfs: dict, dataset_name -> list of train_val_dataset
    :return test_dfs: dict, dataset_name -> list of test_val_dataset [DataFrame]
    :return unique_times: dict, dataset_name -> LOS checkpoints, from train
    :return dataset_names
    """
    fe = FeatureEngineer(verbose=verbose, data_path=data_path)

    # for pancreatitis dataset, the 5 folds division was repeated for 10 times
    fold_nums = [10, 1, 1]

    # arbitrarily selected params for feature engineering part (based on cross
    # validation before).
    low_freq_event_thds = [0.02, 0.01, 0.003]
    low_freq_value_thds = [0.01, 0.005, 0.001]

    # ["pancreatitis", "ich", "sepsis"]
    names_ref = fe.get_diseases_list()

    train_dfs = {}
    test_dfs = {}
    unique_times = {}
    dataset_names = []

    # each dataset index
    for dataset_idx in dataset_idxs:
        dataset_name = names_ref[dataset_idx]
        train_dfs[dataset_name] = []
        test_dfs[dataset_name] = []
        unique_times[dataset_name] = []
        if verbose:
            print("current dataset:", dataset_name)
        dataset_names.append(dataset_name)

        # For validation, we use 5-fold cross validation. For pancreatitis, as
        # the dataset is very small, we repeat the random split for 10 times
        for i in range(fold_nums[dataset_idx] * 5):
            if verbose:
                print("---------------------------------------------")
                print("fold", i)
                print(datetime.datetime.now().strftime("%m%d %H:%M:%S"))

            train_df, test_df, feature_list = \
                fe.load_data(src_idx=dataset_idx,
                             file_prefix="cross_val/" + str(
                                 fold_nums[dataset_idx]) + "-5fold_" + str(
                                 i) + "_",
                             low_freq_event_thd=low_freq_event_thds[
                                 dataset_idx],
                             low_freq_value_thd=low_freq_value_thds[
                                 dataset_idx])

            unique_times[dataset_name].extend(list(train_df["LOS"].unique()))
            train_dfs[dataset_name].append(train_df)
            test_dfs[dataset_name].append(test_df)

        # When doing validation, to make sure we test each validation set with
        # the same checkpoints list (so that the IPEC average make sense), we
        # merged all train unique LOS observation together and then sort
        unique_times[dataset_name] = sorted(
            list(set(unique_times[dataset_name])))

    return train_dfs, test_dfs, unique_times, dataset_names


def load_score_containers(dataset_names, parameters):
    # dimension = tuple([len(parameter) for parameter in parameters])
    # concordances = {dataset_name: np.zeros(dimension) for dataset_name in
    #                 dataset_names}
    # ipecs = {dataset_name: np.zeros(dimension) for dataset_name in
    #          dataset_names}
    # return concordances, ipecs
    pass


def filename_generator(model_name, pca_flag, dataset_idxs,
                       res_path="./"):
    """
    Helper function to generate file name for experiments

    :param res_path:
    :param model_name: "COX", "NNC", "WBR", "KNN", "AAM", "RSF"
    :param pca_flag: true | false
    :param dataset_idxs: list of dataset idx
    :return:
    """
    now = datetime.datetime.now()
    filename = res_path + model_name + "_results/" + model_name + "_"
    if pca_flag:
        filename += "P_"
    for dataset_idx in dataset_idxs:
        filename += str(dataset_idx)
    filename += "_" + now.strftime("%m%d_%H%M") + ".pickle"
    return filename


def calc_scores(model, test_df, checkpoint_times):
    """
    Calculate concordance and ipec (list for different T) of a given model.
    Notice that here we return a list of IPECs. It should has the same length of
    the checkpoint_times. Each IPEC score is the IPEC when use the corresponding
    checkpoint time as T.

    :param model: the model to test on
    :param test_df: test dataset
    :param checkpoint_times: a list of unique time (LOS), used to estimate the
    integral for IPEC

    :return: concordance, ipec
    """
    # Each model has implemented the predict method, which calculate both the
    # predicted median survival time and the survival probability function on
    # each test datapoint.
    pred_median, proba_matrix = model.predict(test_df,
                                              time_list=checkpoint_times)

    obs_test_times = np.array(test_df["LOS"])
    obs_test_events = np.array(test_df["OUT"])

    concordance = concordance_index(obs_test_times, pred_median,
                                    obs_test_events)
    ipec_score_list = calc_ipec_list(proba_matrix, obs_test_times,
                                     obs_test_events, checkpoint_times)

    return concordance, ipec_score_list


def calc_ipec_list(prob_matrix, obs_test_times, obs_test_events, check_points):
    """
    calculate a list of IPEC score for a given results.

    :param prob_matrix: the survival probability function onn each test
    datapoint. In the matrix, each row is the probability distribution of one
    test datapoint; each column is the probability of survival on a checkpoint
    time. i.e., the shape of the matrix is n*m, where n is the number of test
    datapoints (length of obs_test_times), m is the number of checkpoints time
    :param obs_test_times: observed test time
    :param obs_test_events: corresponding observed test event
    :param check_points: a bunch of unique time (LOS) that was used to calculate
    the probability matrix

    :return:
    """
    ###
    # TODO, implement G. currently, G is G(t) = 1.
    # Previously when use IPEC as a python class, we tried G as a constant 1 or
    # Kaplan-Meier predictor (flip the event), which gave almost identical
    # results.

    # Kaplan Meier G would be something like this:
    # def _kaplan_meier_g(obs_times, obs_events):
    #     kmf = KaplanMeierFitter()
    #     kmf.fit(obs_times, obs_events.apply(lambda x: 1 - x))
    #     return kmf.predict
    ###

    # _G_func = lambda time: 1
    # _G_values = [1] * len(obs_times)
    check_points = [0] + check_points

    # We save the result for each T
    ipec_matrix = np.zeros((len(obs_test_times), len(check_points) - 1))

    # IPEC = 1/M * SUM(IPEC_j)
    for obs_j in range(len(obs_test_times)):
        t_j = obs_test_times[obs_j]
        d_j = obs_test_events[obs_j]
        g_j = 1 # temporarily, G(t_j) = 1
        tmp_ipec = 0

        # IPEC formula: *A K-nearest neighbors survival probability prediction
        # method*.
        # Here we substitute the integral with sum
        for i in range(1, len(check_points)):
            t_i = check_points[i]
            t_i_1 = check_points[i - 1]
            g_i = 1 # temporarily, G(t_i) = 1

            # check points list has an addition 0 at the beginning, so the i-1th
            # is the actual ith observation's survival proba
            s_i = prob_matrix[obs_j][i - 1]

            obs_gt_cur = int(t_j > t_i) # I(t_j > t_i)
            obs_lte_cur = 1 - obs_gt_cur # I(t_j <= t_i)

            cur_tmp_ipec = \
                (t_i - t_i_1) * \
                ((d_j * obs_lte_cur / g_j) + (obs_gt_cur / g_i)) * \
                ((obs_gt_cur - s_i) ** 2)

            tmp_ipec += cur_tmp_ipec
            ipec_matrix[obs_j][i - 1] = tmp_ipec

    return np.average(ipec_matrix, axis=0)
