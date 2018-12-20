from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import multiprocessing
import pandas as pd
import numpy as np
import math
from collections import defaultdict


class RandomSurvivalForest():
    def __init__(self, n_trees=30, max_features=20, max_depth=10,
                 min_samples_split=2, split="auto",
                 pca_flag=False, n_components=20):
        """
        The Random Survival Forest Model is implemented from scratch. This class
        is modified based on Wrymm's implementation at
        https://github.com/Wrymm/Random-Survival-Forests
        The original paper can be found at Random Survival Forests for R
        https://pdfs.semanticscholar.org/951a/84f0176076fb6786fdf43320e8b27094dcfa.pdf

        NOTICE: Wrymm provide an implementation to calculate the survival
        probability at a given time. we extend that to predict the median
        survival time. In the original paper, it actually calculate the
        comulative hazard function first. I've also included that implementation
        (which is currently commented). The problem is that the two
        implementations yield different median time prediction. This may be
        something that worth inspection.

        :param n_trees:
        :param max_features:
        :param max_depth:
        :param min_samples_split:
        :param split: "auto" or "logrank", both will use logrank
        :param pca_flag: use PCA before fit the model.
        :param n_components: number of principle components for PCA
        """
        self._n_trees = n_trees
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._split = split
        # only logrank is implemented
        self._split_func = {"auto": self._logrank,
                            "logrank": self._logrank}
        self._max_features = max_features
        # name of the time column used within the class. this can be anything
        self._time_column = "time"
        # name of the event column used within the class. this can be anything
        self._event_column = "event"
        self.pca_flag = pca_flag
        self.pca = PCA(n_components=n_components)
        self.standardalizer = StandardScaler()

    def _train_pca(self, train_df):
        """
        Conduct PCA dimension reduction for train dataset if pca_flag is true.

        :param train_df: original train DataFrame
        :return: train DataFrame after dimension reduction
        """
        if self.pca_flag:
            train_x = train_df.drop(
                columns=[self._duration_col, self._event_col]).values
            # standardize
            self.standardalizer.fit(train_x)
            train_x = self.standardalizer.transform(train_x)
            # fit and transform
            self.pca.fit(train_x)
            reduced_x = self.pca.transform(train_x)
            # convert back to DataFrame
            reduced_train_df = pd.DataFrame(reduced_x).set_index(train_df.index)
            # we don't care about the column name here
            columns = ["C" + str(i) for i in range(reduced_train_df.shape[1])]
            reduced_train_df.columns = columns
            # get back the y (LOS and OUT)
            reduced_train_df[self._duration_col] = train_df[self._duration_col]
            reduced_train_df[self._event_col] = train_df[self._event_col]
            return reduced_train_df
        else:
            return train_df

    def _test_pca(self, test_df):
        """
        Conduct PCA dimension reduction for test dataset if pca_flag is true.

        :param test_df: original test DataFrame
        :return: test DataFrame after dimension reduction
        """
        if self.pca_flag:
            test_x = test_df.drop(
                columns=[self._duration_col, self._event_col]).values
            # standardize with the same rules for the train
            test_x = self.standardalizer.transform(test_x)
            # transform
            reduced_x = self.pca.transform(test_x)
            # convert back to DataFrame
            reduced_test_df = pd.DataFrame(reduced_x).set_index(test_df.index)
            columns = ["C" + str(i) for i in range(reduced_test_df.shape[1])]
            reduced_test_df.columns = columns
            reduced_test_df[self._duration_col] = test_df[self._duration_col]
            reduced_test_df[self._event_col] = test_df[self._event_col]
            return reduced_test_df
        else:
            return test_df

    def _logrank(self, x, feature):
        """
        This implementation use logrank to select feature. This function calc.
        the logrank for the given feature.
        The logrank is used as the information gain of splitting on the given
        feature

        :param x: Dataframe, the dataset at this node
        :param feature: string, the given feature
        :return:
        """
        c = x[feature].median()
        if x[x[feature] <= c].shape[0] < self._min_samples_split or \
                        x[x[feature] > c].shape[0] < self._min_samples_split:
            return 0
        t = list(set(x[self._time_column]))
        get_time = {t[i]: i for i in range(len(t))}
        N = len(t)
        y = np.zeros((3, N))
        d = np.zeros((3, N))
        feature_inf = x[x[feature] <= c]
        feature_sup = x[x[feature] > c]
        count_sup = np.zeros((N, 1))
        count_inf = np.zeros((N, 1))
        for _, r in feature_sup.iterrows():
            t_idx = get_time[r[self._time_column]]
            count_sup[t_idx] = count_sup[t_idx] + 1
            if r[self._event_column]:
                d[2][t_idx] = d[2][t_idx] + 1
        for _, r in feature_inf.iterrows():
            t_idx = get_time[r[self._time_column]]
            count_inf[t_idx] = count_inf[t_idx] + 1
            if r[self._event_column]:
                d[1][t_idx] = d[1][t_idx] + 1
        nb_inf = feature_inf.shape[0]
        nb_sup = feature_sup.shape[0]
        for i in range(N):
            y[1][i] = nb_inf
            y[2][i] = nb_sup
            y[0][i] = y[1][i] + y[2][i]
            d[0][i] = d[1][i] + d[2][i]
            nb_inf = nb_inf - count_inf[i]
            nb_sup = nb_sup - count_sup[i]
        num = 0
        den = 0
        for i in range(N):
            if y[0][i] > 0:
                num = num + d[1][i] - y[1][i] * d[0][i] / float(y[0][i])
            if y[0][i] > 1:
                den += (y[1][i] / float(y[0][i])) * y[2][i] * \
                       ((y[0][i] - d[0][i]) / (y[0][i] - 1)) * d[0][i]
        L = num / math.sqrt(den)
        return abs(L)

    def _find_best_feature(self, x):
        """
        Select the best feature to split on

        :param x: DataFrame, the dataset on this node
        :return:
        """
        features = [f for f in x.columns
                    if f not in {self._time_column, self._event_column}]
        # randomly sample features at each split.
        # Notice that the original implementation by Wrymm does not randomly
        # sample feature at each split. Instead, for each tree, Wrymm randomly
        # samples feature at the beginning, and the whole tree will only use
        # the sampled features.
        # Refer back the Wrymm's implementation if the original implementation
        # is preferred.
        sample_feas = list(np.random.permutation(features))[:self._max_features]
        information_gains = [self._split_func[self._split](x, feature)
                             for feature in sample_feas]
        highest_IG = max(information_gains)
        # This implementation use a greedy stop condition: if a node's all
        # splits do not provide information gain, stop
        if highest_IG == 0:
            return None
        else:
            return features[information_gains.index(highest_IG)]

    def _compute_leaf(self, x, tree):
        """
        if a node is a leaf, compute a bunch of information to support future
        prediction.

        :param x: DataFrame, the dataset at this node (leaf)
        :param tree: the current node
        :return:
        """
        tree["type"] = "leaf"

        # for each distinct death time, count the number of death and
        # individuals at risk at that time.
        # this will be used to calculate the cumulative hazard estimation
        unique_observed_times = np.array(x[self._time_column].unique())
        sorted_unique_times = np.sort(unique_observed_times)
        death_and_risk_at_time = defaultdict(lambda: {"death": 0, "risky": 0})

        # The count is used to calculate survival probability
        # coded by the Wrymm
        count = {}
        for _, r in x.iterrows():
            # update count
            count.setdefault((r[self._time_column], 0), 0)
            count.setdefault((r[self._time_column], 1), 0)
            count[(r[self._time_column], r[self._event_column])] += 1

        # used later to calculate survival probability
        # written by the original author
        total = x.shape[0]
        tree["count"] = count
        tree["t"] = sorted_unique_times
        tree["total"] = total

        # # -------------------------------------------------------------------
        #
        # # count the number of death and at risk at each observed time
        # # coded by the Ren Zuo
        # # based on Random Survival Forests for R
        # for _, r in x.iterrows():
        #     for observed_time in sorted_unique_times:
        #         if observed_time < r[self._time_column]:
        #             death_and_risk_at_time[observed_time]["risky"] += 1.
        #         elif observed_time == r[self._time_column]:
        #             if r[self._event_column] == 1:
        #                 death_and_risk_at_time[observed_time]["death"] += 1.
        #             else:
        #                 death_and_risk_at_time[observed_time]["risky"] += 1.
        #         else:
        #             break
        # # calculate the cumulative hazard function
        # # on the observed time
        # #
        # # the cumulative hazard function for the node is defined as:
        # # H(t) = sum_{t_l <= t} d_l / r_l
        # # where t_l is the distinct death times in this leaf node,
        # # and d_l and r_l equal the number of deaths and individuals
        # # at risk at time t_l.
        # cumulative_hazard = 0
        # cumulative_hazard_function = {}
        # for observed_time in sorted_unique_times:
        #     if death_and_risk_at_time[observed_time]["risky"] != 0:
        #         cumulative_hazard += \
        #             float(death_and_risk_at_time[observed_time]["death"]) / \
        #             float(death_and_risk_at_time[observed_time]["risky"])
        #     else:
        #         cumulative_hazard += 1
        #     cumulative_hazard_function[observed_time] = cumulative_hazard
        # tree["cumulative_hazard_function"] = cumulative_hazard_function

    def _build(self, x, tree, depth):
        """
        Build a regression tree recursively.

        :param x: DataFrame at current node
        :param tree: the current node
        :param depth: current depth
        :return:
        """
        # reach the leaf node.
        if len(pd.unique(x[self._time_column])) == 1 or \
                        depth == self._max_depth:
            self._compute_leaf(x, tree)
            return

        # node the leaf. find the best feature to split on. use logrank.
        best_feature = self._find_best_feature(x)

        # no split can be found. treat this node as a leaf.
        if best_feature is None:
            self._compute_leaf(x, tree)
            return

        # store the splitting information
        feature_median = x[best_feature].median()
        tree["type"] = "node"
        tree["feature"] = best_feature
        tree["median"] = feature_median

        # split
        left_split_x = x[x[best_feature] <= feature_median]
        right_split_x = x[x[best_feature] > feature_median]
        split_dict = [["left", left_split_x], ["right", right_split_x]]
        for name, split_x in split_dict:
            # next node
            tree[name] = {}
            # build the next node
            self._build(split_x, tree[name], depth + 1)

    # def _find_cumulative_hazard(self, tree, row):
    #     """
    #     find cumulative hazard in a tree for an individual i
    #
    #     :param tree:
    #     :param row: the feature vector (Series) of the individual i
    #     :return: the cumulative hazard function
    #     """
    #     if tree["type"] == "leaf":
    #         return tree["cumulative_hazard_function"]
    #     else:
    #         if row[tree["feature"]] > tree["median"]:
    #             return self._find_cumulative_hazard(tree["right"], row)
    #         else:
    #             return self._find_cumulative_hazard(tree["left"], row)

    # def _get_ensemble_cumulative_hazard(self, row):
    #     """
    #     Average the cumulative hazard given by each tree.
    #
    #     :param row:
    #     :return:
    #     """
    #     hazard_functions = [self._find_cumulative_hazard(tree, row)
    #                         for tree in self._trees]
    #
    #     # make sure every hazard_function generated by different tree will
    #     # have a cumulative hazard value for each observed time.
    #     # The problem here is that a cumulative hazard function generate by
    #     # one tree will only have hazard value on a small set of time points.
    #     # We need to complete each cumulative hazard function
    #     for observed_time in np.sort(self._times):
    #         supplement_hazard_value = 0
    #         for index, hazard_function in enumerate(hazard_functions):
    #             if observed_time not in hazard_function:
    #                 hazard_function[observed_time] = supplement_hazard_value
    #             else:
    #                 supplement_hazard_value = hazard_function[observed_time]
    #     # calculate average to generate the ensemble hazard function
    #     ensemble_cumulative_hazard = {}
    #     for observed_time in self._times:
    #         ensemble_cumulative_hazard[observed_time] = \
    #             np.mean([hazard_function[observed_time]
    #                      for hazard_function in hazard_functions])
    #     return ensemble_cumulative_hazard

    # def _estimate_median_time(self, hazard_function, avg_flag):
    #     """
    #     Given the ennsemble cumulative hazard function, find the median time
    #
    #     :param hazard_function:
    #     :param avg_flag:
    #     :return:
    #     """
    #     log_two = np.log(2)
    #     prev_time = 0
    #     final_time = np.inf
    #     for i in range(len(self._times)):
    #         if hazard_function[self._times[i]] == log_two:
    #             return self._times[i]
    #         elif hazard_function[self._times[i]] < log_two:
    #             prev_time = self._times[i]
    #         else:
    #             if avg_flag:
    #                 return (prev_time + self._times[i]) / 2.
    #             else:
    #                 return self._times[i]
    #     return (prev_time + final_time) / 2.

    def _pred_survival(self, tree, row, checkpoints):
        """
        On a given tree, for a given test data point (row), find the survival
        function (survival probability on given checkpoints).

        :param tree:
        :param row:
        :param checkpoints:
        :return:
        """
        # find the leaf node
        if tree["type"] == "leaf":
            count = tree["count"]
            result = []
            # for each checkpoint, calculate the survival probability
            for checkpoint in checkpoints:
                survivors = float(tree["total"])
                tmp_proba = 1
                # calculate the survival probability
                for ti in tree["t"]:
                    if ti <= checkpoint:
                        tmp_proba *= (1 - count[(ti, 1)] / survivors)
                    survivors = survivors - count[(ti, 1)] - count[(ti, 0)]
                result.append(tmp_proba)
            return result
        else:
            if row[tree["feature"]] > tree["median"]:
                return self._pred_survival(tree["right"], row, checkpoints)
            else:
                return self._pred_survival(tree["left"], row, checkpoints)

    def _grow_tree(self, data):
        """
        Build an individual tree.

        :param data:
        :return:
        """
        new_tree = {}
        self._build(data, new_tree, 0)
        return new_tree

    def fit(self, train_df, duration_col='LOS', event_col='OUT',
            num_workers=36):
        """
        Build the forest.

        :param train_df: DataFrame, with the duration and the event column
        :param duration_col: the column name for duration
        :param event_col: the column name for event
        :param num_workers:
        """
        self._duration_col = duration_col
        self._event_col = event_col
        reduced_train_df = self._train_pca(train_df)
        x_train = reduced_train_df.drop(columns=[duration_col, event_col])
        y_train = reduced_train_df[[duration_col, event_col]]
        y_train.columns = [self._time_column, self._event_column]
        self._times = np.sort(list(y_train[self._time_column].unique()))
        x_train = pd.concat((x_train, y_train), axis=1)
        x_train = x_train.sort_values(by=self._time_column)
        x_train.index = range(x_train.shape[0])
        sampled_datas = []
        for i in range(self._n_trees):
            sampled_x = x_train.sample(frac=1, replace=True)
            sampled_x.index = range(sampled_x.shape[0])
            sampled_datas.append(sampled_x)
        with multiprocessing.Pool(num_workers) as tmp_pool:
            trees = tmp_pool.map(self._grow_tree, sampled_datas)
        self._trees = trees

    def pred_proba(self, test_df, time):
        """
        :param test_df: DataFrame
        :param time: checkpoint time to calculate probability on
        :return: the probability matrix. each row is the survival function for
        one test data point.
        """
        reduced_test_df = self._test_pca(test_df)

        if isinstance(time, int) or isinstance(time, float):
            time_list = [time]
        else:
            time_list = time

        proba_pred = []
        for _, row in reduced_test_df.iterrows():
            # for each tree, get the probability distribution
            probas = [self._pred_survival(self._trees[i], row, time_list)
                      for i in range(self._n_trees)]
            # average the probability of each tree
            proba_pred.append(list(np.average(np.array(probas), axis=0)))

        return np.array(proba_pred)

    def pred_median_time(self, test_df, average_to_get_median=True):
        """
        :param test_df: DataFrame
        :return: the list of median survival time for each test datapoint.
        """
        # # Below is the median time prediction using the ensemble cumulative
        # # hazard, based on the paper
        # result = []
        # reduced_test_df = self._test_pca(test_df)
        # for _, row in reduced_test_df.iterrows():
        #     cumulative_hazard = self._get_ensemble_cumulative_hazard(row)
        #     result.append(self._estimate_median_time(
        #         cumulative_hazard, average_to_get_median))
        # return np.array(result)

        #----------------------------------------------------------------------

        # Below is the median time prediction using the original implementation
        proba_matrix = self.pred_proba(test_df, self._times)

        pred_medians = []
        median_time = 0
        for test_idx, survival_proba in enumerate(proba_matrix):
            # the survival_proba is in descending order
            for col, proba in enumerate(survival_proba):
                if proba > 0.5:
                    continue
                if proba == 0.5 or col == 0:
                    median_time = self._times[col]
                else:
                    median_time = (self._times[col - 1] + self._times[col]) / 2
                break
            pred_medians.append(median_time)

        return np.array(pred_medians)

    def predict(self, test_df, time_list):
        """
        :param test_df: DataFrame
        :param time_list: checkpoint time to calculate probability on
        :return: the list of median survival time and the probability matrix
        """
        proba_matrix = self.pred_proba(test_df, time_list)
        pred_medians = []
        median_time = 0
        for test_idx, survival_proba in enumerate(proba_matrix):
            # the survival_proba is in descending order
            for col, proba in enumerate(survival_proba):
                if proba > 0.5:
                    continue
                if proba == 0.5 or col == 0:
                    median_time = time_list[col]
                else:
                    median_time = (time_list[col - 1] + time_list[col]) / 2
                break
            pred_medians.append(median_time)

        return np.array(pred_medians), proba_matrix
