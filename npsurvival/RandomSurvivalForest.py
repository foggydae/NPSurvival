from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import multiprocessing
import pandas as pd
import numpy as np
import math
from collections import defaultdict


class RandomSurvivalForest():
    def __init__(self, n_trees=30, max_features=20, max_depth=10,
                 min_samples_split=2, split="auto", pca_flag=False, n_components=20):
        self._n_trees = n_trees
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._split = split
        self._max_features = max_features
        self._time_column = "time"
        self._event_column = "event"
        self.pca_flag = pca_flag
        self.pca = PCA(n_components=n_components)
        self.standardalizer = StandardScaler()

    def _train_pca(self, train_df):
        if not self.pca_flag:
            return train_df
        train_x = \
            train_df.drop(columns=[self._duration_col, self._event_col]).values
        self.standardalizer.fit(train_x)
        train_x = self.standardalizer.transform(train_x)
        self.pca.fit(train_x)
        reduced_x = self.pca.transform(train_x)
        reduced_train_df = pd.DataFrame(reduced_x).set_index(train_df.index)
        columns = ["C" + str(i) for i in range(reduced_train_df.shape[1])]
        reduced_train_df.columns = columns
        reduced_train_df["LOS"] = train_df["LOS"]
        reduced_train_df["OUT"] = train_df["OUT"]
        return reduced_train_df

    def _test_pca(self, test_df):
        if not self.pca_flag:
            return test_df
        test_x = \
            test_df.drop(columns=[self._duration_col, self._event_col]).values
        test_x = self.standardalizer.transform(test_x)
        reduced_x = self.pca.transform(test_x)
        reduced_test_df = pd.DataFrame(reduced_x).set_index(test_df.index)
        columns = ["C" + str(i) for i in range(reduced_test_df.shape[1])]
        reduced_test_df.columns = columns
        reduced_test_df["LOS"] = test_df["LOS"]
        reduced_test_df["OUT"] = test_df["OUT"]
        return reduced_test_df

    def _logrank(self, x, feature):
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
        split_func = {"auto": self._logrank}
        features = [f for f in x.columns
                    if f not in {self._time_column, self._event_column}]
        sample_feas = list(np.random.permutation(features))[:self._max_features]
        information_gains = [split_func[self._split](x, feature)
                             for feature in sample_feas]
        highest_IG = max(information_gains)
        if highest_IG == 0:
            return None
        else:
            return features[information_gains.index(highest_IG)]

    def _compute_leaf(self, x, tree):
        tree["type"] = "leaf"
        # for each distinct death time, count the number of death and
        # individuals at risk at that time.
        # this will be used to calculate the cumulative hazard estimate
        unique_observed_times = np.array(x[self._time_column].unique())
        sorted_unique_times = np.sort(unique_observed_times)
        death_and_risk_at_time = defaultdict(lambda: {"death": 0, "risky": 0})
        # used to calculate survival probability
        # written by the original author
        count = {}
        for _, r in x.iterrows():
            # count the number of death and individuals at risk at each
            # observed time
            for observed_time in sorted_unique_times:
                if observed_time < r[self._time_column]:
                    death_and_risk_at_time[observed_time]["risky"] += 1.
                elif observed_time == r[self._time_column]:
                    if r[self._event_column] == 1:
                        death_and_risk_at_time[observed_time]["death"] += 1.
                    else:
                        death_and_risk_at_time[observed_time]["risky"] += 1.
                else:
                    break
            # update count
            count.setdefault((r[self._time_column], 0), 0)
            count.setdefault((r[self._time_column], 1), 0)
            count[(r[self._time_column], r[self._event_column])] += 1

        # used to calculate survival probability
        # written by the original author
        total = x.shape[0]
        tree["count"] = count
        tree["t"] = sorted_unique_times
        tree["total"] = total

        # calculate the cumulative hazard function
        # on the observed time
        #
        # the cumulative hazard function for the node is defined as:
        # H(t) = sum_{t_l < t} d_l / r_l
        # where {t_l} is the distinct death times in this leaf node,
        # and d_l and r_l equal the number of deaths and individuals
        # at risk at time t_l.
        cumulated_hazard = 0
        cumulative_hazard_function = {}
        for observed_time in sorted_unique_times:
            if death_and_risk_at_time[observed_time]["risky"] != 0:
                cumulated_hazard += \
                    float(death_and_risk_at_time[observed_time]["death"]) / \
                    float(death_and_risk_at_time[observed_time]["risky"])
            else:
                cumulated_hazard += 1
            cumulative_hazard_function[observed_time] = cumulated_hazard
        tree["cumulative_hazard_function"] = cumulative_hazard_function

    def _build(self, x, tree, depth):
        unique_targets = pd.unique(x[self._time_column])

        if len(unique_targets) == 1 or depth == self._max_depth:
            self._compute_leaf(x, tree)
            return

        best_feature = self._find_best_feature(x)

        if best_feature == None:
            self._compute_leaf(x, tree)
            return

        feature_median = x[best_feature].median()

        tree["type"] = "node"
        tree["feature"] = best_feature
        tree["median"] = feature_median

        left_split_x = x[x[best_feature] <= feature_median]
        right_split_x = x[x[best_feature] > feature_median]
        split_dict = [["left", left_split_x], ["right", right_split_x]]

        for name, split_x in split_dict:
            tree[name] = {}
            self._build(split_x, tree[name], depth + 1)

    def _pred_hazard(self, tree, row):
        if tree["type"] == "leaf":
            return tree["cumulative_hazard_function"]
        else:
            if row[tree["feature"]] > tree["median"]:
                return self._pred_hazard(tree["right"], row)
            else:
                return self._pred_hazard(tree["left"], row)

    def _pred_survival(self, tree, row, checkpoints):
        if tree["type"] == "leaf":
            count = tree["count"]
            result = []
            for checkpoint in checkpoints:
                survivors = float(tree["total"])
                tmp_proba = 1
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

    def _get_ensemble_cumulative_hazard(self, row):
        hazard_functions = [self._pred_hazard(tree, row)
                            for tree in self._trees]
        supplement_hazard_value = np.zeros(len(hazard_functions))
        # for each observed time, make sure every hazard_function generated
        # by different tree will have a cumulative hazard value.
        for observed_time in np.sort(self._times):
            for index, hazard_function in enumerate(hazard_functions):
                if observed_time not in hazard_function:
                    hazard_function[observed_time] = \
                        supplement_hazard_value[index]
                else:
                    supplement_hazard_value[index] = \
                        hazard_function[observed_time]
        # calculate average to generate the ensemble hazard function
        ensemble_cumulative_hazard = {}
        for observed_time in self._times:
            ensemble_cumulative_hazard[observed_time] = \
                np.mean([hazard_function[observed_time]
                         for hazard_function in hazard_functions])
        return ensemble_cumulative_hazard

    def _estimate_median_time(self, hazard_function, avg_flag):
        log_two = np.log(2)
        prev_time = 0
        final_time = np.inf
        for i in range(len(self._times)):
            if hazard_function[self._times[i]] == log_two:
                return self._times[i]
            elif hazard_function[self._times[i]] < log_two:
                prev_time = self._times[i]
            else:
                if avg_flag:
                    return (prev_time + self._times[i]) / 2.
                else:
                    return self._times[i]
        return (prev_time + final_time) / 2.

    def _print_with_depth(self, string, depth):
        print("{0}{1}".format("    " * depth, string))

    def _print_tree(self, tree, depth=0):
        if tree["type"] == "leaf":
            self._print_with_depth(tree["t"], depth)
            return
        self._print_with_depth("{0} > {1}".format(tree["feature"],
                                                  tree["median"]), depth)
        self._print_tree(tree["left"], depth + 1)
        self._print_tree(tree["right"], depth + 1)

    def _grow_tree(self, data):
        new_tree = {}
        self._build(data, new_tree, 0)
        return new_tree

    def _sample_feature(self, x):
        features = \
            list(set(x.columns) - {self._time_column, self._event_column})
        sampled_features = \
            list(np.random.permutation(features))[:self._max_features] + \
            [self._time_column, self._event_column]
        return sampled_features

    def fit(self, train_df, duration_col='LOS', event_col='OUT',
            num_workers=36):
        self._duration_col = duration_col
        self._event_col = event_col
        reduced_train_df = self._train_pca(train_df)
        x_train = reduced_train_df.drop(columns=[duration_col, event_col])
        y_train = reduced_train_df[[duration_col, event_col]]
        assert isinstance(x_train, pd.DataFrame)
        assert isinstance(y_train, pd.DataFrame)
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
        assert isinstance(test_df, pd.DataFrame)
        assert isinstance(time, int) or \
               isinstance(time, float) or \
               isinstance(time, list)

        reduced_test_df = self._test_pca(test_df)
        if isinstance(time, int) or isinstance(time, float):
            time_list = [time]
        else:
            time_list = time

        proba_pred = []
        for _, row in reduced_test_df.iterrows():
            probas = [self._pred_survival(self._trees[i], row, time_list)
                      for i in range(self._n_trees)]
            proba_pred.append(list(np.average(np.array(probas), axis=0)))

        return np.array(proba_pred)

    def pred_median_time(self, test_df, average_to_get_median=True):
        assert isinstance(test_df, pd.DataFrame)
        result = []
        reduced_test_df = self._test_pca(test_df)
        for _, row in reduced_test_df.iterrows():
            cumulative_hazard = self._get_ensemble_cumulative_hazard(row)
            result.append(self._estimate_median_time(
                cumulative_hazard, average_to_get_median))
        return np.array(result)

    def pred_cumulative_hazard(self, x_test):
        result = [self._get_ensemble_cumulative_hazard(row)
                  for _, row in x_test.iterrows()]
        return np.array(result)

    def draw(self):
        for i in range(len(self._trees)):
            print("==========================================\nTree", i)
            self._print_tree(self._trees[i])
