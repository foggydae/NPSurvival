import glmnet_python
from glmnet import glmnet
from glmnetCoef import glmnetCoef
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from collections import Counter

class CoxPHModel:
    def __init__(self, alpha, lambda_, pca_flag=False, n_components=20):
        self._alpha = alpha
        self._lambda = lambda_
        self.pca_flag = pca_flag
        self.pca = PCA(n_components=n_components)
        self.standardalizer = StandardScaler()
        self.mean_remover = StandardScaler()

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
        test_x = test_df.drop(columns=[self._duration_col, self._event_col]).values
        test_x = self.standardalizer.transform(test_x)
        reduced_x = self.pca.transform(test_x)
        reduced_test_df = pd.DataFrame(reduced_x).set_index(test_df.index)
        columns = ["C" + str(i) for i in range(reduced_test_df.shape[1])]
        reduced_test_df.columns = columns
        reduced_test_df["LOS"] = test_df["LOS"]
        reduced_test_df["OUT"] = test_df["OUT"]
        return reduced_test_df

    def _predict_median_survival_times(self, x_test,
                                       average_to_get_median=True):
        num_test_subjects = len(x_test)
        median_survival_times = np.zeros(num_test_subjects)

        log_minus_log_half = np.log(-np.log(0.5))

        for subj_idx in range(num_test_subjects):
            log_hazard = \
                self.log_baseline_hazard + np.inner(self.beta, x_test[subj_idx])
            log_cumulative_hazard = np.zeros(self.num_unique_times)
            for time_idx in range(self.num_unique_times):
                log_cumulative_hazard[time_idx] \
                    = logsumexp(log_hazard[:time_idx + 1])

            t_inf = np.inf
            t_sup = 0.
            for time_idx, t in enumerate(self.sorted_unique_times):
                cur_chazard = log_cumulative_hazard[time_idx]
                if log_minus_log_half <= cur_chazard and t < t_inf:
                    t_inf = t
                if log_minus_log_half >= cur_chazard and t > t_sup:
                    t_sup = t

            if average_to_get_median:
                median_survival_times[subj_idx] = 0.5 * (t_inf + t_sup)
            else:
                median_survival_times[subj_idx] = t_inf

        return median_survival_times

    def _predict_probas(self, x_test):
        pred_matrix = []
        for subject_x in x_test:
            log_hazard = \
                self.log_baseline_hazard + np.inner(self.beta, subject_x)
            survival_proba = np.zeros(self.num_unique_times)
            for time_idx in range(self.num_unique_times):
                log_cumulative_hazard = logsumexp(log_hazard[:time_idx + 1])
                survival_proba[time_idx] = \
                    np.exp(-np.exp(log_cumulative_hazard))
            pred_matrix.append(survival_proba)

        return pred_matrix

    def _find_nearest_time_index(self, time):
        nearest_time_index = -1
        nearest_time = -np.inf
        for index, tmp_time in enumerate(self._sort_unique_times):
            if tmp_time == time:
                return index
            elif tmp_time < time:
                nearest_time = tmp_time
                nearest_time_index = index
            else:
                if time - nearest_time > tmp_time - time:
                    nearest_time = tmp_time
                    nearest_time_index = index
                break
        return nearest_time_index

    def fit(self, train_df, duration_col='LOS', event_col='OUT'):
        self._duration_col = duration_col
        self._event_col = event_col

        train_y = train_df[[duration_col, event_col]].values
        train_x = train_df.drop(columns=[duration_col, event_col]).values
        self._sort_unique_times = sorted(list(train_df[duration_col].unique()))
        self.mean_remover.fit(train_x)

        fit = glmnet(x=train_x.copy(), y=train_y.copy(),
                     family='cox', alpha=self._alpha, standardize=True,
                     intr=False)
        self.beta = glmnetCoef(fit, s=np.array([self._lambda])).flatten()

        # note: from inspecting Lifelines' code, Lifelines does not average to
        # get the median (e.g., find 2 nearest median points and take their
        # average); to get numerically nearly identical results, set parameter
        # `average_to_get_median` to False, plug in the beta learned by the
        # Lifelines' Cox proportional hazards, and be sure to subtract off the
        # training feature means from both `x_test` and `x_train`
        observed_times = train_y[:, 0]
        event_indicators = train_y[:, 1]
        # For each observed time, how many times the event occurred
        event_counts = Counter()
        for t, r in zip(observed_times, event_indicators):
            event_counts[t] += int(r)
        # Sorted list of observed times
        self.sorted_unique_times = np.sort(list(event_counts.keys()))
        self.num_unique_times = len(self.sorted_unique_times)
        self.log_baseline_hazard = np.zeros(self.num_unique_times)

        for time_idx, t in enumerate(self.sorted_unique_times):
            logsumexp_args = []
            for subj_idx, observed_time in enumerate(observed_times):
                if observed_time >= t:
                    logsumexp_args.append(
                        np.inner(self.beta, train_x[subj_idx]))
            if event_counts[t] > 0:
                self.log_baseline_hazard[time_idx] \
                    = np.log(event_counts[t]) - logsumexp(logsumexp_args)
            else:
                self.log_baseline_hazard[time_idx] \
                    = -np.inf - logsumexp(logsumexp_args)

    def pred_median_time(self, test_df, remove_mean=True):
        assert isinstance(test_df, pd.DataFrame)
        test_x = \
            test_df.drop(columns=[self._duration_col, self._event_col]).values
        if remove_mean:
            test_x = self.mean_remover.transform(test_x)
        return self._predict_median_survival_times(test_x)

    def pred_proba(self, test_df, time):
        assert isinstance(test_df, pd.DataFrame)
        assert isinstance(time, int) or isinstance(time, float) or \
               isinstance(time, list)
        if isinstance(time, int) or isinstance(time, float):
            time_indice = [self._find_nearest_time_index(time)]
        else:
            time_indice = [self._find_nearest_time_index(cur_time)
                          for cur_time in time]
        test_x = \
            test_df.drop(columns=[self._duration_col, self._event_col]).values
        tmp_probas = self._predict_probas(test_x)
        proba_matrix = np.zeros((len(test_x), len(time_indice)))
        for row in range(len(test_x)):
            for col, time_index in enumerate(time_indice):
                proba_matrix[row][col] = tmp_probas[row][time_index]
        return proba_matrix
