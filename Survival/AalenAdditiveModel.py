from lifelines import AalenAdditiveFitter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


class AalenAdditiveModel:

    def __init__(self, coef_penalizer, pca_flag=False, n_components=20):
        """
        The Aalen Additive Model directly use the implementation from lifeline.
        This implementationn simply wraps lifeline's implementation with unified
        interface in our project (fit, predict, pred_median_time, pred_proba), and
        also provides PCA layer.

        :param coef_penalizer: parameters from lifeline's implementation.
        :param pca_flag: use PCA before fit the model.
        :param n_components: number of principle components for PCA
        """
        self.aaf = AalenAdditiveFitter(coef_penalizer=coef_penalizer)
        self.pca_flag = pca_flag
        self.pca = PCA(n_components=n_components)
        # standardalizer is used if pca_flag is true. Use this to standardize
        # the input train and test so that PCA make sense
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

    def _find_nearest_time(self, time):
        """
        While the implementation of lifeline provide a function to predict the
        survival function, it seems like that it will return a survival proba.
        distribution for each datapoint **only on each unique time LOS from the
        train dataset**.
        This helper function simply maps [the time_list that we want to
        calculate on] to the train dataset's unique time list.

        :param time: The time list we want to calculate survival probability on
        :return:
        """
        nearest_time = -np.inf
        for tmp_time in self._time_index:
            if tmp_time < time:
                nearest_time = tmp_time
            elif tmp_time == time:
                return time
            else:
                if time - nearest_time < tmp_time - time:
                    pass
                else:
                    nearest_time = tmp_time
                break
        return nearest_time

    def fit(self, train_df, duration_col='LOS', event_col='OUT'):
        """
        :param train_df: DataFrame, with the duration and the event column
        :param duration_col: the column name for duration
        :param event_col: the column name for event
        """
        self._duration_col = duration_col
        self._event_col = event_col
        # the PCA steps is done here ⬇️
        self.aaf.fit(self._train_pca(train_df), duration_col=duration_col,
                     event_col=event_col)
        # remember the train dataset's unique time list
        self._time_index = list(self.aaf.durations)

    def pred_median_time(self, test_df):
        """
        :param test_df: DataFrame
        :return: the list of median survival time for each test datapoint.
        """
        test_time_median_pred = \
            np.array(self.aaf.predict_median(self._test_pca(test_df))).flatten()
        return test_time_median_pred

    def pred_proba(self, test_df, time):
        """
        :param test_df: DataFrame
        :param time: checkpoint time to calculate probability on
        :return: the probability matrix. each row is the survival function
        (probability distribution) for one test datapoint.
        """
        reduced_test_df = self._test_pca(test_df)
        tmp_probas = self.aaf.predict_survival_function(reduced_test_df)

        if isinstance(time, int) or isinstance(time, float):
            time_indice = [self._find_nearest_time(time)]
        else:
            time_indice = [self._find_nearest_time(cur_time)
                           for cur_time in time]

        proba_matrix = np.zeros((test_df.shape[0], len(time_indice)))
        for row, test_index in enumerate(test_df.index):
            for col, time_index in enumerate(time_indice):
                proba_matrix[row][col] = tmp_probas[test_index][time_index]
        return proba_matrix

    def predict(self, test_df, time_list):
        """
        :param test_df: DataFrame
        :param time_list: checkpoint time to calculate probability on
        :return: the list of median survival time and the probability matrix
        """
        return self.pred_median_time(test_df), \
               self.pred_proba(test_df, time_list)
