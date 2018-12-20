from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from lifelines import KaplanMeierFitter
import numpy as np
import pandas as pd


class KNNKaplanMeier():

    def __init__(self, n_neighbors, pca_flag=False, n_components=20):
        """
        The K-Nearest Neighbor model is implemented using sikit-learn's
        KNeighborsClassifier along with lifelines's KaplanMeierFitter.
        The `pred_proba` and `pred_median_time` are based on KaplanMeierFitter's
        predict methods.

        :param n_neighbors:
        :param pca_flag: use PCA before fit the model.
        :param n_components: number of principle components for PCA
        """
        self.n_neighbors = n_neighbors
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

    def fit(self, train_df, duration_col='LOS', event_col='OUT'):
        """
        :param train_df: DataFrame, with the duration and the event column
        :param duration_col: the column name for duration
        :param event_col: the column name for event
        """
        self._duration_col = duration_col
        self._event_col = event_col
        # Dimension reduction first if applicable
        self.train_df = self._train_pca(train_df)
        train_x = self.train_df.drop(columns=[duration_col, event_col]).values
        # For kNN, we simply initiate the KNeighborsClassifier
        # the n_neighbors will be used later when calculating nearest neighbor
        self.neighbors = KNeighborsClassifier()
        self.neighbors.fit(train_x, np.zeros(len(train_x)))
        self.train_points = len(train_x)

    def pred_median_time(self, test_df):
        """
        for each test datapoint, find the k nearest neighbors, and use them to
        fit a Kaplan-Meier Model to get the predicted median time

        :param test_df: DataFrame
        :return: the list of median survival time for each test datapoint.
        """
        # Dimension reduction first if applicable
        reduced_test_df = self._test_pca(test_df)
        test_x = \
            reduced_test_df.drop(
                columns=[self._duration_col, self._event_col]
            ).values
        # calculate distance matrix to find the nearest neighbors
        distance_matrix, neighbor_matrix = \
            self.neighbors.kneighbors(
                X=test_x,
                n_neighbors=int(np.min([self.n_neighbors, self.train_points]))
            )

        test_time_median_pred = []
        for test_idx, test_point in enumerate(test_x):
            # find the k nearest neighbors
            neighbor_train_y = \
                self.train_df.iloc[neighbor_matrix[test_idx]][
                    [self._duration_col, self._event_col]
                ]
            kmf = KaplanMeierFitter()
            kmf.fit(neighbor_train_y[self._duration_col],
                    neighbor_train_y[self._event_col])
            # Kaplan Meier can directly get the predicted median survival time
            test_time_median_pred.append(kmf.median_)

        return np.array(test_time_median_pred)

    def pred_proba(self, test_df, time):
        """
        for each test datapoint, find the k nearest neighbors, and use them to
        fit a Kaplan-Meier Model to get the survival function

        :param test_df: DataFrame
        :param time: checkpoint time to calculate probability on
        :return: the probability matrix. each row is the survival function for
        one test datapoint.
        """
        if isinstance(time, int) or isinstance(time, float):
            time_list = [time]
        else:
            time_list = time

        # Dimension reduction first if applicable
        reduced_test_df = self._test_pca(test_df)
        test_x = \
            reduced_test_df.drop(
                columns=[self._duration_col, self._event_col]
            ).values
        # calculate distance matrix to find the nearest neighbors
        distance_matrix, neighbor_matrix = \
            self.neighbors.kneighbors(
                X=test_x,
                n_neighbors=int(np.min([self.n_neighbors, self.train_points]))
            )

        proba_matrix = []
        for test_idx, test_point in enumerate(test_x):
            # find the k nearest neighbors
            neighbor_train_y = \
                self.train_df.iloc[neighbor_matrix[test_idx]][
                    [self._duration_col, self._event_col]
                ]
            kmf = KaplanMeierFitter()
            kmf.fit(neighbor_train_y[self._duration_col],
                    neighbor_train_y[self._event_col])
            # Instead of using the `survival_function` of the Kaplan Meier Model
            # directly, we here use the `predict` method to calculate
            # probability for each checkpoint time (time_list). This is because
            # the survival_function will only give us the survival probability
            # of unique time points from the training dataset. In this case, the
            # training dataset (neighbor_train_y) can be very small
            proba_matrix.append(kmf.predict(time_list))

        return np.array(proba_matrix)

    def predict(self, test_df, time_list):
        """
        for each test datapoint, find the k nearest neighbors, and use them to
        fit a Kaplan-Meier Model to get the survival function, and then use
        the survival function the calculate the median survival time

        :param test_df: DataFrame
        :param time_list: checkpoint time to calculate probability on
        :return: the list of median survival time and the probability matrix
        """
        # Dimension reduction first if applicable
        reduced_test_df = self._test_pca(test_df)
        test_x = \
            reduced_test_df.drop(
                columns=[self._duration_col, self._event_col]
            ).values
        # calculate distance matrix to find the nearest neighbors
        distance_matrix, neighbor_matrix = \
            self.neighbors.kneighbors(
                X=test_x,
                n_neighbors=int(np.min([self.n_neighbors, self.train_points]))
            )

        proba_matrix = []
        test_time_median_pred = []
        for test_idx, test_point in enumerate(test_x):
            # find the k nearest neighbors
            neighbor_train_y = \
                self.train_df.iloc[neighbor_matrix[test_idx]][
                    [self._duration_col, self._event_col]
                ]
            kmf = KaplanMeierFitter()
            kmf.fit(neighbor_train_y[self._duration_col],
                    neighbor_train_y[self._event_col])
            survival_proba = kmf.predict(time_list)
            # calculate the median survival time.
            # the median survival time is the time at which the survival proba.
            # equals to 0.5. Here the survival_proba is descending sorted from
            # 1 to 0, so we only need to find the first probability that <= 0.5
            median_time = 0
            for col, proba in enumerate(survival_proba):
                if proba > 0.5:
                    continue

                if proba == 0.5:
                    median_time = time_list[col]
                else:
                    # here we take the average of the time before and after
                    # Pr = 0.5
                    median_time = (time_list[col - 1] + time_list[col]) / 2
                break

            test_time_median_pred.append(median_time)
            proba_matrix.append(survival_proba)

        return np.array(test_time_median_pred), np.array(proba_matrix)
