from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from lifelines import KaplanMeierFitter
import numpy as np
import pandas as pd


class KNNKaplanMeier():
    def __init__(self, n_neighbors, pca_flag=False, n_components=20):
        self.n_neighbors = n_neighbors
        self.pca_flag = pca_flag
        self.pca = PCA(n_components=n_components)
        self.standardalizer = StandardScaler()

    def _train_pca(self, train_df):
        if self.pca_flag:
            train_x = train_df.drop(columns=[self._duration_col, self._event_col]).values

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
        else:
            return train_df

    def _test_pca(self, test_df):
        if self.pca_flag:
            test_x = test_df.drop(columns=[self._duration_col, self._event_col]).values
            test_x = self.standardalizer.transform(test_x)
            reduced_x = self.pca.transform(test_x)
            reduced_test_df = pd.DataFrame(reduced_x).set_index(test_df.index)
            columns = ["C" + str(i) for i in range(reduced_test_df.shape[1])]
            reduced_test_df.columns = columns
            reduced_test_df["LOS"] = test_df["LOS"]
            reduced_test_df["OUT"] = test_df["OUT"]
            return reduced_test_df
        else:
            return test_df

    def fit(self, train_df, duration_col="LOS", event_col="OUT"):
        self._duration_col = duration_col
        self._event_col = event_col
        self.train_df = self._train_pca(train_df)
        train_x = self.train_df.drop(columns=[duration_col, event_col]).values
        self.neighbors = KNeighborsClassifier()
        self.neighbors.fit(train_x, np.zeros(len(train_x)))
        self.train_points = len(train_x)

    def pred_median_time(self, test_df):
        assert isinstance(test_df, pd.DataFrame)
        reduced_test_df = self._test_pca(test_df)
        test_x = \
            reduced_test_df.drop(columns=[self._duration_col, self._event_col]).values
        distance_matrix, neighbor_matrix = \
            self.neighbors.kneighbors(X=test_x, n_neighbors=int(np.min([self.n_neighbors, self.train_points])))

        test_time_median_pred = []
        for test_idx, test_point in enumerate(test_x):
            neighbor_train_y = \
                self.train_df.iloc[neighbor_matrix[test_idx]][
                    [self._duration_col, self._event_col]
                ]
            kmf = KaplanMeierFitter()
            kmf.fit(neighbor_train_y[self._duration_col],
                    neighbor_train_y[self._event_col])
            test_time_median_pred.append(kmf.median_)

        return np.array(test_time_median_pred)

    def pred_proba(self, test_df, time):
        assert isinstance(test_df, pd.DataFrame)
        assert isinstance(time, int) \
               or isinstance(time, float) \
               or isinstance(time, list)
        reduced_test_df = self._test_pca(test_df)

        if isinstance(time, int) or isinstance(time, float):
            time_list = [time]
        else:
            time_list = time

        test_x = \
            reduced_test_df.drop(columns=[self._duration_col, self._event_col]).values
        distance_matrix, neighbor_matrix = \
            self.neighbors.kneighbors(X=test_x, n_neighbors=self.n_neighbors)

        proba_matrix = []
        for test_idx, test_point in enumerate(test_x):
            neighbor_train_y = \
                self.train_df.iloc[neighbor_matrix[test_idx]][
                    [self._duration_col, self._event_col]
                ]
            kmf = KaplanMeierFitter()
            kmf.fit(neighbor_train_y[self._duration_col],
                    neighbor_train_y[self._event_col])
            proba_matrix.append(kmf.predict(time_list))

        return np.array(proba_matrix)
