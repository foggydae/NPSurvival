from lifelines import AalenAdditiveFitter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

class AalenAdditiveModel:
    def __init__(self, coef_penalizer, pca_flag=False, n_components=20):
        self.aaf = AalenAdditiveFitter(coef_penalizer=coef_penalizer)
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

    def _find_nearest_time(self, time):
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
        self._duration_col = duration_col
        self._event_col = event_col
        self.aaf.fit(self._train_pca(train_df), duration_col=duration_col, event_col=event_col)
        self._time_index = list(self.aaf.durations)

    def pred_median_time(self, test_df):
        assert isinstance(test_df, pd.DataFrame)
        test_time_median_pred = \
            list(np.array(self.aaf.predict_median(self._test_pca(test_df))).flatten())
        return test_time_median_pred

    def pred_proba(self, test_df, time):
        assert isinstance(test_df, pd.DataFrame)
        assert isinstance(time, int) or \
               isinstance(time, float) or \
               isinstance(time, list)

        reduced_test_df = self._test_pca(test_df)
        tmp_probas = self.aaf.predict_survival_function(reduced_test_df)

        if isinstance(time, int) or isinstance(time, float):
            time_indice = [self._find_nearest_time(time)]
        else:
            time_indice = [self._find_nearest_time(cur_time)
                          for cur_time in time]

        num_test = test_df.shape[0]
        proba_matrix = np.zeros((test_df.shape[0], len(time_indice)))
        for row, test_index in enumerate(test_df.index):
            for col, time_index in enumerate(time_indice):
                proba_matrix[row][col] = tmp_probas[test_index][time_index]
        return proba_matrix
