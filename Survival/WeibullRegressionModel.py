from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import rpy2.robjects as robjects
from rpy2.robjects import FloatVector, IntVector
from rpy2.robjects.packages import importr

import pandas as pd
import numpy as np
import re


class WeibullRegressionModel:
    def __init__(self, pca_flag=False, n_components=20):
        self.survival = importr("survival")
        self.base = importr('base')
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


    def fit(self, train_df, duration_col='LOS', event_col='OUT'):
        self._duration_col = duration_col
        self._event_col = event_col

        reduced_train_df = self._train_pca(train_df)
        self.column_ref = {}

        df_command = 'train.df <- data.frame('
        sr_command = 'survregWeibull <- survreg(s ~ '
        feas = reduced_train_df.drop(columns=[duration_col, event_col]).columns
        for index, column in enumerate(feas):
            r_var_name = "C" + str(index)
            self.column_ref[column] = r_var_name
            robjects.globalenv[r_var_name] = \
                FloatVector(list(reduced_train_df[column]))
            df_command += r_var_name + ', '
            sr_command += r_var_name
            if index != len(reduced_train_df.columns) - 3:
                sr_command += ' + '
        robjects.globalenv["LOS"] = \
            FloatVector(list(reduced_train_df[duration_col]))
        robjects.globalenv["OUT"] = \
            FloatVector(list(reduced_train_df[event_col]))
        df_command += 'LOS, OUT)'
        sr_command += ', train.df, dist = "weibull")'

        robjects.r(df_command)
        robjects.r('s <- Surv(train.df$LOS, train.df$OUT)')
        robjects.r(sr_command)


    def pred_median_time(self, test_df):
        assert isinstance(test_df, pd.DataFrame)
        reduced_test_df = self._test_pca(test_df)

        robjects.r('pct <- seq(.001,.999,by=.001)')
        pred_median = []
        true_median = reduced_test_df["LOS"]
        for _, row in reduced_test_df.iterrows():
            pd_command = 'result <- predict(survregWeibull, newdata=list('
            for index, column in enumerate(self.column_ref):
                r_var_name = self.column_ref[column]
                pd_command += r_var_name + '=' + str(row[column])
                if index != len(self.column_ref) - 1:
                    pd_command += ','
                else:
                    pd_command += '), type="quantile", p=pct)'
            result = list(robjects.r(pd_command))
            pred_median.append(result[499])
        return pred_median


    def pred_proba(self, test_df, time):
        assert isinstance(test_df, pd.DataFrame)
        assert isinstance(time, int) or isinstance(time, float) or \
               isinstance(time, list)
        reduced_test_df = self._test_pca(test_df)

        if isinstance(time, int) or isinstance(time, float):
            timestamps = [time]
        else:
            timestamps = time

        robjects.r('pct <- seq(.001,.999,by=.001)')
        proba_matrix = []
        for _, row in reduced_test_df.iterrows():
            pd_command = 'result <- predict(survregWeibull, newdata=list('
            for index, column in enumerate(self.column_ref):
                r_var_name = self.column_ref[column]
                pd_command += r_var_name + '=' + str(row[column])
                if index != len(self.column_ref) - 1:
                    pd_command += ','
                else:
                    pd_command += '), type="quantile", p=pct)'
            result = np.array(list(robjects.r(pd_command)))

            proba_list = []
            for timestamp in timestamps:
                if result[0] > timestamp:
                    proba_list.append(1)
                elif result[-1] < timestamp:
                    proba_list.append(0)
                else:
                    idx = (np.abs(result - timestamp)).argmin()
                    if result[idx] < timestamp:
                        # nearest proba: 1 - (idx+1)*0.001
                        # second nearest proba: 1 - (idx+2)*0.001
                        proba_list.append(1 - (idx + idx + 3) * 0.001 / 2)
                    elif result[idx] > timestamp:
                        # nearest proba: 1 - (idx+1)*0.001
                        # second nearest proba: 1 - (idx)*0.001
                        proba_list.append(1 - (idx + idx + 1) * 0.001 / 2)
                    else:
                        proba_list.append(1 - (idx + 1) * 0.001)
            proba_matrix.append(proba_list)

        return np.array(proba_matrix)

