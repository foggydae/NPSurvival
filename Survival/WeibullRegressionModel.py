from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import rpy2.robjects as robjects
from rpy2.robjects import FloatVector
from rpy2.robjects.packages import importr

import pandas as pd
import numpy as np


class WeibullRegressionModel:

    def __init__(self, pca_flag=False, n_components=20):
        """
        This is a python wrapper of R's Weibull Regression Model (in Survival
        package). rpy2 is used to call R's functions.

        :param pca_flag: use PCA before fit the model.
        :param n_components: number of principle components for PCA
        """
        # prepare R's packages
        self.survival = importr("survival")
        self.base = importr('base')

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
        As we only need to run a couple of lines in R, here we directly generate
        R command as python string, and call R using rpy2. There should be a
        more elegant way of using rpy2.

        :param train_df: DataFrame, with the duration and the event column
        :param duration_col: the column name for duration
        :param event_col: the column name for event
        """
        self._duration_col = duration_col
        self._event_col = event_col

        reduced_train_df = self._train_pca(train_df)
        self.column_ref = {}

        # Generate R command.
        # Basically we first generate a vector for each feature we have:
        # C.i <- c(1,0,2,1,...)
        # Then we combine those vectors into a dataframe in R
        # train.df <- data.frame(C.1, C.2, ..., OUT, LOS)
        # Then we fit the model
        # s <- Surv(train.df$LOS, train.df$OUT)
        # survregWeibull <- survreg(s ~ C1 + C2 + ..., train.df, dist="weibull")
        df_command = 'train.df <- data.frame('
        sr_command = 'survregWeibull <- survreg(s ~ '
        columns = \
            reduced_train_df.drop(columns=[duration_col, event_col]).columns
        for index, column in enumerate(columns):
            # we don't actually care about the column name in R
            r_var_name = "C." + str(index)
            self.column_ref[column] = r_var_name
            # generate a R vector for each feature
            robjects.globalenv[r_var_name] = \
                FloatVector(list(reduced_train_df[column]))
            df_command += r_var_name + ', '
            sr_command += r_var_name
            if index != len(columns) - 1:
                sr_command += ' + '
        # generate a R vector for LOS
        robjects.globalenv[duration_col] = \
            FloatVector(list(reduced_train_df[duration_col]))
        # generate a R vector for OUT
        robjects.globalenv[event_col] = \
            FloatVector(list(reduced_train_df[event_col]))
        df_command += duration_col + ', ' + event_col + ')'
        sr_command += ', train.df, dist = "weibull")'

        robjects.r(df_command)
        robjects.r('s <- Surv(train.df$' + duration_col + ', train.df$' +
                   event_col + ')')
        robjects.r(sr_command)

    def pred_median_time(self, test_df):
        """
        In R's implementation, the weibull regression model can be used to
        estimate a time given a survival probability. Thus we only need to
        find the time that has a survival probability equals to 0.5.

        :param test_df: DataFrame
        :return: the list of median survival time for each test datapoint.
        """
        reduced_test_df = self._test_pca(test_df)

        # generate a sequence of survival probabilities, and predict the
        # corresponding survival time for each probability
        robjects.r('pct <- seq(.001,.999,by=.001)')
        pred_median = []
        # perform the predictionn for each test timepoint
        for _, row in reduced_test_df.iterrows():
            # generate R command:
            # result <- predict(survregWeibull, newdata=list(C.1=0,C.2=1,...),
            # type="quantile", p=pct)
            pd_command = 'result <- predict(survregWeibull, newdata=list('
            for index, column in enumerate(self.column_ref):
                r_var_name = self.column_ref[column]
                pd_command += r_var_name + '=' + str(row[column])
                if index != len(self.column_ref) - 1:
                    pd_command += ','
                else:
                    pd_command += '), type="quantile", p=pct)'
            # result is a list of time that has the survival probability from
            # 0.999 to 0.001. Notice that time in result list is in ascending
            # order
            result = list(robjects.r(pd_command))
            # the 500th time is the time that has a survival probability of 0.5
            pred_median.append(result[499])

        return np.array(pred_median)

    def pred_proba(self, test_df, time):
        """
        In R's implementation, the Weibull regression model can be used to
        estimate a time given a survival probability. We can utilize this result
        to calculate the estimate survival probability of a given time point. We
        can achieve this by mapping the given time point into some similar time
        in the estimate time list given by R's Weibull model.

        :param test_df: DataFrame
        :param time: checkpoint time to calculate probability on
        :return: the probability matrix. each row is the survival function for
        one test datapoint.
        """
        reduced_test_df = self._test_pca(test_df)

        if isinstance(time, int) or isinstance(time, float):
            timestamps = [time]
        else:
            timestamps = time

        # generate a sequence of survival probabilities, and predict the
        # corresponding survival time for each probability
        robjects.r('pct <- seq(.001,.999,by=.001)')
        proba_matrix = []
        # perform the predictionn for each test timepoint
        for _, row in reduced_test_df.iterrows():
            # generate R command:
            # result <- predict(survregWeibull, newdata=list(C.1=0,C.2=1,...),
            # type="quantile", p=pct)
            pd_command = 'result <- predict(survregWeibull, newdata=list('
            for index, column in enumerate(self.column_ref):
                r_var_name = self.column_ref[column]
                pd_command += r_var_name + '=' + str(row[column])
                if index != len(self.column_ref) - 1:
                    pd_command += ','
                else:
                    pd_command += '), type="quantile", p=pct)'
            # result is a list of time that has the survival probability from
            # 0.999 to 0.001. Notice that time in result list is in ascending
            # order
            result = np.array(list(robjects.r(pd_command)))

            # what we want is a list of probability of a given list of survival
            # time. Thus, here we map each given time to the 2 nearest time in
            # the result list, and then use the corresponding probabilities of
            # those 2 time to generate the estimated probability of the given
            # time
            proba_list = []
            # for each given time
            for timestamp in timestamps:
                # if the given time is smaller than the time that has a survival
                # probability of 0.999
                if result[0] > timestamp:
                    proba_list.append(1)
                # else if the given time is larger than the time that has a
                # survival probability of 0.001
                elif result[-1] < timestamp:
                    proba_list.append(0)
                else:
                    # find the nearest time's index in the result list
                    idx = (np.abs(result - timestamp)).argmin()
                    if result[idx] < timestamp:
                        # nearest proba: 1 - (idx+1)*0.001
                        # second nearest proba: 1 - (idx+2)*0.001
                        # est. proba. = avg(nearest, second nearest)
                        proba_list.append(1 - (idx + idx + 3) * 0.001 / 2)
                    elif result[idx] > timestamp:
                        # nearest proba: 1 - (idx+1)*0.001
                        # second nearest proba: 1 - (idx)*0.001
                        # est. proba. = avg(nearest, second nearest)
                        proba_list.append(1 - (idx + idx + 1) * 0.001 / 2)
                    else:
                        proba_list.append(1 - (idx + 1) * 0.001)
            proba_matrix.append(proba_list)

        return np.array(proba_matrix)

    def predict(self, test_df, time_list):
        """
        :param test_df: DataFrame
        :param time_list: checkpoint time to calculate probability on
        :return: the list of median survival time and the probability matrix
        """
        reduced_test_df = self._test_pca(test_df)

        if isinstance(time_list, int) or isinstance(time_list, float):
            timestamps = [time_list]
        else:
            timestamps = time_list

        # generate a sequence of survival probabilities, and predict the
        # corresponding survival time for each probability
        robjects.r('pct <- seq(.001,.999,by=.001)')
        proba_matrix = []
        pred_median = []
        # perform the predictionn for each test timepoint
        for _, row in reduced_test_df.iterrows():
            # generate R command:
            # result <- predict(survregWeibull, newdata=list(C.1=0,C.2=1,...),
            # type="quantile", p=pct)
            pd_command = 'result <- predict(survregWeibull, newdata=list('
            for index, column in enumerate(self.column_ref):
                r_var_name = self.column_ref[column]
                pd_command += r_var_name + '=' + str(row[column])
                if index != len(self.column_ref) - 1:
                    pd_command += ','
                else:
                    pd_command += '), type="quantile", p=pct)'
            # result is a list of time that has the survival probability from
            # 0.999 to 0.001. Notice that time in result list is in ascending
            # order
            result = np.array(list(robjects.r(pd_command)))
            # the 500th time is the time that has a survival probability of 0.5
            pred_median.append(result[499])

            # what we want is a list of probability of a given list of survival
            # time. Thus, here we map each given time to the 2 nearest time in
            # the result list, and then use the corresponding probabilities of
            # those 2 time to generate the estimated probability of the given
            # time
            proba_list = []
            # for each given time
            for timestamp in timestamps:
                # if the given time is smaller than the time that has a survival
                # probability of 0.999
                if result[0] > timestamp:
                    proba_list.append(1)
                # else if the given time is larger than the time that has a
                # survival probability of 0.001
                elif result[-1] < timestamp:
                    proba_list.append(0)
                else:
                    # find the nearest time's index in the result list
                    idx = (np.abs(result - timestamp)).argmin()
                    if result[idx] < timestamp:
                        # nearest proba: 1 - (idx+1)*0.001
                        # second nearest proba: 1 - (idx+2)*0.001
                        # est. proba. = avg(nearest, second nearest)
                        proba_list.append(1 - (idx + idx + 3) * 0.001 / 2)
                    elif result[idx] > timestamp:
                        # nearest proba: 1 - (idx+1)*0.001
                        # second nearest proba: 1 - (idx)*0.001
                        # est. proba. = avg(nearest, second nearest)
                        proba_list.append(1 - (idx + idx + 1) * 0.001 / 2)
                    else:
                        proba_list.append(1 - (idx + 1) * 0.001)
            proba_matrix.append(proba_list)

        return np.array(pred_median), np.array(proba_matrix)