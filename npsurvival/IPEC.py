# from multiprocessing import Pool
from lifelines import KaplanMeierFitter
import numpy as np
import pandas as pd


class IPEC:

    def __init__(self, train_df, g_type="All_One", t_thd=0.8, t_step="obs",
                 time_col="LOS", death_identifier="OUT", divide_by="N", 
                 verbose=False):
        """
        Constructor
        :param train_df: pandas dataframe, train data used to train the model
        (must be the same dataset)
        :param pred_proba_func: function with args ([pd.Dataframe] test_df,
        [scalar|list] time), the pred_proba function of the trained model, given
        a list of observations test_df, and a list of time,return a list of
        probability of survival at that time
        :param g_type: "All_One" or "Kaplan_Meier" are supported
        :param t_thd: T = the t_thd quantile of the unique observed time list
        in training dataset
        :param t_step: "obs" or scalar(int|float), if "obs", use the unique
        observed time up to T in training dataset
        to calculate the integral; if scalar, use arange(0, T, t_step) as
        support time points to calculate integral
        :param time_col: column name in train_df that refers to time column
        :param death_identifier: column name in train_df that refers to the
        death indicator column
        :param divide_by: "T": divided by upper T; other, do not divide
        """
        obs_times = train_df[time_col]
        obs_events = train_df[death_identifier]

        self._divide_by = divide_by
        self._verbose = verbose
        if g_type == "All_One":
            self._G = lambda time: 1
        elif g_type == "Kaplan_Meier":
            self._G = self._kaplan_meier_g()
        else:
            print("[ERROR] unrecognized G type.")
        self._G_values = [self._G(obs_t) for obs_t in obs_times]

        sorted_obs_times = sorted(list(obs_times.unique()))
        self._upperT = sorted_obs_times[int(len(sorted_obs_times) * t_thd) - 1]
        self.verbose_print("T: " + str(self._upperT))

        if t_step == "obs":
            self._check_points = \
                sorted_obs_times[:int(len(sorted_obs_times) * t_thd)]
            if self._check_points[0] != 0:
                self._check_points = [0] + self._check_points
        else:
            assert type(t_step) == float or type(t_step) == int
            self._check_points = \
                list(np.arange(0, self._upperT, t_step)) + [self._upperT]
        self.verbose_print("number of check points: " + str(len(self._check_points) - 1))

    def _kaplan_meier_g(self, obs_times, obs_events):
        kmf = KaplanMeierFitter()
        kmf.fit(obs_times, obs_events.apply(lambda x: 1 - x))
        return kmf.predict

    def get_check_points(self):
        return self._check_points[1:]

    def calc_ipec(self, prob_matrix, obs_test_times, obs_test_events):
        assert prob_matrix.shape[0] == len(obs_test_times) and \
               len(obs_test_times) == len(obs_test_events)
        assert prob_matrix.shape[1] == len(self._check_points) - 1

        for obs_id in range(len(obs_test_times)):
            t_obs = obs_test_times[obs_id]
            d_obs = obs_test_events[obs_id]
            g_obs = self._G(t_obs)
            t_times_ipec = 0

            for i in range(1, len(self._check_points)):
                t_cur = self._check_points[i]
                t_prev = self._check_points[i - 1]
                g_cur = self._G_values[i - 1]
                s_cur = prob_matrix[obs_id][i - 1]
                obs_gt_cur = int(t_obs > t_cur)
                obs_lte_cur = 1 - obs_gt_cur

                cur_tmp_ipec = \
                    (t_cur - t_prev) * \
                    ((d_obs * obs_lte_cur / g_obs) + (obs_gt_cur / g_cur)) * \
                    ((obs_gt_cur - s_cur) ** 2)
                t_times_ipec += cur_tmp_ipec
                
                self.verbose_print("observed time: " + str(t_obs) + \
                    ", check point: " +  str(t_cur) + \
                    ", pred survival: " + str(s_cur) + \
                    ", cur ipec: " + str(cur_tmp_ipec))
            self.verbose_print("")

            if self._divide_by == "T":
                return t_times_ipec / self._upperT
            else:
                return t_times_ipec

    def verbose_print(self, content):
        if self._verbose:
            print(content)


