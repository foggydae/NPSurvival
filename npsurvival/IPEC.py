# from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as Pool
from lifelines import KaplanMeierFitter
import numpy as np
import pandas as pd


class IPEC:
    def __init__(self, train_df, pred_proba_func, g_type, t_thd=1, t_step=0.01,
                 time_col="LOS", death_identifier="OUT", divide_by="T", 
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
        self._train_df = train_df
        self._time_col = time_col
        self._death_identifier = death_identifier
        self._S = pred_proba_func
        self._divide_by = divide_by
        self._verbose = verbose
        if g_type == "All_One":
            self._G = lambda time: 1
        elif g_type == "Kaplan_Meier":
            self._G = self._kaplan_meier_g()
        else:
            print("[ERROR] unrecognized G type.")
        sorted_obs_times = sorted(list(train_df[time_col].unique()))
        self._upperT = sorted_obs_times[int(len(sorted_obs_times) * t_thd) - 1]
        if self._verbose:
            print("T:",  self._upperT)
        if t_step == "obs":
            self._check_points = \
                sorted_obs_times[:int(len(sorted_obs_times) * t_thd)]
            if self._check_points[0] != 0:
                self._check_points = [0] + self._check_points
        else:
            assert type(t_step) == float or type(t_step) == int
            self._check_points = \
                list(np.arange(0, self._upperT, t_step)) + [self._upperT]
        if self._verbose:
            print("number of check points:", len(self._check_points))

    def _kaplan_meier_g(self):
        kmf = KaplanMeierFitter()
        kmf.fit(self._train_df[self._time_col],
                self._train_df[self._death_identifier].apply(lambda x: 1 - x))
        return kmf.predict

    def _ipec_for_obe_obs(self, obs_df):
        t_obs = obs_df[self._time_col].iloc[0]
        d_obs = obs_df[self._death_identifier].iloc[0]
        g_obs = self._G(t_obs)
        t_times_ipec = 0
        for i in range(1, len(self._check_points)):
            t_cur = self._check_points[i]
            t_prev = self._check_points[i - 1]
            g_cur = self._G(t_cur)
            s_cur = self._S(obs_df, t_cur)[0]
            obs_gt_cur = int(t_obs > t_cur)
            obs_lte_cur = 1 - obs_gt_cur
            cur_tmp_ipec = \
                (t_cur - t_prev) * \
                ((d_obs * obs_lte_cur / g_obs) + (obs_gt_cur / g_cur)) * \
                ((obs_gt_cur - s_cur) ** 2)
            t_times_ipec += cur_tmp_ipec
            if self._verbose:
                print("observed time:", t_obs, 
                    ", check point:", t_cur, 
                    ", pred survival:", s_cur,
                    ", cur ipec:", cur_tmp_ipec)
        if self._verbose:
            print()
        if self._divide_by == "T":
            return t_times_ipec / self._upperT
        else:
            return t_times_ipec

    def avg_ipec(self, test_df, num_workers=36, print_result=True, 
        use_multiprocess=True):
        obs_list = [test_df.iloc[[i]] for i in range(test_df.shape[0])]
        if use_multiprocess:
            with Pool(num_workers) as tmp_pool:
                ipec_list = tmp_pool.map(self._ipec_for_obe_obs, obs_list)
        else:
            ipec_list = [self._ipec_for_obe_obs(obs_df) for obs_df in obs_list]

        if print_result:
            print("IPEC:", np.average(ipec_list))
        return np.average(ipec_list)
