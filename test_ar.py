# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 09:51:02 2021

@author: boettner
"""

import numpy as np
import matplotlib.pyplot as plt
from _help_functions import shorten_ts
from _classical_EWS import rolling_variance, rolling_autocorr
from _regression    import ar1n_fit

ts = np.load("lm_23_3.npz")
#ts = np.load("dw_0_4_trad.npz")

ts_cuts = shorten_ts(ts); noise = np.sqrt(ts["v"][0]); ts = ts["x"]
num = ts.shape[1]
l = np.amin(np.diff(ts_cuts,axis=0)) # length of shortest ts  
ts_c = np.array([ts[ts_cuts[1,i]-l:ts_cuts[1,i],i] for i in range(num)]).T
window_size = l//8

ts_v                = rolling_variance(ts_c, window_size)
ts_a                = rolling_autocorr(ts_c, window_size)
ts_lb, ts_l, ts_rho = ar1n_fit(ts_c, window_size)

v    = np.nanmean(ts_v, axis=1)
a    = np.nanmean(ts_a, axis=1)
lb   = np.nanmean(ts_lb,axis=1)
l    = np.nanmean(ts_l, axis=1); rho = np.nanmean(ts_rho, axis=1)


# from _regression import ho_fit
# from _significance import trend_test
# order = 2
# lin_bic, lin_llf = ho_fit(ts_c, window_size, 1)     # calc for linear model
# ho_bic, ho_llf   = ho_fit(ts_c, window_size, order) # calc for ho model
# delta_bic        = ho_bic - lin_bic                 # bic difference
# ts_bicp = trend_test(delta_bic[window_size:,:]) # check for trend