# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 15:43:02 2021

@author: boettner
"""

import numpy as np
from _classical_EWS import rolling_variance, rolling_autocorr
from _regression    import ar1n_fit
from _significance  import trend_test
from _help_functions import shorten_ts

source = "ar_noise"

if source == "ar_noise":
    lm = np.load("roc/" + source + "/lm_3_5_n.npz")
    #lm = np.load("_models/lm_3_5_n.npz") # taken from changing variance with sn = 0.04
    #dw = np.load("roc/" + source + "/dw_0_5_big.npz")
    #sp = np.load("roc/" + source + "/sp_0_5_big.npz")
elif source == "simple":
    lm = np.load("roc/" + source + "/lm_13_3.npz") # taken from changing variance with sn = 0.04
    dw = np.load("roc/" + source + "/dw_13_3.npz")
    sp = np.load("roc/" + source + "/sp_13_3.npz")    

thresholds  = np.arange(0,1.01, 0.01)

def calc_positives(ts, thresholds):
    ts_cuts = shorten_ts(ts); ts = ts["x"]
    
    num = ts.shape[1]

    l = np.amin(np.diff(ts_cuts,axis=0)) # length of shortest ts  
    ts_c = np.array([ts[ts_cuts[1,i]-l:ts_cuts[1,i],i] for i in range(num)]).T
    
    window_size = l//8
    if window_size < 10:
        window_size = 10
   
    # variance
    ts_v = rolling_variance(ts_c, window_size); ts_vp = trend_test(ts_v[window_size:,:])
    # autocorrelation
    ts_a = rolling_autocorr(ts_c, window_size); ts_ap = trend_test(ts_a[window_size:,:])
    # phi fit
    #ts_l = first_order_phi_OLS(ts_c, window_size); ts_lp = trend_test(ts_l[window_size:,:], 1000)
    ts_lb, ts_l, _ = ar1n_fit(ts_c, window_size)
    ts_lbp = trend_test(ts_lb[window_size:,:])
    ts_lp  = trend_test(ts_l[int(2*window_size):,:])
    
    positives = np.empty([len(thresholds),5])
    for i in range(len(thresholds)):
        positives[i, 0] = thresholds[i]
        # positives
        positives[i, 1] = np.sum(ts_vp<thresholds[i])/num
        positives[i, 2] = np.sum(ts_ap<thresholds[i])/num
        positives[i, 3] = np.sum(ts_lbp<thresholds[i])/num
        positives[i, 4] = np.sum(ts_lp<thresholds[i])/num
    return(positives)

lm_p = calc_positives(lm, thresholds)
np.save("roc/" + source + "/lm_positive_n_roc.npy", lm_p)
#np.save("lm_positive_n_roc.npy", lm_p)

#dw_p = calc_positives(dw, thresholds)
#np.save("roc/" + source + "/dw_positive.npy", dw_p)

#sp_p = calc_positives(sp, thresholds)
#np.save("roc/" + source + "/sp_positive.npy", sp_p)