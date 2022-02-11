# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 08:57:27 2021

@author: boettner
"""

import numpy as np
from _classical_EWS import rolling_variance, rolling_autocorr
from _regression    import first_order_phi_OLS 
from _significance  import trend_test
from _help_functions import shorten_ts

ts = np.load("lm_53_2.npz")
p_val = 0.05

ts_cuts = shorten_ts(ts); noise = np.sqrt(ts["v"][0]); ts = ts["x"]

num = ts.shape[1]

l = np.amin(np.diff(ts_cuts,axis=0)) # length of shortest ts  
ts_c = np.array([ts[ts_cuts[1,i]-l:ts_cuts[1,i],i] for i in range(num)]).T

window_size = l//8
   
# variance
ts_v = rolling_variance(ts_c, window_size); ts_vp = trend_test(ts_v[window_size:,:])
# autocorrelation
ts_a = rolling_autocorr(ts_c, window_size); ts_ap = trend_test(ts_a[window_size:,:])
# phi fit
ts_l = first_order_phi_OLS(ts_c, window_size); ts_lp = trend_test(ts_l[window_size:,:])

# enter noise/signal
ns = noise/4.26 # noise/lambda_0
# true positives
v_tp = np.sum(ts_vp<p_val)/num # variance
a_tp = np.sum(ts_ap<p_val)/num # autocorrelation
l_tp = np.sum(ts_lp<p_val)/num # phi

print(v_tp,a_tp,l_tp)