# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 09:29:09 2021

@author: boettner
"""

import numpy as np
from _classical_EWS import rolling_variance, rolling_autocorr
from _regression    import first_order_phi_OLS 
from _significance  import trend_test
from _help_functions import parallel_computation, model_selection, shorten_ts

import os
import sys

p_val = 0.05

model         = model_selection(2)

window_sizes = np.arange(20, 240, 2)

curr_path = os.getcwd()
sys.path.append(os.path.abspath(curr_path))
def p_calculation(n):
    if model =="double_well":
        ts = np.load(curr_path+"/changing_variance/dw_13_3.npz") # sn = 0.04
    if model =="sub_pitchfork":
        ts = np.load(curr_path+"/changing_variance/sp_13_3.npz")
    if model =="linear_model":
        ts = np.load(curr_path+"/changing_variance/lm_13_3.npz")
        
    ts_cuts = shorten_ts(ts); ts = ts["x"]
    
    window_size = window_sizes[n]
    
    num = ts.shape[1]

    l = np.amin(np.diff(ts_cuts,axis=0)) # length of shortest ts  
    ts_c = np.array([ts[ts_cuts[1,i]-l:ts_cuts[1,i],i] for i in range(num)]).T
   
    # variance
    ts_v = rolling_variance(ts_c, window_size); ts_vp = trend_test(ts_v[window_size:,:])
    # autocorrelation
    ts_a = rolling_autocorr(ts_c, window_size); ts_ap = trend_test(ts_a[window_size:,:])
    # phi fit
    ts_l = first_order_phi_OLS(ts_c, window_size); ts_lp = trend_test(ts_l[window_size:,:])
    
    # true positives
    v_tp = np.sum(ts_vp<p_val)/num
    a_tp = np.sum(ts_ap<p_val)/num
    l_tp = np.sum(ts_lp<p_val)/num
    return([window_size, v_tp, a_tp, l_tp])

tp = np.stack(parallel_computation(p_calculation, len(window_sizes)))

if model =="double_well":
    np.save("window_size/dw_positive.npy", tp)
if model =="sub_pitchfork":
    np.save("window_size/sp_positive.npy", tp)
if model =="linear_model":
    np.save("window_size/lm_positive.npy", tp)