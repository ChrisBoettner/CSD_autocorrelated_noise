# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 11:57:20 2021

@author: boettner
"""

import numpy as np
from _classical_EWS import rolling_variance, rolling_autocorr
from _regression    import ar1n_fit 
from _significance  import trend_test_shuffle
from _help_functions import parallel_computation, model_selection, shorten_ts

import os
import sys

files = 100
p_val = 0.05

model         = model_selection(int(sys.argv[1]))
memory_saving = True
sn            = 0.07

curr_path = os.getcwd()
sys.path.append(os.path.abspath(curr_path))
def p_calculation(n):
    if model =="double_well":
        ts = np.load(curr_path+f"/ar_changing_sampling/dw_{sn}_{n}.npz")
        #ts = np.load(curr_path+f"/changing_sampling/dw_{sn}_{n}.npz")
    if model =="sub_pitchfork":
        ts = np.load(curr_path+f"/ar_changing_sampling/sp_{sn}_{n}.npz")
        #ts = np.load(curr_path+f"/changing_sampling/sp_{sn}_{n}.npz")
    if model =="linear_model":
        ts = np.load(curr_path+f"/ar_changing_sampling/lm_{sn}_{n}.npz")
        #ts = np.load(curr_path+f"/changing_sampling/lm_{sn}_{n}.npz")

    ts_cuts = shorten_ts(ts); ts = ts["x"]
    
    num = ts.shape[1]

    l = np.amin(np.diff(ts_cuts,axis=0)) # length of shortest ts  
    ts_c = np.array([ts[ts_cuts[1,i]-l:ts_cuts[1,i],i] for i in range(num)]).T
    
    window_size = l//8
    if window_size < 10:
        window_size = 10
        
    # variance
    ts_v = rolling_variance(ts_c, window_size); ts_vp = trend_test_shuffle(ts_v[window_size:,:])
    # autocorrelation
    ts_a = rolling_autocorr(ts_c, window_size); ts_ap = trend_test_shuffle(ts_a[window_size:,:])
    # phi fit
    ts_lb, ts_l, _ = ar1n_fit(ts_c, window_size);
    ts_lbp = trend_test_shuffle(ts_lb[window_size:,:])
    ts_lp  = trend_test_shuffle(ts_l[int(2*window_size):,:])
    
    samp = l # available data points
    # true positives
    v_tp = np.sum(ts_vp<p_val)/np.sum(~np.isnan(ts_vp))
    a_tp = np.sum(ts_ap<p_val)/np.sum(~np.isnan(ts_ap))
    lb_tp = np.sum(ts_lbp<p_val)/np.sum(~np.isnan(ts_lbp))
    l_tp = np.sum(ts_lp<p_val)/np.sum(~np.isnan(ts_lp))
    return([samp, v_tp, a_tp, lb_tp, l_tp])

tp = np.stack(parallel_computation(p_calculation, files))

if model =="double_well":
    np.save(f"{sn}_dw_positive.npy", tp)
if model =="sub_pitchfork":
    np.save(f"{sn}_sp_positive.npy", tp)
if model =="linear_model":
    np.save(f"{sn}_lm_positive.npy", tp)