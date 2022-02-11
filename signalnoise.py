# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 20:28:18 2021

@author: boettner
"""

import numpy as np
from _classical_EWS import rolling_variance, rolling_autocorr
from _regression    import first_order_phi_OLS 
from _significance  import trend_test
from _help_functions import parallel_computation, model_selection, shorten_ts

import os
import sys

files = 100
p_val = 0.05

model = model_selection(1)
samp        = 3  # log(number of data points)

curr_path = os.getcwd()
sys.path.append(os.path.abspath(curr_path))
def p_calculation(n):
    if model =="double_well":
        ts = np.load(curr_path+f"/changing_variance/dw_{n}_{samp}.npz")
        #ts = np.load(curr_path+f"/models/dw_{n}_3.npz")
    if model =="sub_pitchfork":
        ts = np.load(curr_path+f"/changing_variance/sp_{n}_{samp}.npz")
        #ts = np.load(curr_path+f"/models/sp_{n}_3.npz")
    if model =="linear_model":
        ts = np.load(curr_path+f"/changing_variance/lm_{n}_{samp}.npz")
        #ts = np.load(curr_path+f"/models/sp_{n}_3.npz")


    ts_cuts = shorten_ts(ts); noise = np.sqrt(ts["v"][0]); ts = ts["x"]
    
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
    ts_l = first_order_phi_OLS(ts_c, window_size); ts_lp = trend_test(ts_l[window_size:,:])
    
    # enter noise/signal
    ns = noise/4.26 # noise/lambda_0
    # true positives
    v_tp = np.sum(ts_vp<p_val)/num
    a_tp = np.sum(ts_ap<p_val)/num
    l_tp = np.sum(ts_lp<p_val)/num
    return([ns, v_tp, a_tp, l_tp])

tp = np.stack(parallel_computation(p_calculation, files))

if model =="double_well":
    np.save(f"{samp}_dw_positive.npy", tp)
if model =="sub_pitchfork":
    np.save(f"{samp}_sp_positive.npy", tp)
if model =="linear_model":
    np.save(f"{samp}_lm_positive.npy", tp)

