# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 16:16:25 2021

@author: boettner
"""

import numpy as np
from models import ar1_noise, linear_model, double_well, sub_pitchfork
from _help_functions import parallel_computation, model_selection

import os
import sys
   
def create_ts(model, a, sigma, r, general_parameter):
    dt   = general_parameter[0]
    samp = general_parameter[1]
    num  = general_parameter[2]
    
    sys.path.append(os.path.abspath(os.getcwd()))
    def ts_calculation(i):      
        noise  = ar1_noise(a, sigma) 
        if model == "linear_model":
            ts     = linear_model(0, r, dt, noise)[::samp]
        if model == "double_well":
            ts     = double_well(-1.325, r, dt, noise)[::samp]
        if model == "sub_pitchfork":
            ts     = sub_pitchfork(0, r, dt, noise)[::samp]
        return(ts)
        
    x = np.stack(parallel_computation(ts_calculation, num)).T
    return(x)

# cutting
def cut_tipping(model, x, threshold):
    if x.ndim == 1:
        x = x[:,np.newaxis]
    
    length   = x.shape[0]
    n        = x.shape[1]
    
    if model == "linear_model":
        tipping = np.repeat(int(0.4*length), n)+int(0.5*length)
    if model == "double_well": # cut before detrending!
        tipping = np.argmax(x>threshold, axis = 0) # look for tipping 
        tipping[tipping == 0] = int(0.7*length) # if tp can't be found
    if model == "sub_pitchfork":
        tipping =np.argmax(np.abs(x)>threshold, axis = 0)
        tipping[tipping == 0] = int(0.7*length)
    # end cut
    end = (tipping - 0.02*length).astype(int)
    # maximum possible length
    cut_length = 0.4*length
    # begin cut
    start = end - cut_length
    start[start<0] = 0                                      # if ts shorter than 0.4*length, start at 0

    cut_indices = np.array([start,end]).astype(int)
    return(cut_indices)

# general parameter
t_len = 100000                               # number of time steps
dt    = 0.01                                # interval between time steps
samp  = 1                                  # sampling interval

model = model_selection(1)                  # model selection
num   = 100                                   # number of time series

saving = True

# model parameter (ar1)
if model == "double_well":
    r         = np.linspace(-1, 1, t_len)       # control parameter
if model == "sub_pitchfork":
    r         = np.linspace(-4.26, 1.9, t_len)
if model == "linear_model":
    r         = np.linspace(-4.26, -4.26, t_len)
    
a         = np.linspace(0.4, 0.4, t_len)            # autocorrelation

variances = np.linspace(0.000020,0.22, 3)
variances = [0.02890626] # ns = 0.04
a         = np.linspace(0.4, 0.4, t_len)   

for i, var in enumerate(variances):
    v     = np.linspace(var, var, t_len)        # noise strength
    
    # create and cut
    x   = create_ts(model, a, v, r, [dt, samp, num])
    cut = cut_tipping(model, x, 1)
    
    # detrend
    if model == "double_well":
        ts_wn  = double_well(-1.325, r, dt, np.zeros(x.shape[0]*samp))[::samp]
        ts_wn  = ts_wn[:,np.newaxis]
        x      = x - ts_wn
    
    # save
    if saving:
        t = (np.arange(t_len)*dt)[::samp]
        data = {
                "t"     : t,
                "x"     : x,
                "cut"   : cut,
                "t_len" : t_len,
                "dt"    : dt,
                "samp"  : samp, 
                "model" : model,
                "r"     : r,
                "a"     : a,
                "v"     : v}
        
        if model == "double_well":
            np.savez(f"dw_{i}_{np.log10(x.shape[0]).astype(int)}.npz", **data)
        if model == "sub_pitchfork":
            np.savez(f"sp_{i}_{np.log10(x.shape[0]).astype(int)}.npz", **data)
        if model == "linear_model":
            np.savez(f"lm_{i}_{np.log10(x.shape[0]).astype(int)}.npz", **data)