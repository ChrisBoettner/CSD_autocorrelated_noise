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
dt    = 0.01                                # interval between time steps
samp  = 10                                  # sampling interval

model = model_selection(1)                  # model selection
num   = 5                                   # number of time series

saving = True


t_length = np.logspace(3,3, 1).astype(int)                 # number of time steps

for i, t_len in enumerate(t_length):
    if model == "double_well":
        r         = np.linspace(-1, 1, t_len)       # control parameter
    if model == "sub_pitchfork":
        r         = np.linspace(-4.26, 1.9, t_len)
    if model == "linear_model":
        r         = np.linspace(-4.26, -4.26, t_len)
    
    # model parameter (ar1)
    a         = np.linspace(0, 0, t_len)                # autocorrelation
    v         = np.linspace(0.029, 0.029, t_len)        # noise strength
    
    # create and cut
    x   = create_ts(model, a, v, r, [dt, samp, num])
    cut = cut_tipping(model, x, 1)
    
    # detrend
    if model == "double_well":
        ts_wn  = double_well(-1.325, r, dt, np.zeros(a.shape[0]))[::samp]
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
        
        sn = np.round(np.sqrt(v[0])/4.26,2)
        if model == "double_well":
            np.savez(f"dw_{sn}_{i}.npz", **data)
        if model == "sub_pitchfork":
            np.savez(f"sp_{sn}_{i}.npz", **data)
        if model == "linear_model":
            np.savez(f"lm_{sn}_{i}.npz", **data)