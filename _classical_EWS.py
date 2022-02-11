# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 19:38:10 2021

@author: boettner
"""
import numpy as np
import pandas as pd

def rolling_variance(data, window_size):
    data = pd.DataFrame(data)
    
    variance = data.rolling(window_size).var().values
    return(variance)

def rolling_autocorr(data, window_size):
    if data.ndim == 1:
        data = data[:,np.newaxis]
    
    data1 = pd.DataFrame(data[1:,:])
    data2 = pd.DataFrame(data[:-1,:])
    autocorr = data1.rolling(window_size).corr(data2).values
    
    autocorr = np.vstack([np.empty([1,data1.shape[1]])+np.nan, autocorr]) # return to original length
    return(autocorr)