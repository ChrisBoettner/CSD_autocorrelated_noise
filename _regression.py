# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 13:41:58 2021

@author: boettner
"""

import numpy as np
import pandas as pd
from statsmodels.regression.rolling import RollingOLS
from _help_functions import parallel_computation

def first_order_phi_OLS(data, window_size):
    if data.ndim == 1:
        data = data[:,np.newaxis]
    
    data1 = data[1:,:]
    data2 = data[:-1,:]
    
    def phi_computation(i):
        model    = RollingOLS(data1[:,i], np.vander(data2[:,i], 2, increasing=True),
                              window=window_size, missing="skip")
        return(model.fit(params_only=True).params[:,1])
        
    phi = np.stack(parallel_computation(phi_computation, data1.shape[1])).T
    
    phi = np.vstack([np.empty([1,data1.shape[1]])+np.nan, phi]) # return to original length
    return(phi)

def ho_fit(data, window_size, order = 1):
    if data.ndim == 1:
        data = data[:,np.newaxis]
    
    data1 = data[1:,:]
    data2 = data[:-1,:]
    
    def bic_computation(i):
        y        = data1[:,i]
        X        = np.vander(data2[:,i], order+1, increasing=True)
        model    = RollingOLS(y, X,
                              window=window_size, missing="skip")
        params = model.fit().params
        s_e2   = [np.linalg.norm(y[j-window_size:j]-np.matmul(X[j-window_size:j],params[j-1,:]))**2/window_size
                  for j in range (window_size,y.shape[0]+1)]
        bic    = window_size*np.log(s_e2) + (order+1)*np.log(window_size)
        return(bic) # calculate model bic
    
    bic = np.stack(parallel_computation(bic_computation, data1.shape[1])).T
    bic = np.vstack([np.empty([window_size,data1.shape[1]])+np.nan, bic]) # return to original length
    return(bic)
# needed for higher order fits, if you want parameter:
#model    = RollingOLS(data1[:,i],np.vander(data2[:,i], 2, increasing=True), # no intercept
#                      window=window_size, missing="skip")
#return(model.fit(params_only=True).params) # here choose which parameter!!

def ar1n_fit(data, window_size):
    if data.ndim == 1:
        data = data[:,np.newaxis]
    
    # calculate biased estimator for restoring force
    data1 = data[1:,:]
    data2 = data[:-1,:]
    def phi_computation(i):
        model    = RollingOLS(data1[:,i], np.vander(data2[:,i], 2, increasing=True),
                              window=window_size, missing="skip")
        return(model.fit(params_only=True).params)    
    params_star = np.stack(parallel_computation(phi_computation, data1.shape[1])).T
    
    # calculate biased residuals, ar coefficient
    e    = data1- params_star[0,:,:] - params_star[1,:,:]*data2
    e1   = e[1:,:]; e2 = e[:-1,:]
    e1e2 = pd.DataFrame(e1*e2);   e22  = pd.DataFrame(e2**2)
    e1e2 = e1e2.rolling(window_size).sum().values
    e22  = e22.rolling(window_size).sum().values
    rho_star  = e1e2/e22
    
    # calculate corrected estimator
    a = params_star[1,1:,:] + rho_star
    b = rho_star/params_star[1,1:,:]
    c = a**2 - 4*b # to check if sqrt can be taken, and calc true vals
    c[c<0] = np.nan
    phi = (a + np.sqrt(c))/2
    rho = (a - np.sqrt(c))/2
    
     # return to original length
    phi_star = np.vstack([np.empty([1,data1.shape[1]])+np.nan, params_star[1,:,:]])
    phi      = np.vstack([np.empty([2,data1.shape[1]])+np.nan, phi])
    rho      = np.vstack([np.empty([2,data1.shape[1]])+np.nan, rho])
    return(phi_star, phi, rho)
    