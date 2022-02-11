# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 17:10:46 2021

@author: boettner
"""

import numpy as np

def fourier_surrogates(data, surrogate_num = 100):
    if data.ndim == 1:
        data = data[:,np.newaxis]
        
    data_fourier  = np.fft.rfft(data,axis = 0).T # FT data
    
    # create random phases
    random_angles = np.random.uniform(0, 2 * np.pi,(surrogate_num, data.shape[0] // 2 + 1))
    random_phases = np.exp(1.0j * random_angles)
    
    four_surrogates = data_fourier * np.repeat(random_phases[:,np.newaxis], 
                                               data.shape[1], axis=1)   # create surrogates
    
    # inverse fourier transform
    data_surrogates = np.real(np.fft.irfft(four_surrogates,axis=2)).T
    return(data_surrogates)


def trend_test(data,  surrogate_num = 100, memory_saving = False, absolute = False,
               slope = False):   
    x_space = np.arange(data.shape[0])
    
    ref = np.polynomial.polynomial.polyfit(x_space, data, 1)[1] # calculate reference value for slope
    
    # memory saving mode (looping through realizations, slower)
    if memory_saving:
        p = []
        for i in range(data.shape[1]):
            surrogates = fourier_surrogates(data[:,i], surrogate_num)
            # perfom slope fits to surrogates
            fits = np.array([np.polynomial.polynomial.polyfit(np.arange(surrogates.shape[0]),
                                                              surrogates[:,:,i], 1)[1]
                             for i in range(surrogate_num)])  
            # calculate p value
            if absolute:
                p_i = 1 - np.sum(np.less(np.abs(fits),np.abs(ref[i])),axis=0)/surrogate_num
            else:
                p_i = 1 - np.sum(np.less(fits,ref[i]),axis=0)/surrogate_num
            p.append(*p_i)
        p = np.array(p)
    
    # faster, but more memory required
    else:
        surrogates = fourier_surrogates(data, surrogate_num)        # create time series surrogates
        # perfom slope fits to surrogates
        fits = np.array([np.polynomial.polynomial.polyfit(np.arange(surrogates.shape[0]),
                                                          surrogates[:,:,i], 1)[1]
                        for i in range(surrogate_num)])  
        # calculate p value
        if absolute:
            p = 1 - np.sum(np.less(np.abs(fits),np.abs(ref)),axis=0)/surrogate_num
        else:
            p = 1 - np.sum(np.less(fits,ref),axis=0)/surrogate_num
    if slope:
        return(p, ref) # return p values and slopes
    else:
        return(p) # return only p values

    
def trend_test_shuffle(data,  surrogate_num = 100, 
               absolute = False, slope = False):   
    x_space = np.arange(data.shape[0])
    
    ref = []
    p   = []
    for i in range(data.shape[1]):
        d = data[:,i]
        
        idx   = np.isfinite(d)
        if np.sum(idx)<0.3*len(x_space):
            ref.append(np.nan)
            p.append(np.nan)
        else:
            ref_i = np.polynomial.polynomial.polyfit(x_space[idx], d[idx], 1)[1]
            
            surrogates = np.zeros((d.shape[0],1,surrogate_num))
            for j in range(surrogate_num):
                surrogates[:,0,j] = np.random.permutation(d) 
                
            # perfom slope fits to surrogates
            idc = np.isfinite(surrogates).reshape(len(x_space),surrogate_num)
            fits = np.array([np.polynomial.polynomial.polyfit(
                             x_space[idc[:,k]],
                             surrogates[:,:,k][idc[:,k]], 1)[1]
                             for k in range(surrogate_num)])  
            # calculate p value
            if absolute:
                p_i = 1 - np.sum(np.less(np.abs(fits),np.abs(ref_i)),axis=0)/surrogate_num
            else:
                p_i = 1 - np.sum(np.less(fits,ref_i),axis=0)/surrogate_num
            ref.append(ref_i)
            p.append(*p_i)
    ref = np.array(ref)
    p   = np.array(p)
    
    if slope:
        return(p, ref) # return p values and slopes
    else:
        return(p) # return only p values    