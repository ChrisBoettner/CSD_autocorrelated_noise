# -*- coding: utf-8 -*-
"""
Created on Tue May 11 15:32:59 2021

@author: boettner
"""

import numpy as np
import matplotlib.pyplot as plt

# load
num    = 4
source = "ar_noise"


lm = np.load(source + f"/e{num}/lm_positive_roc.npy");# lm[-1,:] = np.array([1,1,1,1,1])
#lm = np.load(source + f"/e{num}/dw_positive_roc.npy"); lm[-1,:] = np.array([1,1,1,1,1])
#sp = np.load(source + "/sp_positive.npy"); sp[-1,:] = np.array([1,1,1,1,1])

lm_n = np.load(source + f"/e{num}/lm_positive_roc_n.npy");# lm_n[-1,:] = np.array([1,1,1,1,1])
#lm_n = np.load(source + f"/e{num}/dw_positive_roc_n.npy"); lm_n[-1,:] = np.array([1,1,1,1,1])
#sp_n = np.load(source + "/sp_positive_n.npy"); sp_n[-1,:] = np.array([1,1,1,1,1])


# plot
plt.close("all")

    
plt.figure()
plt.title("linear model AR(1) noise")
plt.plot(lm[:,0],lm[:,1], label = "variance")
plt.plot(lm[:,0],lm[:,2], label = "lag-1 autocorrelation")
plt.plot(lm[:,0],lm[:,3], label = "biased phi")
plt.plot(lm[:,0],lm[:,4], label = "phi")
plt.xlabel("significance level")
plt.ylabel("false positive rate")
plt.plot(lm[:,0],lm[:,0], color="black", linestyle= "--", label = "null model")
plt.legend()

plt.figure()
plt.title("linear model, white noise")
plt.plot(lm_n[:,0],lm_n[:,1], label = "variance")
plt.plot(lm_n[:,0],lm_n[:,2], label = "lag-1 autocorrelation")
plt.plot(lm_n[:,0],lm_n[:,3], label = "biased phi")
plt.plot(lm_n[:,0],lm_n[:,4], label = "phi")
plt.xlabel("significance level")
plt.ylabel("false positive rate")
plt.plot(lm_n[:,0],lm_n[:,0], color="black", linestyle= "--", label = "null model")
plt.legend()