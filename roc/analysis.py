# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 16:35:12 2021

@author: boettner
"""

import numpy as np
import matplotlib.pyplot as plt

# load
source = "simple"


lm = np.load(source + "/lm_positive.npy"); lm[-1,:] = np.array([1,1,1,1])
dw = np.load(source + "/dw_positive.npy"); dw[-1,:] = np.array([1,1,1,1])
sp = np.load(source + "/sp_positive.npy"); sp[-1,:] = np.array([1,1,1,1])

# sort
def sort(fp, tp):
    ind = fp.argsort()
    return(fp[ind],tp[ind])

dw_v = sort(lm[:,1], dw[:,1]); sp_v = sort(lm[:,1], sp[:,1])
dw_a = sort(lm[:,2], dw[:,2]); sp_a = sort(lm[:,2], sp[:,2])
dw_l = sort(lm[:,3], dw[:,3]); sp_l = sort(lm[:,3], sp[:,3])

    
#smooth
#win = 12
#mean = data.rolling(win, win_type='gaussian').mean(std=2).values
#mean=data.values


# plot
plt.close("all")

plt.figure()
plt.title("double well")
plt.plot(*dw_v, label = "variance")
plt.plot(*dw_a, label = "lag-1 autocorrelation")
plt.plot(*dw_l, label = "phi")
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.plot(lm[:,0],lm[:,0], color="black", linestyle= "--", label = "random classifier")
plt.legend()
print(1-np.trapz(*dw_v), 1-np.trapz(*dw_a), 1-np.trapz(*dw_l)) # AUC, probability for positive detection

plt.figure()
plt.title("subcritical pitchfork")
plt.plot(*sp_v, label = "variance")
plt.plot(*sp_a, label = "lag-1 autocorrelation")
plt.plot(*sp_l, label = "phi")
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.plot(lm[:,0],lm[:,0], color="black", linestyle= "--", label = "random classifier")
plt.legend()
print(1-np.trapz(*sp_v), 1-np.trapz(*sp_a), 1-np.trapz(*sp_l))