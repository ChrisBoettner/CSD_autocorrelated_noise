# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 10:44:47 2021

@author: boettner
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load

dw = np.load("dw_positive.npy")
sp = np.load("sp_positive.npy")
lm = np.load("lm_positive.npy")

size  = (dw[:,0]/400) # 400 is 0.4*length for these, so available data
ratiodw = pd.DataFrame(dw/lm)
ratiosp = pd.DataFrame(sp/lm)

# smooth
win = 10
meandw = ratiodw.rolling(win, win_type='gaussian').mean(std=2).values
meansp = ratiosp.rolling(win, win_type='gaussian').mean(std=2).values


# plot
plt.title("double well")
plt.plot(size,meandw[:,1], label = "variance")
plt.plot(size,meandw[:,2], label = "lag-1 autocorrelation")
plt.plot(size,meandw[:,3], label = "phi")
plt.xlabel("window size/total data")
plt.ylabel("true positive rate/ false positive rate")
plt.legend()

plt.figure()
plt.title("subcritical pitchfork")
plt.plot(size,meansp[:,1], label = "variance")
plt.plot(size,meansp[:,2], label = "lag-1 autocorrelation")
plt.plot(size,meansp[:,3], label = "phi")
plt.xlabel("window size/total data")
plt.ylabel("true positive rate/ false positive rate")
plt.legend()

# plot
#plt.figure()
#plt.plot(ts[:,0],ts[:,1], label = "variance")
#plt.plot(ts[:,0],ts[:,2], label = "lag-1 autocorrelation")
#plt.plot(ts[:,0],ts[:,3], label = "phi")
#plt.xlabel("window size")
#plt.ylabel("positive rate")