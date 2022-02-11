# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 11:31:52 2021

@author: boettner
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

samp = 2

lm = pd.DataFrame(np.load(f"{samp}_lm_positive.npy")[:82,:])
dw = pd.DataFrame(np.load(f"{samp}_dw_positive.npy")[:82,:])
sp = pd.DataFrame(np.load(f"{samp}_sp_positive.npy")[:82,:])

path = "C://Users//boettner//Google Drive//Uni//Masterarbeit//Thesis//5_Results//"

#ratio = data/lm
win = 10
#mean = ratio.rolling(win, win_type='gaussian').mean(std=2).values
meanlm = lm.rolling(win, win_type='gaussian').mean(std=2).values
meandw = dw.rolling(win, win_type='gaussian').mean(std=2).values
meansp = sp.rolling(win, win_type='gaussian').mean(std=2).values
#mean=data.values

plt.close("all")

plt.figure()
plt.title(f"linear model {samp}")
plt.plot(lm.iloc[:,0],meanlm[:,1], label = "variance")
plt.plot(lm.iloc[:,0],meanlm[:,2], label = "lag-1 autocorrelation")
plt.plot(lm.iloc[:,0],meanlm[:,3], label = "phi")
plt.xlabel("noise/signal")
plt.ylabel("significant trends/total number")
plt.legend()
plt.savefig(path + f"84_LM{samp}.pdf")

plt.figure()
plt.title(f"double well {samp}")
plt.plot(dw.iloc[:,0],meandw[:,1], label = "variance")
plt.plot(dw.iloc[:,0],meandw[:,2], label = "lag-1 autocorrelation")
plt.plot(dw.iloc[:,0],meandw[:,3], label = "phi")
plt.xlabel("noise/signal")
plt.ylabel("significant trends/total number")
plt.legend()
plt.savefig(path + f"84_DW{samp}.pdf")

plt.figure()
plt.title(f"subcritical pitchfork {samp}")
plt.plot(sp.iloc[:,0],meansp[:,1], label = "variance")
plt.plot(sp.iloc[:,0],meansp[:,2], label = "lag-1 autocorrelation")
plt.plot(sp.iloc[:,0],meansp[:,3], label = "phi")
plt.xlabel("noise/signal")
plt.ylabel("significant trends/total number")
plt.legend()
plt.savefig(path + f"84_SP{samp}.pdf")
#plt.ylabel("significant trends/total number")
#plt.xscale("log")