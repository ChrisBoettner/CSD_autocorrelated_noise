# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 12:58:59 2021

@author: boettner
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

samp = 2

lm_n = pd.DataFrame(np.load(f"{samp}_lm_positive_n.npy"))
lm = pd.DataFrame(np.load(f"{samp}_lm_positive.npy"))

path = "C://Users//boettner//Google Drive//Uni//Masterarbeit//Thesis//5_Results//"

win = 10
meanlm_n = lm_n.rolling(win, win_type='gaussian').mean(std=2).values
meanlm   = lm.rolling(win,   win_type='gaussian').mean(std=2).values

diff = meanlm_n - meanlm

plt.close("all")

plt.figure()
plt.title(f"linear model {samp}")
plt.plot(lm.iloc[:,0],meanlm[:,1], label = "variance")
plt.plot(lm.iloc[:,0],meanlm[:,2], label = "lag-1 autocorrelation")
plt.plot(lm.iloc[:,0],meanlm[:,3], label = "phi biased")
plt.plot(lm.iloc[:,0],meanlm[:,4], label = "phi corrected")
plt.xlabel("noise/signal")
plt.ylabel("significant trends/total number")
plt.legend()
plt.savefig(path + f"102_LM{samp}.pdf")

y_lim = plt.ylim()[-1]

plt.figure()
plt.gca().set_ylim(top=1)
plt.title(f"linear model {samp} n")
plt.plot(lm.iloc[:,0],meanlm_n[:,1], label = "variance")
plt.plot(lm.iloc[:,0],meanlm_n[:,2], label = "lag-1 autocorrelation")
plt.plot(lm.iloc[:,0],meanlm_n[:,3], label = "phi biased")
plt.plot(lm.iloc[:,0],meanlm_n[:,4], label = "phi corrected")
plt.xlabel("noise/signal")
plt.ylabel("significant trends/total number")
plt.legend()
plt.savefig(path + f"102_LM{samp}_n.pdf")
plt.figure()
plt.title(f"linear model {samp} diff")
plt.plot(lm.iloc[:,0],diff[:,1], label = "variance")
plt.plot(lm.iloc[:,0],diff[:,2], label = "lag-1 autocorrelation")
plt.plot(lm.iloc[:,0],diff[:,3], label = "phi biased")
plt.plot(lm.iloc[:,0],diff[:,4], label = "phi corrected")
plt.xlabel("noise/signal")
plt.ylabel("difference")
plt.legend()
plt.savefig(path + f"102_LM{samp}_diff.pdf")