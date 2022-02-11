# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 13:29:57 2021

@author: boettner
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sn = 0.01

lm_n = pd.DataFrame(np.load(f"{sn}_lm_positive_n.npy")).sort_values(by=[0])
lm = pd.DataFrame(np.load(f"{sn}_lm_positive.npy")).sort_values(by=[0])

path = "C://Users//boettner//Google Drive//Uni//Masterarbeit//Thesis//5_Results//"

win = 10
meanlm_n = lm_n.rolling(win, win_type='gaussian').mean(std=2).values
meanlm = lm.rolling(win, win_type='gaussian').mean(std=2).values
#mean=data.values

diff = meanlm_n - meanlm

plt.close("all")

plt.figure()
plt.title(f"linear model {sn}")
plt.plot(lm.iloc[:,0],meanlm[:,1], label = "variance")
plt.plot(lm.iloc[:,0],meanlm[:,2], label = "lag-1 autocorrelation")
plt.plot(lm.iloc[:,0],meanlm[:,3], label = "phi biased")
plt.plot(lm.iloc[:,0],meanlm[:,4], label = "phi corrected")
plt.xlabel("noise/signal")
plt.ylabel("significant trends/total number")
plt.xscale("log")
plt.legend()
plt.savefig(path + f"102_LM00{str(sn)[-1]}.pdf")

y_lim = plt.ylim()[-1]

plt.figure()
plt.gca().set_ylim(top=y_lim)
plt.title(f"linear model {sn} n")
plt.plot(lm.iloc[:,0],meanlm_n[:,1], label = "variance")
plt.plot(lm.iloc[:,0],meanlm_n[:,2], label = "lag-1 autocorrelation")
plt.plot(lm.iloc[:,0],meanlm_n[:,3], label = "phi biased")
plt.plot(lm.iloc[:,0],meanlm_n[:,4], label = "phi corrected")
plt.xlabel("noise/signal")
plt.ylabel("significant trends/total number")
plt.xscale("log")
plt.legend()
plt.savefig(path + f"102_LM00{str(sn)[-1]}_n.pdf")
plt.figure()
plt.title(f"linear model {sn} diff")
plt.plot(lm.iloc[:,0],diff[:,1], label = "variance")
plt.plot(lm.iloc[:,0],diff[:,2], label = "lag-1 autocorrelation")
plt.plot(lm.iloc[:,0],diff[:,3], label = "phi biased")
plt.plot(lm.iloc[:,0],diff[:,4], label = "phi corrected")
plt.xlabel("noise/signal")
plt.ylabel("difference")
plt.xscale("log")
plt.legend()
plt.savefig(path + f"102_LM00{str(sn)[-1]}_diff.pdf")



