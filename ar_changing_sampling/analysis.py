# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 13:44:38 2021

@author: boettner
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sn = 0.07

n = True

if n: # ts without ar1 noise, with white noise
    lm = pd.DataFrame(np.load(f"{sn}_lm_positive_n.npy")).sort_values(by=[0])
    dw = pd.DataFrame(np.load(f"{sn}_dw_positive_n.npy")).sort_values(by=[0])
    sp = pd.DataFrame(np.load(f"{sn}_sp_positive_n.npy")).sort_values(by=[0])
else:
    lm = pd.DataFrame(np.load(f"{sn}_lm_positive.npy")).sort_values(by=[0])
    dw = pd.DataFrame(np.load(f"{sn}_dw_positive.npy")).sort_values(by=[0])
    sp = pd.DataFrame(np.load(f"{sn}_sp_positive.npy")).sort_values(by=[0])

path = "C://Users//boettner//Google Drive//Uni//Masterarbeit//Thesis//5_Results//"

win = 10
meanlm = lm.rolling(win, win_type='gaussian').mean(std=2).values
meandw = dw.rolling(win, win_type='gaussian').mean(std=2).values
meansp = sp.rolling(win, win_type='gaussian').mean(std=2).values
#mean=data.values

plt.close("all")

plt.figure()
plt.title(f"linear model {sn}")
plt.plot(lm.iloc[:,0],meanlm[:,1], label = "variance")
plt.plot(lm.iloc[:,0],meanlm[:,2], label = "lag-1 autocorrelation")
plt.plot(lm.iloc[:,0],meanlm[:,3], label = "phi biased")
plt.plot(lm.iloc[:,0],meanlm[:,4], label = "phi true")
plt.xlabel("available data points")
plt.ylabel("significant trends/total number")
plt.xscale("log")
plt.legend()
if n:
    plt.savefig(path + f"102_LM00{str(sn)[-1]}_n.pdf")
else:
    plt.savefig(path + f"102_LM00{str(sn)[-1]}.pdf")

plt.figure()
plt.title(f"double well {sn}")
plt.plot(dw.iloc[:,0],meandw[:,1], label = "variance")
plt.plot(dw.iloc[:,0],meandw[:,2], label = "lag-1 autocorrelation")
plt.plot(dw.iloc[:,0],meandw[:,3], label = "phi")
plt.plot(dw.iloc[:,0],meandw[:,4], label = "phi true")
plt.xlabel("available data points")
plt.ylabel("significant trends/total number")
plt.xscale("log")
plt.legend()
if n:
    plt.savefig(path + f"102_DW00{str(sn)[-1]}_n.pdf")
else:
    plt.savefig(path + f"102_DW00{str(sn)[-1]}.pdf")

plt.figure()
plt.title(f"subcritical pitchfork {sn}")
plt.plot(sp.iloc[:,0],meansp[:,1], label = "variance")
plt.plot(sp.iloc[:,0],meansp[:,2], label = "lag-1 autocorrelation")
plt.plot(sp.iloc[:,0],meansp[:,3], label = "phi")
plt.plot(sp.iloc[:,0],meansp[:,4], label = "phi true")
plt.xlabel("available data points")
plt.ylabel("significant trends/total number")
plt.xscale("log")
plt.legend()
if n:
    plt.savefig(path + f"102_SP00{str(sn)[-1]}_n.pdf")
else:
    plt.savefig(path + f"102_SP00{str(sn)[-1]}.pdf")



