# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 13:23:26 2021

@author: boettner
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

samp = 2

dw_n = pd.DataFrame(np.load(f"{samp}_dw_positive_n.npy"))
sp_n = pd.DataFrame(np.load(f"{samp}_sp_positive_n.npy"))
dw   = pd.DataFrame(np.load(f"{samp}_dw_positive.npy"))
sp   = pd.DataFrame(np.load(f"{samp}_sp_positive.npy")) 

path = "C://Users//boettner//Google Drive//Uni//Masterarbeit//Thesis//5_Results//"

win = 10
meandw_n = dw_n.rolling(win, win_type='gaussian').mean(std=2).values
meandw   = dw.rolling(win,   win_type='gaussian').mean(std=2).values
meansp_n = sp_n.rolling(win, win_type='gaussian').mean(std=2).values
meansp   = sp.rolling(win,   win_type='gaussian').mean(std=2).values

plt.close("all")

plt.figure()
plt.title(f"double well {samp}")
plt.plot(dw.iloc[:,0],meandw[:,1], label = "variance")
plt.plot(dw.iloc[:,0],meandw[:,2], label = "lag-1 autocorrelation")
plt.plot(dw.iloc[:,0],meandw[:,3], label = "phi biased")
plt.plot(dw.iloc[:,0],meandw[:,4], label = "phi corrected")
plt.xlabel("noise/signal")
plt.ylabel("significant trends/total number")
plt.legend()
plt.savefig(path + f"103_DW{samp}.pdf")
plt.figure()
plt.title(f"double well {samp} n")
plt.plot(dw.iloc[:,0],meandw_n[:,1], label = "variance")
plt.plot(dw.iloc[:,0],meandw_n[:,2], label = "lag-1 autocorrelation")
plt.plot(dw.iloc[:,0],meandw_n[:,3], label = "phi biased")
plt.plot(dw.iloc[:,0],meandw_n[:,4], label = "phi corrected")
plt.xlabel("noise/signal")
plt.ylabel("significant trends/total number")
plt.legend()
plt.savefig(path + f"103_DW{samp}_n.pdf")
plt.figure()
plt.title(f"double well {samp} diff")
plt.plot(dw.iloc[:,0],meandw[:,3]-meandw[:,4], label = "AR(1) noise")
plt.plot(dw.iloc[:,0],meandw_n[:,3]-meandw_n[:,4], label = "white noise")
plt.xlabel("noise/signal")
plt.ylabel("phi difference")
plt.legend()
plt.savefig(path + f"103_DW{samp}_diff.pdf")

plt.figure()
plt.title(f"subcritical pitchfork {samp}")
plt.plot(sp.iloc[:,0],meansp[:,1], label = "variance")
plt.plot(sp.iloc[:,0],meansp[:,2], label = "lag-1 autocorrelation")
plt.plot(sp.iloc[:,0],meansp[:,3], label = "phi biased")
plt.plot(sp.iloc[:,0],meansp[:,4], label = "phi corrected")
plt.xlabel("noise/signal")
plt.ylabel("significant trends/total number")
plt.legend()
plt.savefig(path + f"103_SP{samp}.pdf")
plt.figure()
plt.title(f"subcritical pitchfork {samp} n")
plt.plot(sp.iloc[:,0],meansp_n[:,1], label = "variance")
plt.plot(sp.iloc[:,0],meansp_n[:,2], label = "lag-1 autocorrelation")
plt.plot(sp.iloc[:,0],meansp_n[:,3], label = "phi biased")
plt.plot(sp.iloc[:,0],meansp_n[:,4], label = "phi corrected")
plt.xlabel("noise/signal")
plt.ylabel("significant trends/total number")
plt.legend()
plt.savefig(path + f"103_SP{samp}_n.pdf")
plt.figure()
plt.title(f"subcritical pitchfork {samp} diff")
plt.plot(sp.iloc[:,0],meansp[:,3]-meansp[:,4], label = "AR(1) noise")
plt.plot(sp.iloc[:,0],meansp_n[:,3]-meansp_n[:,4], label = "white noise")
plt.xlabel("noise/signal")
plt.ylabel("phi difference")
plt.legend()
plt.savefig(path + f"103_SP{samp}_diff.pdf")