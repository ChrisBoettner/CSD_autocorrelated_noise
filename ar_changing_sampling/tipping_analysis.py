# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 12:54:41 2021

@author: boettner
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sn = 0.01


dw_n = pd.DataFrame(np.load(f"{sn}_dw_positive_n.npy")).sort_values(by=[0])
sp_n = pd.DataFrame(np.load(f"{sn}_sp_positive_n.npy")).sort_values(by=[0])
dw   = pd.DataFrame(np.load(f"{sn}_dw_positive.npy")).sort_values(by=[0])
sp   = pd.DataFrame(np.load(f"{sn}_sp_positive.npy")).sort_values(by=[0])

path = "C://Users//boettner//Google Drive//Uni//Masterarbeit//Thesis//5_Results//"

win = 10
meandw_n = dw_n.rolling(win, win_type='gaussian').mean(std=2).values
meandw   = dw.rolling(win,   win_type='gaussian').mean(std=2).values
meansp_n = sp_n.rolling(win, win_type='gaussian').mean(std=2).values
meansp   = sp.rolling(win,   win_type='gaussian').mean(std=2).values
#mean=data.values

plt.close("all")

plt.figure()
plt.title(f"double well {sn}")
plt.plot(dw.iloc[:,0],meandw[:,1], label = "variance")
plt.plot(dw.iloc[:,0],meandw[:,2], label = "lag-1 autocorrelation")
plt.plot(dw.iloc[:,0],meandw[:,3], label = "phi biased")
plt.plot(dw.iloc[:,0],meandw[:,4], label = "phi corrected")
plt.xlabel("noise/signal")
plt.ylabel("significant trends/total number")
plt.xscale("log")
plt.legend()
plt.savefig(path + f"103_DW00{str(sn)[-1]}.pdf")
y_lim = plt.ylim()[-1]
plt.figure()
plt.gca().set_ylim(top=y_lim)
plt.title(f"double well {sn} n")
plt.plot(dw.iloc[:,0],meandw_n[:,1], label = "variance")
plt.plot(dw.iloc[:,0],meandw_n[:,2], label = "lag-1 autocorrelation")
plt.plot(dw.iloc[:,0],meandw_n[:,3], label = "phi biased")
plt.plot(dw.iloc[:,0],meandw_n[:,4], label = "phi corrected")
plt.xlabel("noise/signal")
plt.ylabel("significant trends/total number")
plt.xscale("log")
plt.legend()
plt.savefig(path + f"103_DW00{str(sn)[-1]}_n.pdf")
plt.figure()
plt.title(f"double well {sn} diff")
plt.plot(dw.iloc[:,0],meandw[:,3]-meandw[:,4], label = "AR(1) noise")
plt.plot(dw.iloc[:,0],meandw_n[:,3]-meandw_n[:,4], label = "white noise")
plt.xlabel("noise/signal")
plt.ylabel("phi difference")
plt.xscale("log")
plt.legend()
plt.savefig(path + f"103_DW00{str(sn)[-1]}_diff.pdf")

plt.figure()
plt.title(f"subcritical pitchfork {sn}")
plt.plot(sp.iloc[:,0],meansp[:,1], label = "variance")
plt.plot(sp.iloc[:,0],meansp[:,2], label = "lag-1 autocorrelation")
plt.plot(sp.iloc[:,0],meansp[:,3], label = "phi biased")
plt.plot(sp.iloc[:,0],meansp[:,4], label = "phi corrected")
plt.xlabel("noise/signal")
plt.ylabel("significant trends/total number")
plt.xscale("log")
plt.legend()
plt.savefig(path + f"103_SP00{str(sn)[-1]}.pdf")
y_lim = plt.ylim()[-1]
plt.figure()
plt.gca().set_ylim(top=y_lim)
plt.title(f"subcritical pitchfork {sn} n")
plt.plot(sp.iloc[:,0],meansp_n[:,1], label = "variance")
plt.plot(sp.iloc[:,0],meansp_n[:,2], label = "lag-1 autocorrelation")
plt.plot(sp.iloc[:,0],meansp_n[:,3], label = "phi biased")
plt.plot(sp.iloc[:,0],meansp_n[:,4], label = "phi corrected")
plt.xlabel("noise/signal")
plt.ylabel("significant trends/total number")
plt.xscale("log")
plt.legend()
plt.savefig(path + f"103_SP00{str(sn)[-1]}_n.pdf")
plt.figure()
plt.title(f"double well {sn} diff")
plt.plot(sp.iloc[:,0],meansp[:,3]-meansp[:,4], label = "AR(1) noise")
plt.plot(sp.iloc[:,0],meansp_n[:,3]-meansp_n[:,4], label = "white noise")
plt.xlabel("noise/signal")
plt.ylabel("phi difference")
plt.xscale("log")
plt.legend()
plt.savefig(path + f"103_SP00{str(sn)[-1]}_diff.pdf")



