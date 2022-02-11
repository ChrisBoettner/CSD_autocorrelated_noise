# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 13:59:22 2021

@author: boettner
"""

import numpy as np

t_len = 10000

# control parameter spaces
r_dw = np.linspace(-1, 1, t_len)
r_sp = np.linspace(-4.26, 1.9, t_len)

# roots
dw_roots = []
for r in r_dw:
    dw_roots.append(np.polynomial.polynomial.polyroots([r,1,0,-1])[0])  
dw_roots = np.array(dw_roots); dw_roots[np.iscomplex(dw_roots)] = np.nan
dw_roots = np.real(dw_roots)

sp_roots = np.zeros(t_len); sp_roots[np.where(r_sp>0)] = np.nan

# lambdas
l_dw  = - (-3*dw_roots**2 + 1)
l_sp  = - (-5*sp_roots**5 + 3*sp_roots**2 + r_sp)
l2_dw = - (-6*dw_roots)
l3_sp = - np.linspace(6,6,np.argmax(r_sp>0))

# order of magnitude comparison for higher orders
x_dw = (9*dw_roots**2-1)/(6*dw_roots)     # second order
x_sp = np.sqrt(-r_sp[np.where(r_sp<0)]/6) # third order

from matplotlib import rc_file
rc_file(r'C:\Users\boettner\Google Drive\Uni\Masterarbeit\ews_analysis\__plots\settings.rc')
import matplotlib.pyplot as plt

orange = '#D55E00'
purple = '#330066'

plt.close("all")
fig = plt.figure()
# =============================================================================

l_dw[6924] = 0

plt.close("all")
fig, ax = plt.subplots()

ax.plot(np.linspace(0,1,len(l_dw)), l_dw, label="Double Well", color = "black")
#plt.plot(np.linspace(0,1,len(l_sp)), l_sp, label="Subcritical Pitchfork", color = "grey", linestyle = "--")
ax.plot(np.linspace(0,1,t_len), np.repeat(4.26,t_len), label="Linear Model", color = "black", linestyle = "-.")
ax.set_xlim([0,1])
ax.set_ylim([0,4.5])
ax.legend(loc=6)
ax.set_xlabel(r"Normalized Simulation Time $T$")
ax.set_ylabel("Linear Restoring Rate $\lambda$")

ax1 = ax.twinx()
ax1.plot(np.linspace(0,1,100),np.linspace(-1,1,100),color="red", alpha =0.2, label = "Control Parameter r")
ax1.legend(loc=7)
ax1.set_ylabel("Control Parameter r")

# =============================================================================
fig.tight_layout(pad=0.1) 
plt.subplots_adjust(hspace=0.1, wspace=0.1)