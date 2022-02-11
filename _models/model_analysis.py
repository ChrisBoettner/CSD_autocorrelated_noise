# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 23:10:33 2021

@author: boettner
"""

import numpy as np
import matplotlib.pyplot as plt

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

plt.figure()
plt.plot(np.linspace(0,1,t_len), np.repeat(4.26,t_len), label="Linear Model")
plt.plot(np.linspace(0,1,len(l_dw)), l_dw, label="Double Well")
plt.plot(np.linspace(0,1,len(l_sp)), l_sp, label="Subcritical Pitchfork")
plt.xlim([0,1])
plt.ylim([0,4.5])
plt.legend()
plt.xlabel("time [a.u.]")
plt.ylabel("restoring rate $\lambda$")