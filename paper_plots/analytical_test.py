# -*- coding: utf-8 -*-
"""
Created on Mon May 31 15:32:34 2021

@author: boettner
"""
import numpy as np
import pandas as pd

from joblib import Parallel, delayed
import multiprocessing
import os
import sys

def parallel_computation(func, i_len):    
    if os.name == "nt": # on windows operating system
         num_cores = multiprocessing.cpu_count()         
    else:               # otherwise (on cluster)
         num_cores = 32
    results = Parallel(n_jobs=num_cores)(delayed(func)(i) for i in range(i_len))
    return(results)

def ts(phi, rho, var, t_len):
    """
    x_(i+1) = a_i * x_i + np.sqrt(v_i) * random_normal
    """
    sigma = np.sqrt(var)
    noise = np.zeros(t_len)
    for i in range(t_len - 1):
        noise[i + 1] = rho * noise[i] + sigma * np.random.standard_normal(1)
        
    x = np.zeros(t_len)
    x[0] = 0
    for i in range(t_len - 1):
        x[i + 1] = phi * x[i] + noise[i]
    return(x, noise)

def analytical(phi, rho, var):
    noise_var_an = var/(1-rho**2)
    noise_ac_an  = rho
    x_var_an     = 1/(1-phi**2)*noise_var_an*(1+(2*(phi*rho)/(1-phi*rho)))
    x_ac_an      = phi + rho/(1-rho*phi)*noise_var_an/x_var_an
    return(noise_var_an, noise_ac_an, x_var_an, x_ac_an)

def numerical(phi, rho, var):
    x, noise  = ts(phi, rho, var, t_len)
    
    noise_var = np.var(noise)
    noise_ac  = np.corrcoef(noise[1:], noise[:-1])[0,1] 
    x_var     = np.var(x)
    x_ac      = np.corrcoef(x[1:], x[:-1])[0,1]       
    return(noise_var, noise_ac, x_var, x_ac)


phi = 0.5
rho = 0.5    
var = 0.03

noise_var_an, noise_ac_an, x_var_an, x_ac_an = analytical(phi,rho,var)

t_len     = 100000
num       = 100000

noise_var_num = []; noise_ac_num = []; x_var_num = []; x_ac_num = [] 
noise_var_mean = []; noise_ac_mean = []; x_var_mean = []; x_ac_mean = [] 

curr_path = os.getcwd()
sys.path.append(os.path.abspath(curr_path))
def calc_num(i):
    ts = numerical(phi, rho, var)
    noise_var_num = ts[0]
    noise_ac_num  = ts[1]
    x_var_num     = ts[2]
    x_ac_num      = ts[3]
    return(noise_var_num, noise_ac_num, x_var_num, x_ac_num)

results = np.stack(parallel_computation(calc_num, num))

noise_var_rel = (noise_var_an-results[:,0])/noise_var_an
noise_ac_rel  = (noise_ac_an-results[:,1])/noise_ac_an
x_var_rel     = (x_var_an-results[:,2])/x_var_an
x_ac_rel      = (x_ac_an-results[:,3])/x_ac_an

rel = pd.DataFrame(np.array([noise_var_rel,noise_ac_rel,x_var_rel,x_ac_rel]).T)

means  = rel.expanding().mean()

means.to_pickle("means.pkl")

#plt.plot(noise_var_rel, label = "n var")
#plt.plot(noise_ac_rel,  label = "n ac")
#plt.plot(x_var_rel,     label = "x var")
#plt.plot(x_ac_rel,      label = "x ac")
#plt.legend()

        
