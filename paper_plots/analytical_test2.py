# -*- coding: utf-8 -*-
"""
Created on Mon May 31 15:32:34 2021

@author: boettner
"""
import numpy as np
import time

def analytical(phi, rho, var):
    noise_var_an = var/(1-rho**2)
    noise_ac_an  = rho
    x_var_an     = 1/(1-phi**2)*noise_var_an*(1+(2*(phi*rho)/(1-phi*rho)))
    x_ac_an      = phi + rho/(1-rho*phi)*noise_var_an/x_var_an
    return(noise_var_an, noise_ac_an, x_var_an, x_ac_an)

def numerical(noise, x):
    noise_var = np.var(noise)
    noise_ac  = np.corrcoef(noise[1:], noise[:-1])[0,1] 
    x_var     = np.var(x)
    x_ac      = np.corrcoef(x[1:], x[:-1])[0,1]       
    return(np.array([noise_var, noise_ac, x_var, x_ac]))


phi = 0.9
rho = 0.9   
var = 0.3
sigma = np.sqrt(var)

an = analytical(phi,rho,var)

tol     = 1e-8
tol_met = False


noise = np.random.standard_normal(1)
x     = np.random.standard_normal(1)

i = 0
start = time.time()
while tol_met == False:
    
    noise = np.append(noise, (rho * noise[-1] + sigma * np.random.standard_normal(1)))
    x     = np.append(x, (phi * x[-1] + noise[-1]))
    
    if len(x)%1000 == 0:
        rel = 1 - numerical(noise,x)/an   
        print(rel)
        if (rel < tol).all():
            tol_met = True

end = time.time()

res = np.array([int(len(x)),(end-start)/3600])

np.savetxt("done.txt", res)
