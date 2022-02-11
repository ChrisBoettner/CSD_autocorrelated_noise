# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 14:03:08 2021

@author: boettner
"""

import numpy as np

# =============================================================================
# noise
# =============================================================================
def ar1_noise(a, v):
    """
    x_(i+1) = a_i * x_i + np.sqrt(v_i) * random_normal
    """
    sigma = np.sqrt(v)
    noise = np.zeros(len(a))
    for i in range(len(a) - 1):
        noise[i + 1] = a[i] * noise[i] + sigma[i] * np.random.standard_normal(1)
    return(noise)

# =============================================================================
# models
# =============================================================================

def linear_model(x_0, r, dt, noise):
    """
    dx = -k * x *  dt + noise * sqrt(dt)
    """
    x = np.zeros(len(noise))
    x[0] = x_0               # inital value
    for i in range(len(noise) - 1):
        x[i + 1] = x[i] + r[i] * x[i] * dt + np.sqrt(dt) * noise[i]
    return(x)

def double_well(x_0, r, dt, noise):
    x = np.zeros(len(noise))
    x[0] = x_0               # inital value
    for i in range(len(noise) - 1):
        x[i + 1] = x[i] + (-x[i]**3+x[i]+r[i]) * dt + np.sqrt(dt) * noise[i]
    return(x)

def sub_pitchfork(x_0, r, dt, noise):
    x = np.zeros(len(noise))
    x[0] = x_0               # inital value
    for i in range(len(noise) - 1):
        x[i + 1] = x[i] + (-x[i]**5+x[i]**3+r[i]*x[i]) * dt + np.sqrt(dt) * noise[i]
    return(x)
    