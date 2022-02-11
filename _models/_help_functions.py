# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 15:46:31 2021

@author: boettner
"""

from joblib import Parallel, delayed
import multiprocessing

import os

def parallel_computation(func, num):
    if os.name == "nt": # on windows operating system
         num_cores = multiprocessing.cpu_count()         
    else:               # otherwise (on cluster)
         num_cores = 32
    
    results = Parallel(n_jobs=num_cores)(delayed(func)(n) for n in range(num))
    
    return(results)

def model_selection(number):
    model = {0: "linear_model",
             1: "double_well",
             2: "sub_pitchfork"}
    return(model[number])