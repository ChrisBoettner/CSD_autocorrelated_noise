# -*- coding: utf-8 -*-
"""
Created on Mon May 31 20:01:46 2021

@author: boettner
"""

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt

results = pd.read_pickle("means2.pkl").values
results2 = pd.read_pickle("vars2.pkl").values

plt.close()
plt.plot(results)

print(results[-1,:])
print(np.sqrt(results2[-1,:]))