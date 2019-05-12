#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 12:47:46 2019

@author: tom
"""

import matplotlib.pyplot as plt
import numpy as np

t_min, t_max = 0, 30
resolution = 0.1

batch_size=50
n_steps=20
batch_idx = 0

def time_series(t):
    return t * np.sin(t) / 3 + 2 * np.sin(t*5)

def next_batch(batch_size, n_steps):
    t0 = np.random.rand(batch_size, 1) * (t_max - t_min - n_steps * resolution)
    Ts = t0 + np.arange(0., n_steps + 1) * resolution
    ys = time_series(Ts)
    batch_idx += 1
    return ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(-1, n_steps, 1)


batch_idx = 0
def sine_batch(batch_size, n_steps):
    global batch_idx
    t = np.arange( (batch_idx * batch_size * n_steps), ((batch_idx + 1) * batch_size * n_steps), step=resolution )
    ys = t * np.sin(t) / 3 + 2 * np.sin(t*5)
    batch_idx += 1
    return ys[19:-1].reshape(-1, n_steps, 1), ys[20:].reshape(-1, n_steps, 1)
    
#y = next_batch(batch_size, n_steps)
#idx = 0
#for yi in y:
#    yi = yi.ravel()
#    ln = yi.shape[0]
#    plt.plot( range(idx * ln, (idx + 1)*ln ), yi.ravel())
#    idx += 1    
    

x, y = sine_batch(batch_size, n_steps)
print(y.shape)
plt.plot(y.ravel()[:100])
    
