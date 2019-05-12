#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 17:12:00 2019

@author: tom
"""

import numpy as np
import pylab as plt
import os
import pickle
# =============================================================================
# batch_size = 4
# seq_len = 20
# 
# t_min = 0
# t_max = 30
# 
# rng = t_max - t_min
# 
# resolution = 0.1
# 
# step_size = batch_size
# n_steps = int((rng//batch_size)/resolution)
# 
# t = np.linspace(t_min, t_max, int(rng//resolution) + 1)
# 
# time_len = t.shape[0]
# 
# 
# n_seqs = int(np.floor(time_len // (seq_len * batch_size)))
# for i in range(n_seqs):
#     res = t[i * seq_len * batch_size : (i+1) * seq_len *batch_size]
#     res = res.reshape(-1, seq_len, 1)
#     print(res)
#     print(res.shape)
#     
# =============================================================================
    

plt.figure(figsize=(15,8))
    
def time_series(t):
    return t * np.sin(t) / 3 + 2 * np.sin(t*5)
# =============================================================================
# 
# def sine_generator(batch_size, seq_len, t_min=0, t_max=30, resolution=0.1, shift=1):
#     shift *= resolution
#     rng = t_max - t_min
#     t = np.linspace(t_min, t_max, int(rng//resolution) + 1)
#     t_len = t.shape[0]
#     n_batches = int(np.ceil(t_len / (seq_len * batch_size)))
#     r = t_len % seq_len
#     if r > 0:    
#         t = t[:-r]
#     for batch_idx in range(n_batches):    
#         res = t[batch_idx * seq_len * batch_size : (batch_idx+1) * seq_len *batch_size]
#         res = res.reshape(-1, seq_len, 1)
#         yield res, time_series(res), time_series(res+shift)
#         
# =============================================================================
        
def sine_generator(batch_size, seq_len, t_min=0, t_max=30, resolution=0.1, shift=1):
        shift *= resolution
        rng = t_max - t_min
        t = np.linspace(t_min, t_max, int(rng//resolution) + 1)
        t_len = t.shape[0]
        n_batches = int(np.ceil(t_len / (seq_len * batch_size)))
        r = t_len % seq_len
        if r > 0:
            t = t[:-r]
        for batch_idx in range(n_batches):
            res = t[batch_idx * seq_len * batch_size : (batch_idx+1) * seq_len * batch_size]
            res = res.reshape(-1, seq_len, 1)
            yield res, time_series(res), time_series(res + shift)
        return
        
def stochastic_sine_generator(batch_size, seq_len, t_min=0, t_max=30, resolution=0.1, shift=1, random_density=1):
    shift *= resolution
    rng = (t_max - t_min)
    n_ticks = int(rng/resolution)
    if n_ticks < seq_len:
        raise ValueError('sequence length cannot be greater than generator range')
    t0 = np.random.uniform(t_min, t_max-(seq_len*resolution), rng * random_density)
    t = np.arange(0, seq_len)*resolution
    tr = t + t0.reshape(-1, 1)
    tr_indexes = list(range(tr.shape[0]))
    for _t in tr:
        _t = np.random.choice(tr_indexes, batch_size)
        _t = tr[_t]
        yield _t, time_series(_t), time_series(_t + shift) 
    return

generator = stochastic_sine_generator(10, 50, shift=20)

for t, y, l in generator:
    print('gen')
    for _t, _y in zip(t, y):
        plt.plot(_t.ravel(), _y.ravel())
    for _t, _l in zip(t, l):
        plt.plot(_t.ravel(), _l.ravel())


generator = sine_generator(1, 200, shift=20, t_min=30, t_max=50)

for t, y, l in generator:
    print('gen')
    for _t, _y in zip(t, y):
        plt.plot(_t.ravel(), _y.ravel())
    for _t, _l in zip(t, l):
        plt.plot(_t.ravel(), _l.ravel())

    
    
#gen = sine_generator(batch_size=1, seq_len=110)
#for t, y, l in gen:
#    print(t, y)
#    print(t.shape, y.shape)
#    plt.scatter(t.ravel(), y.ravel())
#    plt.scatter(t, l)


# =========================================================================
    
PARAM_MODE = 'BATCH_SIZE'

files = os.listdir(f'./stats/{PARAM_MODE}/')    


plt.figure(figsize=(15,8))
plt.title('validation_loss')
for file_name in files:
    if file_name[:5] == 'valid':
        f = open(f'./stats/{file_name}', 'rb')
        stat=pickle.load(f)
        params=pickle.load(f)
        plt.plot(stat.values(), label=f'BS: {params["BATCH_SIZE"]}')
        plt.legend()
        f.close()

plt.title('test_loss')
plt.figure(figsize=(15,8))
plt.yscale('log')
for file_name in files:
    if file_name[:5] == 'train':
        f = open(f'./stats/{file_name}', 'rb')
        stat=pickle.load(f)
        params=pickle.load(f)
        plt.plot(stat.values(), label=f'BS: {params["BATCH_SIZE"]}')
        plt.legend()
        f.close()



