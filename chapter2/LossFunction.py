#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 16:56:33 2017

@author: swkim
"""
import numpy as np
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y,t):
    delta = 1e-7
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)
        
    batch_size = y.shape[0]
    print(batch_size)
    return -np.sum(t*np.log(y+delta)) / batch_size

def function_2(x):
    return x[0]**2 + x[1]**2

def numerical_diff (f,x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)
        
        x[idx] = tmp_val - h
        fxh2 = f(x)
        
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)

x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]


t = [0,0,1,0,0,0,0,0,0,0]

y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]

t = np.array(t)
y1 = np.array(y1)
y2 = np.array(y2)