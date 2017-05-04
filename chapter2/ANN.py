#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 16:05:00 2017

@author: Diana
"""

import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    return np.array(x > 0, dtype=np.int)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0,x)

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c) #오버플로우 방지
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y


#x = np.arange([ -5.0, 5.0, 0.1])
#x = np.arange(-5.0, 5.0, 0.1)
#y = step_function(x)
#plt.plot(x,y)
#plt.ylim(-0.1, 1.1)
#plt.show()

#x = np.arange(-5.0, 5.0, 0.1)
#y = sigmoid(x)
#plt.plot(x,y)
#plt.ylim(-0.1, 1.1)
#plt.show()
#
#x = np.arange(-5.0, 5.0, 0.1)
#y = relu(x)
#plt.plot(x,y)
#plt.ylim(-0.1, 1.1)
#plt.show()

x = np.arange(-5.0, 5.0, 0.1)
#y = softmax(x)
y = np.exp(x)
plt.plot(x,y)
plt.ylim(-0.1, 1.1)
plt.show()
