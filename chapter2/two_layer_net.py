#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 18:50:54 2017

@author: Diana
"""

import sys, os
sys.path.append(os.pardir)

from ANN import sigmoid, softmax

class TwoLayerNet:
    def __init__ (self, input_size, hidden_size, output_size, 
                  weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * \
                   np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
                   np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
    def predict(self, x):
        W1, W2 =  self.params['W1'], self.params['W2']
        b1, b2 =  self.params['b1'], self.params['b2']
        
        a1 = np.dot(x,W1) +  b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2)+b2
        y = softmax(a2)
        
        return y
    
    def loss(self, x, t):
        y = self.predict(x)
        