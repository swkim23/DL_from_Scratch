#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 17:16:20 2017

@author: Diana
"""

import sys, os
sys.path.append(os.pardir) #부모 디렉토리의 파일을 가져 올 수 있도록 설정
import numpy as np
from dataset.mnist import load_mnist
import pickle
from PIL import Image
from ANN import sigmoid, softmax


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()
    return

def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl",'rb') as f:
        network = pickle.load(f)

    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = softmax(a3)
    
    return y

x, t = get_data()
network = init_network()

accuracy_cnt = 0
batch_size = 100

for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1
        
print (accuracy_cnt)
print (len(x))

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch,axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print(y_batch[1])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
    