# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 14:09:36 2020

@author: vipasha
"""

import numpy as np
from perceptron import Perceptron

training_inputs=[]
training_inputs.append(np.array([0,0,0]))
training_inputs.append(np.array([1,1,0]))
training_inputs.append(np.array([0,1,0]))
training_inputs.append(np.array([0,1,1]))
training_inputs.append(np.array([1,0,0]))
training_inputs.append(np.array([1,1,1]))

labels=np.array([0,0,0,0,0,1])
perceptron=Perceptron(3)
perceptron.train(training_inputs,labels) 

inputs=np.array([1,1,0])
print(perceptron.predict(inputs))