# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 12:17:33 2020

@author: vipas
"""

import numpy as np
from perceptron import Perceptron
from datetime import datetime
start_time = datetime.now()


training_inputs=[]
training_inputs.append(np.array([1,1]))
training_inputs.append(np.array([0,1]))
training_inputs.append(np.array([1,0]))
training_inputs.append(np.array([0,0]))

labels=np.array([0,1,1,0])
perceptron=Perceptron(2)
perceptron.train(training_inputs,labels) 


inputs=np.array([1,0])

print(perceptron.predict(inputs))
end_time = datetime.now()
print('Process Duration: {}'.format(end_time - start_time))