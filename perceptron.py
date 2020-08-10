# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 15:17:17 2020

@author: vipas
"""

import numpy as np
class Perceptron(object):
    def __init__(self,no_of_inputs,threshold=100,learning_rate=0.5):
        self.threshold=threshold
        self.learning_rate=learning_rate
        self.weights=np.zeros(no_of_inputs+1)
    def predict(self,inputs):
        summation=np.dot(inputs,self.weights[1:])+self.weights[0]
        #print("sum")
        ##print(summation)
        #x=self.weights
        #print(x)
        print("Inputs : ", inputs, end="\t")
        print("Labels : ", label)
        print("Outputs : ", prediction)
        print("Weights : ", self.weights[1:])
        print("Bais : ", self.weights[0])        
        if summation>0:
            activation=1
        else:
            activation=0
        return activation
    def train(self,training_inputs,labels):
        for _ in range(self.threshold):
            for inputs,label in zip(training_inputs,labels):
                prediction=self.predict(inputs)
                self.weights[1:]+=self.learning_rate*(label-prediction)*inputs
                self.weights[0]+=self.learning_rate*(label-prediction)
