# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 17:42:24 2020

@author: egoral
"""

def build_classifier():
    #initialise ANN
    classifier = Sequential()

    # EG Adding the input layer and the first hidden layer 
    classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))

    # EG Adding the second hidden layer
    classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))

    # EG Adding the output layer
    classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

    # EG Conpiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return classifier