#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 11:07:28 2017

@author: hc
"""
import numpy as np
ModelsPath = "KerasModels/"
DataPath = "data/play"
RBGMode = True          #If false, recorded data shall be Gray
ConvFolder2Gray = False #If True, recorded RGB data is converted to Gray

Temporal_Buffer = 4 #Either 4 or 0 (1 same as 0) else udpate keras model accordingly

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])