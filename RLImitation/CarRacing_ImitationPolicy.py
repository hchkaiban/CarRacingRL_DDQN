#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 09:55:26 2017

@author: hc
"""

#To generate video:
#ffmpeg -f image2 -r 6 -i 'Img%01d.png' output.mp4


import gym
import numpy as np
from keras.models import load_model
import CarConfig


ModelsPath = CarConfig.ModelsPath
RBG_Mode = CarConfig.RBGMode
ConvFolder2Gray = CarConfig.ConvFolder2Gray
Temporal_Buffer = CarConfig.Temporal_Buffer



class Imitation:
    ''' Predict steering policy for each observation '''
    
    def __init__(self, state=np.zeros([96, 96, 3])):
        if RBG_Mode == False or ConvFolder2Gray==True:
            self.x4 = self.x3 = self.x2 = self.x1 = np.empty([1, 96, 96, 1])
            self.x = self.x1
        else:
            self.x4 = self.x3 = self.x2 = self.x1 = np.empty([1, 96, 96, 3])
            self.x = self.x1
            
        self.model = load_model(ModelsPath+"Model_weights_.h5")
        self.state = state
        
    def update_states(self, state):

        if Temporal_Buffer > 1:
            if RBG_Mode == False or ConvFolder2Gray==True:
                s = CarConfig.rgb2gray(state)
                self.x4 = self.x3
                self.x3 = self.x2
                self.x2 = self.x1
                self.x1[0,:,:,0] = s
            else:
                self.x4 = self.x3
                self.x3 = self.x2
                self.x2 = self.x1
                self.x1[0,:,:,:] = state
       
        else:
            if RBG_Mode == False or ConvFolder2Gray==True:
                self.x[0,:,:,0] = CarConfig.rgb2gray(state)
            else:
                self.x[0,:,:,:] = state

   
    
    def Predict_Angle(self, state=np.zeros([96, 96, 3])):
        ''' Predict actions based on trained model in CarRacing_Learn.py '''

        if Temporal_Buffer > 1:
            self.update_states(state) 
            act = self.model.predict([self.x1,self.x2,self.x3,self.x4])            
        else:
            self.update_states(state) 
            act = self.model.predict(self.x)
         
        #Return leaned policy for steering angle    
        return act[0][0][0]



#########################
#         main          #
######################### 
if __name__ == '__main__':

    env = gym.make('CarRacing-v0')
    env.seed(2)
    
    from gym import envs
    envs.box2d.car_racing.WINDOW_H = 500
    envs.box2d.car_racing.WINDOW_W = 600
    
    state = env.reset()
    Imitate = Imitation(state)
    print(Imitate.Predict_Angle())
     
    env.close()        

