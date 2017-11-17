#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 11:07:28 2017

@author: hc

Global config 
"""
import shutil, os
import numpy as np
import pandas as pd
from scipy import misc
import pickle
import matplotlib.pyplot as plt


ModelsPath = "KerasModels/"
DataPath = "data/play"
RBGMode = True          #If false, recorded data shall be Gray
ConvFolder2Gray = False #If True, recorded RGB data is converted to Gray

#Number of images to stack and train (memory)
#Shall be Either 4 or 0 (1 same as 0) else keras model in CarRacing_Learn shall be updated either


Temporal_Buffer = 4 


#try to feed in diffs of images and remove bias in CNN
# Cyclic epslilon decay (reinit after a number of preriods - same possible for learning rate)
#Coursera robotics
# try RGB
#Check how to go out of local min stuck- reset periodically LR too
# reward calculation as sum: R or without
def rgb2gray(rgb, norm):
    #Consider to normalize features
    gray = np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    
    if norm:
        gray = gray.astype('float32') / 128 - 1 # normalize
        #check typecased division if float
    return gray 


def save_data(Path, action_l, reward_l, state_l):
    ''' Saves actions, rewards and states (images) in DataPath'''
    if not os.path.exists(Path):
        os.makedirs(Path)
    else:
        shutil.rmtree(Path)
        os.makedirs(Path)
    
    df = pd.DataFrame(action_l, columns=["Steering", "Throttle", "Brake"])
    df["Reward"] = reward_l
    df.to_csv(Path+'/CarRacing_ActionsRewards.csv', index=False)
    
    #img = np.empty(len(state_l))
    for i in range(len(state_l)):
        if RBGMode == False:
            image = rgb2gray(state_l[i])
        else:
            image = state_l[i]
        ##misc.imsave("d7ata/Img"+str(i)+".png", state_l[i])
        misc.imsave(Path+"/Img"+str(i)+".png", image)
      
        
        
def save_DDQL(Path, Name, agent, R):
    ''' Saves actions, rewards and states (images) in DataPath'''
    if not os.path.exists(Path):
        os.makedirs(Path)
    agent.brain.model.save(Path+Name)
    print(Name, "saved")
    print('...')
    
    dump_pickle(agent.memory, Path+Name+'Memory')
    
    dump_pickle([agent.epsilon, agent.steps, agent.brain.opt.get_config()], Path+Name+'AgentParam')
    dump_pickle(R, Path+Name+'Rewards')
    print('Memory pickle dumped')


def dump_pickle(obj, name):
    with open(name, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
def load_pickle(name):
    with open(name, 'rb') as handle:
        obj = pickle.load(handle)
    return obj

def Plot(array, save):   
    ''' Plot 1D array '''    
    plt.plot(array)
    axes = plt.gca()
    axes.set_xlim([900,1500])
    axes.set_ylim([-10,200])
    plt.xlabel('Episodes')
    plt.ylabel('array')
    plt.title('plot')
    plt.grid(True)
    if save:
        plt.savefig("Rplot.png")
    plt.show()