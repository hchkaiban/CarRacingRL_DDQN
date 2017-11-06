#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 10:22:59 2017

@author: hc
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:03:46 2017

@author: hc
"""
# Add checkpoint callback
# End episode on greed screen?

import CarConfig
import random, math, gym
from SumTree import SumTree
import numpy as np

ModelsPath = CarConfig.ModelsPath
LoadWeithsAndTest = True

IMAGE_WIDTH = 96
IMAGE_HEIGHT = 96
IMAGE_STACK = 3
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_STACK)

LEARNING_RATE = 0.001
#HUBER_LOSS_DELTA = 2.0

action_buffer = np.array([
                    [0.0, 0.0, 0.0],     #Brake
                    [-0.6, 0.05, 0.0],   #Sharp left
                    [0.6, 0.05, 0.0],    #Sharp right
                    [0.0, 0.3, 0.0]] )   #Staight

NumberOfDiscActions = len(action_buffer)
    
def SelectAction(Act):
    return action_buffer[Act]

def SelectArgAction(Act):
    for i in range(NumberOfDiscActions):
        if np.all(Act == action_buffer[i]):
            return i
    raise ValueError('SelectArgAction: Act not in action_buffer')
#    
#def f_softmax(x):
#    """Compute softmax values for each sets of scores in x."""
#    e_x = np.exp(x - np.max(x))
#    return e_x / e_x.sum(axis=0)

#-------------------- BRAIN ---------------------------
import keras.optimizers as Kopt
from keras.layers import Convolution2D
from keras.layers import Dense, Flatten, Input, concatenate, Dropout
from keras.models import Model
from keras.utils import plot_model

class Brain:

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.model = self._createModel()
        self.model_ = self._createModel()  # target network

    def _createModel(self):
        brain_in = Input(shape=self.stateCnt, name='brain_in')
        x = brain_in
        x = Convolution2D(16, (16,16), strides=(2,2), activation='relu')(x)
        x = Convolution2D(32, (4,4), strides=(2,2), activation='relu')(x)
        x = Convolution2D(32, (3,3), strides=(2,2), activation='relu')(x)
        x = Flatten(name='flattened')(x)
        
        x = Dense(256, activation='relu')(x)
        
        x = Dense(self.actionCnt, activation="linear")(x)
        
        model = Model(inputs=brain_in, outputs=x)
        opt = Kopt.RMSprop(lr=LEARNING_RATE)
        model.compile(loss='mean_squared_error', optimizer=opt)
      
        plot_model(model, to_file='brain_model.png', show_shapes = True)
    
        return model
    
    def train(self, x, y, epochs=1, verbose=0):
        self.model.fit(x, y, batch_size=30, epochs=epochs, verbose=verbose)

    def predict(self, s, target=False):
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)

    def predictOne(self, s, target=False):
        x = s[np.newaxis,:,:,:]
        return self.predict(x, target)

    def updateTargetModel(self):
        self.model_.set_weights(self.model.get_weights())

#-------------------- MEMORY --------------------------
class Memory:   # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample) 

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append( (idx, data) )

        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)

#-------------------- AGENT ---------------------------

MEMORY_CAPACITY = int(4e4)
BATCH_SIZE = 320

MAX_REWARD = 100

MAX_NB_EPISODES = int(1e3) #total episodes = random + larning agent
MAX_NB_STEP = 1200

GAMMA = 0.97
MAX_EPSILON = 1
MIN_EPSILON = 0.05

EXPLORATION_STOP = int(MAX_NB_STEP*50)   # at this step epsilon will be 0.01
LAMBDA = - math.log(0.01) / EXPLORATION_STOP  # speed of decay fn of episodes of learning agent

UPDATE_TARGET_FREQUENCY = int(2e2)  #Threshold on counted learning agent's steps

ACTION_REPEAT = 4

class Agent:
    steps = 0
    epsilon = MAX_EPSILON
    memory = Memory(MEMORY_CAPACITY)
    
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt
        
        self.brain = Brain(stateCnt, self.actionCnt)
        self.no_state = np.zeros(stateCnt)
        self.x = np.zeros((BATCH_SIZE,)+IMAGE_SIZE)
        self.y = np.zeros([BATCH_SIZE, self.actionCnt])
        self.errors = np.zeros(BATCH_SIZE)
        self.rand = False
        
        self.agentType = 'Learning'
        
    def act(self, s):
        if random.random() < self.epsilon:
            #return random.randint(0, self.actionCnt-1)
            best_act = np.random.randint(self.actionCnt)
            self.rand=True
            return SelectAction(best_act), SelectAction(best_act)
        else:
            #x = s[np.newaxis,:,:,:]
            act_soft = self.brain.predictOne(s)
            best_act = np.argmax(act_soft)
            self.rand=False
            return SelectAction(best_act), act_soft

    def observe(self, sample):  # in (s, a, r, s_) format
        #x, y, errors = self._getTargets([(0, sample)])
        #self.memory.add(errors[0], sample)

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.brain.updateTargetModel()
            print
        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def _getTargets(self, batch):
        states = np.array([ o[1][0] for o in batch ])
        states_ = np.array([ (self.no_state if o[1][3] is None else o[1][3]) for o in batch ])
             
        p = agent.brain.predict(states)
        
        p_ = agent.brain.predict(states_, target=False)
        pTarget_ = agent.brain.predict(states_, target=True)

        for i in range(len(batch)):
            o = batch[i][1]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            
            a = SelectArgAction(a)
            t = p[i]
            
            oldVal = t[a]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * pTarget_[i][ np.argmax(p_[i]) ]  # double DQN
            
            self.x[i] = s
            self.y[i] = t
            
            #sec_bets = t[np.argsort(t)[::-1][1]]
            if self.steps % 20 == 0 and i == len(batch)//2:
                print('a',a, 't',t[a], 'r: %.4f' % r,'mean t',np.mean(t))
                   
            self.errors[i] = abs(oldVal - t[a])

        return (self.x, self.y, self.errors)

    def replay(self):    
        batch = self.memory.sample(BATCH_SIZE)
        x, y, errors = self._getTargets(batch)

        #update errors
        for i in range(len(batch)):
            idx = batch[i][0]
            self.memory.update(idx, errors[i])

        self.brain.train(x, y)
        
#    def _to_onehot(self, arg):
#        a = np.zeros(self.actionCnt)
#        a[arg] = 1
#        return a

class RandomAgent:
    memory = Memory(MEMORY_CAPACITY)
    exp = 0
    steps = 0
    
    def __init__(self, actionCnt):
        self.actionCnt = actionCnt
        self.agentType = 'Random'

    def act(self, s):
        best_act = np.random.randint(self.actionCnt)
        return SelectAction(best_act), SelectAction(best_act)

    def observe(self, sample):  # in (s, a, r, s_) format
        error = abs(sample[2])  # reward
        self.memory.add(error, sample)
        self.exp += 1
        self.steps += 1
        
    def replay(self):
        pass

#-------------------- ENVIRONMENT ---------------------
class Environment:
    def __init__(self, problem):
        self.problem = problem
        self.env = gym.make(problem)
        self.env.seed(2)
        from gym import envs
        envs.box2d.car_racing.WINDOW_H = 500
        envs.box2d.car_racing.WINDOW_W = 600
        
        self.episode = 0
        self.reward = []
        self.step = 0
        
    def run(self, agent):              
        img = self.env.reset()
        img =  CarConfig.rgb2gray(img)
        s = np.zeros(IMAGE_SIZE)
        for i in range(IMAGE_STACK):
            s[:,:,i] = img
            
        s_ = s
        #a = [0.0, 0.3, 0.0]
        R = 0
        self.step = 0

        while True: 
            #img_old = img
            if self.step % ACTION_REPEAT == 0:
                a, a_soft = agent.act(s)
            
                img, r, done, info = self.env.step(a)
            #s_ = np.array([s[1], processImage(img)]) #last two screens
            
                if not done:
                    img =  CarConfig.rgb2gray(img)
                #for i in range(IMAGE_STACK-1,-1,-1):    #count down
                    for i in range(IMAGE_STACK-1):
                        s_[:,:,i] = s_[:,:,i+1]
                    s_[:,:,IMAGE_STACK-1] = img
            
                else:
                   s_ = None

                R += r
                r = R/MAX_REWARD    
                #r = np.clip(r, -1 ,1)
            
                agent.observe( (s, a, r, s_) )
                agent.replay()            
                s = s_
            
            if (self.step % 20 == 0) and (agent.agentType=='Learning'):
                print('step:', self.step, 'R: %.1f' % R, a_soft, 'rand:', agent.rand)
            
            self.step += 1
            
            if done or (R<-5) or (self.step > MAX_NB_STEP):
                self.episode += 1
                self.reward.append(R)
                break

        print("Episode ",self.episode,"/", MAX_NB_EPISODES, agent.agentType) 
        print("Avg Episode R:", R/self.step, "Total R:", sum(self.reward))
    
    
    def test(self, agent):
        img = self.env.reset()
        img = CarConfig.rgb2gray(img)
        s = np.zeros(IMAGE_SIZE)
        for i in range(IMAGE_STACK):
            s[:,:,i] = img

        R = 0
        self.step = 0
        done = False
        while True :
            self.env.render('human')
            
            if self.step == 0:
                act = SelectAction(3)
                act1=[0,0,0,0]
            else:
                
                #img_old = img
                if self.step % ACTION_REPEAT == 0:
                    if(agent.agentType == 'Learning'):
                        act1 = agent.brain.predictOne(s)
                        act = SelectAction(np.argmax(act1))
                    else:
                        act = agent.act(s)
                
                    img, r, done, info = self.env.step(act)
                    img = CarConfig.rgb2gray(img)
                    R += r
            
                    for i in range(IMAGE_STACK-1):
                        s[:,:,i] = s[:,:,i+1]
                    s[:,:,IMAGE_STACK-1] = img
            
            if(self.step % 10) == 0:
                print('Step:', self.step, 'action:',act, 'R: %.1f' % R)
            
            self.step += 1
            
            if done or (R<-5) or (agent.steps > MAX_NB_STEP):
                R = 0
                self.step = 0
                break
            
#-------------------- MAIN ----------------------------
if __name__ == "__main__":

    PROBLEM = 'CarRacing-v0'
    env = Environment(PROBLEM)
    
    stateCnt  = IMAGE_SIZE
    
    actionCnt = env.env.action_space.shape[0] #env.env.action_space.n
    assert action_buffer.shape[1] == actionCnt, "Lenght of Env action space does not match action buffer"
    
    agent = Agent(stateCnt, NumberOfDiscActions)
    randomAgent = RandomAgent(NumberOfDiscActions)
    
    try:
        #Train agent
        if LoadWeithsAndTest == False:
            print("Initialization with random agent. Fill memory")
            while randomAgent.exp < MEMORY_CAPACITY:
                env.run(randomAgent)
                print(randomAgent.exp, "/", MEMORY_CAPACITY)
        
            agent.memory = randomAgent.memory
            randomAgent = None
        
            print("Starts learning")
            
            while env.episode < MAX_NB_EPISODES:
                env.run(agent)
            
            agent.brain.model.save(ModelsPath+"CarRacing_DDQN_model.h5")  
            print("CarRacing_DDQN_model.h5 saved")
            
        else:
            print('Load agent and play')
            agent.brain.model.load_weights(ModelsPath+"CarRacing_DDQN_model.h5")
            
            done_ctr = 0
            while done_ctr < 5 :
                env.test(agent)
                done_ctr += 1
                
                
    except KeyboardInterrupt:
        print('User interrupt')
        env.env.close()
        
        if LoadWeithsAndTest == False:
             print('Save model: Y or N?')
             save = input()
             if save == 'Y':
                 agent.brain.model.save(ModelsPath+"CarRacing_DDQN_model.h5")
                 print("CarRacing_DDQN_model.h5 saved")
             else:
                print('Model discarded')
            
    #finally:
     #   agent.brain.model.save(ModelsPath+"CarRacing_DDQN_model.h5")