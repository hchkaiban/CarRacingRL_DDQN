"""
Created on Thu Nov  2 16:03:46 2017

@author: hc
"""
        
import CarConfig
import random, math, gym
from SumTree import SumTree
import numpy as np


ModelsPath = CarConfig.ModelsPath
LoadWeithsAndTest = False  #Validate model, no training
LoadWeithsAndTrain = False  #Load model and saved agent and train further
TrainEnvRender = False       #Diplay game while training

ENABLE_EPSILON_RST = True

IMAGE_WIDTH = 96
IMAGE_HEIGHT = 96
IMAGE_STACK = 4     
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_STACK)

LEARNING_RATE = 0.001     #atari 0.00025
HUBER_LOSS_DELTA = 1.0

dropout_thr = 0.1
ENV_SEED = 2

MEMORY_CAPACITY = int(1e4)  #atari 1e6, shall be ratio of number of frames
BATCH_SIZE = 120            #atari is 32

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

  
#def f_softmax(x):
#    """Compute softmax values for each sets of scores in x."""
#    e_x = np.exp(x - np.max(x))
#    return e_x / e_x.sum(axis=0)


def huber_loss(y_true, y_pred):
    from keras import backend as K
    err = y_true - y_pred

    cond = K.abs(err) <= HUBER_LOSS_DELTA
    if cond == True:
        loss = 0.5 * K.square(err)

    else:
        loss = 0.5 * HUBER_LOSS_DELTA**2 + HUBER_LOSS_DELTA * (K.abs(err) - HUBER_LOSS_DELTA)

    return K.mean(loss)


#-------------------- BRAIN ---------------------------
import keras
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

        self.ModelsPath_cp = ModelsPath+"CarRacing_DDQN_model_cp.h5"
        self.ModelsPath_cp_per = ModelsPath+"CarRacing_DDQN_model_cp_per.h5"
        
        save_best = keras.callbacks.ModelCheckpoint(self.ModelsPath_cp,
                                                monitor='loss',
                                                verbose=1,
                                                save_best_only=True,
                                                mode='min',
                                                period=20)
        save_per = keras.callbacks.ModelCheckpoint(self.ModelsPath_cp_per,
                                                monitor='loss',
                                                verbose=1,
                                                save_best_only=False,
                                                mode='min',
                                                period=400)
        
#        early_stop = keras.callbacks.EarlyStopping(monitor='loss',
#                                           min_delta=0.001,   
#                                           patience=0,
#                                           verbose=1,
#                                           mode='auto')
    
        self.callbacks_list = [save_best, save_per]#, early_stop]
  
      
    def _createModel(self):
        brain_in = Input(shape=self.stateCnt, name='brain_in')
        x = brain_in
        x = Convolution2D(16, (16,16), strides=(2,2), activation='relu')(x)
        x = Convolution2D(32, (8,8), strides=(2,2), activation='relu')(x)
        #x = Convolution2D(32, (3,3), strides=(2,2), activation='relu')(x)
        x = Flatten(name='flattened')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(self.actionCnt, activation="linear")(x)
        
        model = Model(inputs=brain_in, outputs=x)
        
        self.opt = Kopt.RMSprop(lr=LEARNING_RATE)
        #self.opt = Kopt.RMSprop()
        
        #'mean_squared_error'
        model.compile(loss=huber_loss, optimizer=self.opt)
      
        plot_model(model, to_file='brain_model.png', show_shapes = True)
    
        return model
 
    
    def train(self, x, y, epochs=1, verbose=0):
        self.model.fit(x, y, batch_size=(BATCH_SIZE//2), epochs=epochs, verbose=verbose, callbacks=self.callbacks_list)
        
   
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

MAX_REWARD = 10
GREEN_SC_PLTY = 0.4

ACTION_REPEAT = 8

MAX_NB_EPISODES = int(10e3) 
MAX_NB_STEP = ACTION_REPEAT * 100

GAMMA = 0.99            #atari 0.99
MAX_EPSILON = 1
MIN_EPSILON = 0.02      #atari 0.1

EXPLORATION_STOP = int(MAX_NB_STEP*10)          # at this step epsilon will be MIN_EPSILON
LAMBDA = - math.log(0.001) / EXPLORATION_STOP   # speed of decay fn of episodes of learning agent

UPDATE_TARGET_FREQUENCY = int(200)              #Threshold on counted learning agent's steps # atari 10 000

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
        self.maxEpsilone = MAX_EPSILON
        
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
        x, y, errors = self._getTargets([(0, sample)])
        self.memory.add(errors[0], sample)

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.brain.updateTargetModel()
            #print ("Target network update")
        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (self.maxEpsilone - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def _getTargets(self, batch):
        states = np.array([ o[1][0] for o in batch ])
        states_ = np.array([ (self.no_state if o[1][3] is None else o[1][3]) for o in batch ])
             
        p = agent.brain.predict(states)
        
        p_ = agent.brain.predict(states_, target=False)
        pTarget_ = agent.brain.predict(states_, target=True)
        act_ctr = np.zeros(self.actionCnt)
        
        for i in range(len(batch)):
            o = batch[i][1]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            
            a = SelectArgAction(a)
            t = p[i]
            act_ctr[a] += 1
            # Append target for plotting
            oldVal = t[a]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * pTarget_[i][ np.argmax(p_[i]) ]  # double DQN
            
            self.x[i] = s
            self.y[i] = t
            
            #sec_best = t[np.argsort(t)[::-1][1]]
            if self.steps % 20 == 0 and i == len(batch)-1:
                print('t',t[a], 'r: %.4f' % r,'mean t',np.mean(t))
                print ('act ctr: ', act_ctr)
                   
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
        self.rand = True

    def act(self, s):
        np.random.seed(1)
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
        self.env.seed(ENV_SEED)
        from gym import envs
        envs.box2d.car_racing.WINDOW_H = 500
        envs.box2d.car_racing.WINDOW_W = 600
        
        self.episode = 0
        self.reward = []    
        self.step = 0
        self.action_stuck = 0
        
    def run(self, agent):
        self.env.seed(ENV_SEED)              
        img = self.env.reset()
        img =  CarConfig.rgb2gray(img, True)
        s = np.zeros(IMAGE_SIZE)
        for i in range(IMAGE_STACK):
            s[:,:,i] = img
            
        s_ = s
        R = 0
        self.step = 0
                        
        a_soft = a_old = np.zeros(agent.actionCnt)
        a = action_buffer[0]
        while True: 
            if agent.agentType=='Learning' : 
                if TrainEnvRender == True :
                    self.env.render('human')
 

            if self.step % ACTION_REPEAT == 0:
                
                if agent.rand == False:
                    a_old = a_soft
                
                a, a_soft = agent.act(s)
                
                #Enable periodic reset of epsilon & LR to avoid local minima
                if ENABLE_EPSILON_RST:
                    if agent.rand == False:
                        if a_soft.argmax() == a_old.argmax():
                            self.action_stuck += 1
                            if self.action_stuck >= 200:
                                print('\n ______Stuck in local minimum, epsilon reset')
                                agent.steps = 0
                                agent.brain.opt.lr.set_value(LEARNING_RATE*10)
                                
                                self.action_stuck = 0
                        else:
                            self.action_stuck = max(self.action_stuck -2, 0)
                            agent.brain.opt.lr.set_value(LEARNING_RATE)
                        
                img_rgb, r, done, info = self.env.step(a)
            
                if not done:
                    img =  CarConfig.rgb2gray(img_rgb, True)
                    #for i in range(IMAGE_STACK-1,-1,-1):    #count down
                    for i in range(IMAGE_STACK-1):
                        s_[:,:,i] = s_[:,:,i+1]
                    s_[:,:,IMAGE_STACK-1] = img
            
                else:
                   s_ = None

                R += r
                r = (r/MAX_REWARD)# - ( GREEN_SC_PLTY * abs(np.mean(img_rgb[:,:,1]) - 150) ) / (185.1 - 150)    
                if np.mean(img_rgb[:,:,1]) > 185.0:
                    r -= GREEN_SC_PLTY
 
                r = np.clip(r, -1 ,1)
            
                agent.observe( (s, a, r, s_) )
                agent.replay()            
                s = s_
            
            else:
                img_rgb, r, done, info = self.env.step(a)
                if not done:
                    img =  CarConfig.rgb2gray(img_rgb, True)
                    #for i in range(IMAGE_STACK-1,-1,-1):    #count down
                    for i in range(IMAGE_STACK-1):
                        s_[:,:,i] = s_[:,:,i+1]
                    s_[:,:,IMAGE_STACK-1] = img
                else:
                   s_ = None
                R += r
                s = s_
                
            if (self.step % (ACTION_REPEAT * 5) == 0) and (agent.agentType=='Learning'):
                print('step:', self.step, 'R: %.1f' % R, a, 'rand:', agent.rand)
            
            self.step += 1
            
            if done or (R<-5) or (self.step > MAX_NB_STEP) or np.mean(img_rgb[:,:,1]) > 185.1:
                self.episode += 1
                self.reward.append(R)
                print('Done:', done, 'R<-5:', (R<-5), 'Green>185.1:',np.mean(img_rgb[:,:,1]))
                break

        print("\n Episode ",self.episode,"/", MAX_NB_EPISODES, agent.agentType) 
        print("Avg Episode R:", R/self.step, "Total R:", sum(self.reward))
    
    
    def test(self, agent):
        self.env.seed(ENV_SEED)
        img= self.env.reset()
        img = CarConfig.rgb2gray(img, True)
        s = np.zeros(IMAGE_SIZE)
        for i in range(IMAGE_STACK):
            s[:,:,i] = img

        R = 0
        self.step = 0
        done = False
        while True :
            self.env.render('human')
                            
            #img_old = img
            if self.step % ACTION_REPEAT == 0:
                if(agent.agentType == 'Learning'):
                    act1 = agent.brain.predictOne(s)
                    act = SelectAction(np.argmax(act1))
                else:
                    act = agent.act(s)
                    
                if self.step <= 8:
                    act = SelectAction(3)
                    
                img_rgb, r, done, info = self.env.step(act)
                img = CarConfig.rgb2gray(img_rgb, True)
                R += r
        
                for i in range(IMAGE_STACK-1):
                    s[:,:,i] = s[:,:,i+1]
                s[:,:,IMAGE_STACK-1] = img
            
            if(self.step % 10) == 0:
                print('Step:', self.step, 'action:',act, 'R: %.1f' % R)
                print(np.mean(img_rgb[:,:,0]), np.mean(img_rgb[:,:,1]), np.mean(img_rgb[:,:,2]))
            self.step += 1
            
            if done or (R<-5) or (agent.steps > MAX_NB_STEP) or np.mean(img_rgb[:,:,1]) > 185.1:
                R = 0
                self.step = 0
                print('Done:', done, 'R<-5:', (R<-5), 'Green>185.1:',np.mean(img_rgb[:,:,1]))
                break

           
#-------------------- MAIN ----------------------------
if __name__ == "__main__":

    PROBLEM = 'CarRacing-v0'
    env = Environment(PROBLEM)
    
    stateCnt  = IMAGE_SIZE

    actionCnt = env.env.action_space.shape[0] #env.env.action_space.n
    assert action_buffer.shape[1] == actionCnt, "Lenght of Env action space does not match action buffer"
    
    random.seed(901)
    agent = Agent(stateCnt, NumberOfDiscActions)
    randomAgent = RandomAgent(NumberOfDiscActions)
    
    try:
        #Train agent
        if LoadWeithsAndTest == False:
            if LoadWeithsAndTrain == False:
                print("Initialization with random agent. Fill memory")
                while randomAgent.exp < MEMORY_CAPACITY:
                    env.run(randomAgent)
                    print(randomAgent.exp, "/", MEMORY_CAPACITY)
        
                agent.memory = randomAgent.memory
                randomAgent = None
                
                print("Starts learning")
                
                while env.episode < MAX_NB_EPISODES:
                    env.run(agent)
                
                CarConfig.save_DDQL(ModelsPath, "CarRacing_DDQN_model.h5", agent, env.reward)
                
            else:
                #from keras.models import load_model
                
                print('Load pre-trained agent and learn')
                agent.brain.model.load_weights(ModelsPath+"CarRacing_DDQN_model.h5")
                #agent.brain.model = load_model(ModelsPath+"CarRacing_DDQN_model.h5")
                agent.brain.updateTargetModel()
                try :
                    agent.memory = CarConfig.load_pickle(ModelsPath+"CarRacing_DDQN_model.h5"+"Memory")
                    Params = CarConfig.load_pickle(ModelsPath+"CarRacing_DDQN_model.h5"+"AgentParam")
                    agent.epsilon = Params[0]
                    agent.steps = Params[1]
                    opt = Params[2]
                    agent.brain.opt.decay.set_value(opt['decay'])
                    agent.brain.opt.epsilon = opt['epsilon']
                    agent.brain.opt.lr.set_value(opt['lr'])
                    agent.brain.opt.rho.set_value(opt['rho'])
                    #gent.brain.opt.set_weights(Params[2])
            
                    env.reward = CarConfig.load_pickle(ModelsPath+"CarRacing_DDQN_model.h5"+"Rewards")
                    
                    del Params, opt
                except:
                    print("Invalid DDQL_Memory_.csv to load")
                    print("Initialization with random agent. Fill memory")
                    #Memory empty or corrupted, reload
                    while randomAgent.exp < MEMORY_CAPACITY:
                        env.run(randomAgent)
                        print(randomAgent.exp, "/", MEMORY_CAPACITY)
            
                    agent.memory = randomAgent.memory
                    randomAgent = None
                
                    agent.maxEpsilone = MAX_EPSILON/5
                
                print("Starts learning")
                
                while env.episode < MAX_NB_EPISODES:
                    env.run(agent)
                
                CarConfig.save_DDQL(ModelsPath, "CarRacing_DDQN_model.h5", agent, env.reward)
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
             if save.lower() == 'y':
                 CarConfig.save_DDQL(ModelsPath, "CarRacing_DDQN_model.h5", agent, env.reward)
             else:
                print('Model discarded')
            
    #finally:
     #   agent.brain.model.save(ModelsPath+"CarRacing_DDQN_model.h5")