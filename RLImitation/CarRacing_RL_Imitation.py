"""
Created on Thu Nov  2 16:03:46 2017

@author: hc
"""
             
import CarConfig
import random, math, gym
from SumTree import SumTree
import numpy as np
import CarRacing_ImitationPolicy

ModelsPath = CarConfig.ModelsPath
LoadWeithsAndTest = False  #Validate model, no training
LoadWeithsAndTrain = True  #Load model and saved agent and train further
TrainEnvRender = True      #Diplay game while training

ENABLE_EPSILON_RST = True

IMAGE_WIDTH = 96
IMAGE_HEIGHT = 96
IMAGE_STACK = 3     
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_STACK)

LEARNING_RATE = 0.0005     #deepmind 0.00025
HUBER_LOSS_DELTA = 1.0

dropout_thr = 0.1
ENV_SEED = 2

MEMORY_CAPACITY = int(5e3)  #deepmind 1e6, shall be ratio of number of frames
BATCH_SIZE = 32             #deepmind is 32

action_buffer = np.array([
                    [0.0, 0.0],     #None   0  
                    #[0.02, 0.0],   #Acc    +
                    #[0.1, 0.0],    #Acc    ++
                    [0.6, 0.0],     #Acc    +++
                    [0.0, 0.3]] )   #Brake  -

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
from keras.layers import Dense, Flatten, Input
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import History

import matplotlib.pyplot as plt
#import gc

class PlotHistory(History):

    def __init__(self, file_name='history.png'):
        History.__init__(self)
        self.file_name = file_name
        self.losses = []
        self.f = plt.figure(0)
        self.ax = self.f.gca()
    
    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.plot_logs(self.f, self.ax)
            
#    def on_batch_end(self, batch, logs={}):
#        self.losses.append(logs.get('loss'))
#        self.plot_logs()
       
    def plot_logs(self, f, ax1):
        
        f.set_figwidth(10)

        ax1.set_title('Training Loss')
        #ax1.set_xlim([0,50])
        ax1.plot(self.losses, label= 'Loss')
        ax1.legend(loc='lower right')
        
        f.savefig(self.file_name)
        ax1.cla()
        plt.close(f)

         

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
        
        
        plot_history_callback = PlotHistory('RL_Im_Loss.png') 
        self.callbacks_list = [save_best, save_per, plot_history_callback ]
  
      
    def _createModel(self):
        brain_in = Input(shape=self.stateCnt, name='brain_in')
        x = brain_in
        x = Convolution2D(12, (16,16), strides=(2,2), activation='relu')(x)
        x = Convolution2D(24, (8,8), strides=(2,2), activation='relu')(x)
        #x = Convolution2D(32, (3,3), strides=(2,2), activation='relu')(x)
        x = Flatten(name='flattened')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(self.actionCnt, activation="linear")(x)
        
        model = Model(inputs=brain_in, outputs=x)
        
        self.opt = Kopt.RMSprop(lr=LEARNING_RATE)
        
        model.compile(loss=huber_loss, optimizer=self.opt)
      
        plot_model(model, to_file='brain_model.png', show_shapes = True)
        
        return model
 
    
    def train(self, x, y, epochs=1, verbose=0):
        self.hist = self.model.fit(x, y, batch_size=(BATCH_SIZE//2), epochs=epochs, verbose=verbose, callbacks=self.callbacks_list)
        
   
    def predict(self, s, target=False):
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)
    
    
    def predictOne(self, s, target=False):
        return self.predict(s[np.newaxis,:,:,:], target)
    
   
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
        self.batch = []
        self.segment = self.tree.total() / n

        for i in range(n):
            s = random.uniform(self.segment * i, self.segment * (i + 1))
            (idx, p, data) = self.tree.get(s)
            self.batch.append( (idx, data) )

        return self.batch

    def update(self, idx, error):
        self.tree.update(idx, self._getPriority(error))

#-------------------- AGENT ---------------------------

MAX_REWARD = 10
GREEN_SC_PLTY = 0.2

ACTION_REPEAT = 12

MAX_NB_EPISODES = int(10e3) 
MAX_NB_STEP = ACTION_REPEAT * 120

GAMMA = 0.99         #deepmind 0.99
MAX_EPSILON = 1
MIN_EPSILON = 0.1    #deepmind 0.1

EXPLORATION_STOP = int(MAX_NB_STEP*5)            # at this step epsilon will be MIN_EPSILON
LAMBDA = - math.log(0.001) / EXPLORATION_STOP    # speed of decay fn of episodes of learning agent

UPDATE_TARGET_FREQUENCY = int(200)               #Threshold on counted learning agent's steps / deepmind 10 000

class Agent:
    steps = 0
    epsilon = MAX_EPSILON
    memory = Memory(MEMORY_CAPACITY)
    Imitate = CarRacing_ImitationPolicy.Imitation()

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
        self.pTarget = []
        self.pStateVal = []
        
    def act(self, s_rl, s_im):
        if random.random() < self.epsilon:
            self.rand=True
            best_act = np.random.randint(self.actionCnt)
            self.a = np.concatenate([[self.Imitate.Predict_Angle(s_im)], SelectAction(best_act)])
            self.act_soft = np.zeros(self.actionCnt)
            del best_act
            
        else:
            self.rand=False 
            self.act_soft = self.brain.predictOne(s_rl)
            act_rl = SelectAction(np.argmax(self.act_soft))
            self.a = np.concatenate([[self.Imitate.Predict_Angle(s_im)], act_rl])
            del act_rl
            
        self.pStateVal.append(np.mean(self.act_soft))
        self.pTarget.append(np.mean(self.brain.predictOne(s_rl, True)))
        
        return self.a, self.act_soft

    def observe(self, sample):  # in (s, a, r, s_) format
        x, y, errors = self._getTargets([(0, sample)])
        self.memory.add(errors[0], sample)
        del x,y, errors
        
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
            
            #sec_bets = t[np.argsort(t)[::-1][1]]
            if self.steps % 20 == 0 and i == len(batch)-1:
                print('t',t[a], 'r: %.4f' % r,'mean t',np.mean(t))
                print ('act ctr: ', act_ctr)
                   
            self.errors[i] = abs(oldVal - t[a])
        del states, states_, p, p_, pTarget_, act_ctr, o, s, a, r, s_, t, oldVal   
        return (self.x, self.y, self.errors)

    def replay(self):    
        self.batch = self.memory.sample(BATCH_SIZE)
        x, y, errors = self._getTargets(self.batch)

        #update errors
        for i in range(len(self.batch)):
            idx = self.batch[i][0]
            self.memory.update(idx, errors[i])

        self.brain.train(x, y)
        del x, y, errors
        
#    def _to_onehot(self, arg):
#        a = np.zeros(self.actionCnt)
#        a[arg] = 1
#        return a

class RandomAgent:
    memory = Memory(MEMORY_CAPACITY)
    Imitate = CarRacing_ImitationPolicy.Imitation()
    exp = 0
    steps = 0
    
    def __init__(self, actionCnt):
        self.actionCnt = actionCnt
        self.agentType = 'Random'
        self.rand = True
        self.act_soft = np.zeros(actionCnt)

    def act(self, s_rl, s_im):
        act_rl = SelectAction(np.random.randint(self.actionCnt))
        a = np.concatenate([[self.Imitate.Predict_Angle(s_im)], act_rl])
            
        return a, act_rl

    def observe(self, sample):  # in (s, a, r, s_) format
        error = abs(sample[2])  # reward
        self.memory.add(error, sample)
        self.exp += 1
        self.steps += 1
        del error
        
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
        self.throttle = self.steering = self.brake = []
        
        self.f, self.ax = plt.subplots(1,3)
        
    def run(self, agent):
        self.env.seed(ENV_SEED)              
        s_im = self.env.reset()
        img =  CarConfig.rgb2gray(s_im, True)
        self.s = np.zeros(IMAGE_SIZE)
        
        for i in range(IMAGE_STACK):
            self.s[:,:,i] = img
            
        self.s_ = self.s
        R = done = r = 0
        self.step = 0
                        
        a_soft = a_old = np.zeros(agent.actionCnt)
        a = np.concatenate([[0], np.r_[0.1,0]])     #little push
        while True: 
            if agent.agentType=='Learning' : 
                if TrainEnvRender == True :
                    self.env.render('human')
 
            if self.step % ACTION_REPEAT == 0:
                
                if agent.rand == False:
                    a_old = a_soft
                if self.step > 20:
                    a, a_soft = agent.act(self.s, s_im)
                self.steering.append(a[0])
                self.throttle.append(a[1])
                self.brake.append(a[2])
                
                
                if ENABLE_EPSILON_RST:
                #Enable periodic epsilon reset                    
                    if agent.rand == False:
                        if a_soft.argmax() == a_old.argmax():
                            self.action_stuck += 1
                            if self.action_stuck >= 120:
                                print('\n ______Stuck in local minimum, epsilon reset')
                                agent.steps = 0
                                agent.brain.opt.lr.set_value(LEARNING_RATE*10)
                                
                                self.action_stuck = 0
                        else:
                            self.action_stuck = max(self.action_stuck -2, 0)
                            agent.brain.opt.lr.set_value(LEARNING_RATE)
                
                
                s_im, r, done, info = self.env.step(a)
            
                if not done:
                    img =  CarConfig.rgb2gray(s_im, True)
                    for i in range(IMAGE_STACK-1):
                        self.s_[:,:,i] = self.s_[:,:,i+1]
                    self.s_[:,:,IMAGE_STACK-1] = img
            
                else:
                   self.s_ = None
                
                R += r
                r = (r/MAX_REWARD)
                
                if np.mean(s_im[:,:,1]) > 185.0:
                    r -= GREEN_SC_PLTY  #penalize off-road
                
                r = np.clip(r, -1 ,1)
            
                agent.observe( (self.s, a[1:3], r, self.s_) )
                agent.replay()            
                self.s = self.s_
                
                self.plot_logs(self.f, self.ax)
                #print  (len( gc.get_objects()))
                #print sys.getsizeof(gc.get_objects())
                
            else:
                a[0] = agent.Imitate.Predict_Angle(s_im)
                s_im, r, done, info = self.env.step(a)
                agent.Imitate.update_states(s_im)
                
                if not done:
                    img =  CarConfig.rgb2gray(s_im, True)
                    #for i in range(IMAGE_STACK-1,-1,-1):    #count down
                    for i in range(IMAGE_STACK-1):
                        self.s_[:,:,i] = self.s_[:,:,i+1]
                    self.s_[:,:,IMAGE_STACK-1] = img
                else:
                   self.s_ = None
                   
                R += r
                
                self.s = self.s_
                
            if (self.step % (ACTION_REPEAT * 5) == 0) and (agent.agentType=='Learning'):
                print('step:', self.step, 'R: %.1f' % R, a, 'rand:', agent.rand)
            
            self.step += 1
            
            if done or (R<-5) or (self.step > MAX_NB_STEP) or np.mean(s_im[:,:,1]) > 185.1:
                self.episode += 1
                self.reward.append(R)
                print('Done:', done, 'R<-5:', (R<-5), 'Green>185.1:',np.mean(s_im[:,:,1]))
                
                agent.pStateVal = []
                agent.pTarget = []
                self.steering = []
                self.throttle = []
                self.brake  = []
                                                            
                break
        
        print("\n Episode ",self.episode,"/", MAX_NB_EPISODES, agent.agentType) 
        print("Avg Episode R:", R/self.step, "Total R:", sum(self.reward))
        del s_im, img, a_soft, a, info, r, R
    
    def test(self, agent):
        self.env.seed(ENV_SEED)
        s_im= self.env.reset()
        img = CarConfig.rgb2gray(s_im, True)
        self.s = np.zeros(IMAGE_SIZE)
        for i in range(IMAGE_STACK):
            self.s[:,:,i] = img

        R = 0
        self.step = 0
        done = False
        act = np.concatenate([[0], np.r_[0.1,0]])
        while True :
            self.env.render('human')
                            
            if self.step % ACTION_REPEAT == 0:
                if self.step > 20:
                    if(agent.agentType == 'Learning'):
                        act1 = agent.brain.predictOne(self.s)
                        act = SelectAction(np.argmax(act1))
                        act = np.concatenate([[agent.Imitate.Predict_Angle(s_im)], act])
                    else:
                        act = agent.act(self.s, s_im)
                        
                s_im, r, done, info = self.env.step(act)
                img = CarConfig.rgb2gray(s_im, True)
                R += r
        
                for i in range(IMAGE_STACK-1):
                    self.s[:,:,i] = self.s[:,:,i+1]
                self.s[:,:,IMAGE_STACK-1] = img
                               
            else:
                act[0] = agent.Imitate.Predict_Angle(s_im)
                s_im, r, done, info = self.env.step(act)  

                img =  CarConfig.rgb2gray(s_im, True)
                for i in range(IMAGE_STACK-1):
                    self.s[:,:,i] = self.s[:,:,i+1]
                self.s[:,:,IMAGE_STACK-1] = img

                R += r
                    
                
            if(self.step % 10) == 0:
                print('Step:', self.step, 'action:',act, 'R: %.1f' % R)
                
            self.step += 1
            
            if done or (R<-5) or (agent.steps > MAX_NB_STEP) or np.mean(s_im[:,:,1]) > 185.1:
                R = 0
                self.step = 0
                print('Done:', done, 'R<-5:', (R<-5), 'Green>185.1:',np.mean(s_im[:,:,1]))
                break
        
        del s_im, img, act, info, r, R
            
    def plot_logs(self, f, ax):

        f.set_figwidth(10)

        #ax1.set_xlim([0,50])
        ax[0].plot(self.reward, label= 'Reward')
        if len(self.reward) > 10:
            mav = np.convolve(self.reward, np.ones((10,))/10, mode='valid')
            ax[0].plot(mav, label= 'MovingAv10')
        ax[0].set_title('Reward')
        ax[0].legend(loc='upper right')
        
        #ax2.set_xlim([0,50])
        ax[1].plot(agent.pTarget, label='TargetValue')
        ax[1].plot(agent.pStateVal, label='StateValue')
        ax[1].set_title('Value functions')
        ax[1].legend(loc='lower right')
        
        ax[2].plot(self.steering, label='Steering angle')
        ax[2].plot(self.throttle, label='Throttle')
        ax[2].plot(self.brake, label='Brake')
        ax[2].set_title('Actions')
        ax[2].legend(loc='lower right')
        
        f.savefig('Reward_Values_Actions')
        ax[0].cla(); ax[1].cla(); ax[2].cla() 
        
        plt.close(f)
        #gc.collect()

           
        
#-------------------- MAIN ----------------------------
if __name__ == "__main__":

    PROBLEM = 'CarRacing-v0'
    env = Environment(PROBLEM)
    
    stateCnt  = IMAGE_SIZE

    actionCnt = env.env.action_space.shape[0] - 1 #Steering angle is is output of Imitation policy
    assert action_buffer.shape[1] == actionCnt, "Lenght of Env action space does not match action buffer"
    
    random.seed(ENV_SEED)
    np.random.seed(1)
    
    agent = Agent(stateCnt, NumberOfDiscActions)
    randomAgent = RandomAgent(NumberOfDiscActions)
    
    try:
        if LoadWeithsAndTest == False:
            #Train agent
            if LoadWeithsAndTrain == False:
                #Start training from scratch
                print("Initialization with random agent. Fill memory")
                while randomAgent.exp < MEMORY_CAPACITY:
                    env.run(randomAgent)
                    print(randomAgent.exp, "/", MEMORY_CAPACITY)
                
                CarConfig.dump_pickle(randomAgent.memory, ModelsPath+'RandomMemory')
                agent.memory = randomAgent.memory
                del randomAgent
                
                agent.memory = CarConfig.load_pickle(ModelsPath+'RandomMemory')
                print("Starts learning")
                
                while env.episode < MAX_NB_EPISODES:
                    env.run(agent)
                
                CarConfig.save_DDQL(ModelsPath, "CarRacing_DDQN_model.h5", agent, env.reward)
                
            else:            
                print('Load pre-trained agent and learn')
                agent.brain.model.load_weights(ModelsPath+"CarRacing_DDQN_model.h5")
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
                    del randomAgent
                
                    agent.maxEpsilone = MAX_EPSILON/5
                
                print("Starts learning")
                
                while env.episode < MAX_NB_EPISODES:
                    env.run(agent)
                
                CarConfig.save_DDQL(ModelsPath, "CarRacing_DDQN_model.h5", agent, env.reward)
        else:
            print('Load agent and play')
            agent.brain.model.load_weights(ModelsPath+"CarRacing_DDQN_model_cp_per_500.h5")
            
            done_ctr = 0
            while done_ctr < 5 :
                env.test(agent)
                done_ctr += 1
                
            env.env.close()
                              
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
