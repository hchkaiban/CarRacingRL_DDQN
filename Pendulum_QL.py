import numpy as np
import gym
import CarConfig
from keras.layers import Dense, Flatten, Input, Dropout
from keras.models import Model
import keras.optimizers as Kopt
import logging
import matplotlib.pyplot as plt
import random, math


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ModelsPath = CarConfig.ModelsPath
LoadWeithsAndTest = True
TrainEnvRender = False

LEARNING_RATE = 0.0001
HUBER_LOSS_DELTA = 1.0
num_episodes = 100000
MAX_REWARD = 2000

MAX_EPSILON = 1
MIN_EPSILON = 0.1       
EXPLORATION_STOP = 1000   # at this step epsilon will be MIN_EPSILON
LAMBDA = - math.log(0.01) / EXPLORATION_STOP  # speed of decay fn of episodes of learning agent

SARSA = False
alpha = 0.98
gamma = 0.1
    
#action_buffer = np.arange(-2, 2.4, 0.4)
action_buffer = np.array([-2,2])

NumberOfDiscActions = len(action_buffer)    
    
def SelectAction(Act):
    return action_buffer[Act]


def SelectArgAction(Act):
    for i in range(NumberOfDiscActions):
        if np.all(Act == action_buffer[i]):
            return i
    raise ValueError('SelectArgAction: Act not in action_buffer')


def huber_loss(y_true, y_pred):
    from keras import backend as K
    err = y_true - y_pred

    cond = K.abs(err) <= HUBER_LOSS_DELTA
    if cond == True:
        loss = 0.5 * K.square(err)
        
    else:
        loss = 0.5 * HUBER_LOSS_DELTA**2 + HUBER_LOSS_DELTA * (K.abs(err) - HUBER_LOSS_DELTA)

    return K.mean(loss)


def build_model(input_dim, output_dim):
    
    action_input = Input(shape=(1,)+input_dim)
    x = Flatten()(action_input)
    x = Dense(32, activation="tanh")(x)
    x = Dense(32, activation="tanh")(x)
    #x = Dropout(0.1, seed=2)(x)
    x = Dense(16, activation="tanh")(x)
    x = Dense(output_dim, activation="tanh")(x)


    model = Model(inputs=action_input, outputs=x)
    
    return model


def plot_logs(f, ax1, losses):
    
    f.set_figwidth(10)
    ax1.set_title('Reward')
    ax1.plot(losses, label= 'Reward')
    if len(losses) > 10:
        mav = np.convolve(losses, np.ones((100,))/100, mode='valid')
        ax1.plot(mav, label= 'MovingAv10')
    
    ax1.legend(loc='lower right')
    f.savefig('Pendulum_QL_Reward.png')
    ax1.cla()
    plt.close(f)

    
def learn(env):
    Gs = []
    steps = 0
    best_score = -200
    
    for episode in range(num_episodes):
        x = env.reset()
        X, Q, A, R = [], [], [], []
        done = False
        
        if not LoadWeithsAndTest : 
            rand = 0
            while not done:
               
                if TrainEnvRender :
                    env.render()
                epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * steps)
                
                if random.random() < epsilon:
                    #return random.randint(0, self.actionCnt-1)
                    a = np.random.randint(NumberOfDiscActions)
                    q = np.zeros([NumberOfDiscActions])
                    rand += 1
    
                else:
                    s = x[np.newaxis, :]
                    ss = s[:,np.newaxis,:]
                    q = model.predict(ss)[0]
                    a = np.argmax(q)
                
                X.append(x)
                A.append(a)
                Q.append(q)
                
                x, r, done, info = env.step(SelectAction([a]))
                R.append(r)
    
            
            T = len(X)
            G = np.sum(R)
            for t in range(T):
                a = A[t]
                
                if t == T-1 :
                    Q[t][a] = (1-alpha) * Q[t][a] + alpha * R[t]
                else:
                    if SARSA:
                        qv = Q[t+1][A[t+1]] 
                    else:
                        # Q-learning
                        qv = np.max(Q[t+1][:])
            
                    #Q[t][a] = (1-alpha) * Q[t][a] + alpha * (G/MAX_REWARD)
                    Q[t][a] = (1-alpha) * Q[t][a] + alpha * ( (R[t+1]) + gamma* qv)
                
            if G > best_score : 
                  best_score = G
                  model.save(ModelsPath+"Pendulum_QL_model_best.h5")
                  
            obs = np.asarray(X)
            obs = obs[:,np.newaxis,:]
            model.fit(obs, np.asarray(Q), verbose=0, batch_size=T)
            #logger.debug("Episode: %d/%d, Reward: %.2f" % (episode, num_episodes, G)) 
            Gs.append(G)
            
            steps += 1
            if steps % 500 == 0 :
                if len(A)>0 and T>0 :
                    logger.debug("Mean act: %.2f, Mean Rand %.2f" % (np.mean(A), rand/T)) 
                plot_logs(f, ax, Gs)
        else:
            model.load_weights(ModelsPath+"Pendulum_QL_model_best.h5")
            
            done_ctr = 0
            R = []
            while done_ctr < 5 :
                x = env.reset()
                while not done:
                    env.render()
                    s = x[np.newaxis, :]
                    s = s[:,np.newaxis,:]
                    q = model.predict(s)[0]
                    a = np.argmax(q)
                    x, r, done, info = env.step(SelectAction([a]))
                    R.append(r)
                    
                done_ctr += 1
                
    env.env.close()
    return Gs

if __name__ == "__main__":
    
    env = gym.make("Pendulum-v0")
    
    logger.info("Action Space: %s" % str(env.action_space))
    logger.info("Observation Space: %s" % str(env.observation_space))
    
    #env.action_space.shape[0]
    model = build_model(input_dim = env.observation_space.shape, output_dim = NumberOfDiscActions)
    
    opt = Kopt.RMSprop(lr=LEARNING_RATE)
    model.compile(loss = huber_loss, optimizer = opt)#, metrics=["mae", "mse"])
    
    f = plt.figure(0)
    ax = f.gca()
    
    Gs = learn(env)
        
    model.save(ModelsPath+"Pendulum_QL_model.h5")    
    
    logger.info("Average Reward: %.3f" % np.mean(Gs))
    env.close()


