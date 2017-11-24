# CarRacingRL
Double Deep Q learning from Deep Mind's "Playing Atari with Deep Reinforcement Learning" dec 2013. 

CarRacing-v0:
Actions: Steering: real valued in [-1, 1] Gas: real valued in [0, 1] Break: real valued in [0, 1]
Observations: STATE_W = 96 * STATE_H = 96 * 3 : RGB Image
Reward: +1000/N for every N tile on track else -0,1

DDQN benchmarked on pendulum-v0: 
- Pendulum_RL_DDQN_linearAct.py
- Pendulum_QL.py: simple QL
- Pendulum_DDQN_Reward.png: Learning/ Exploration DDQN curve

Application of DDQN to CarRacing-v0:
- CarConfig.py: Global configuration 
- CarRacing_RL_DDQN_linearAct.py: main file
- brain_model.png: Keras plot of brain's CNN
- SumTree.py: Class for priorized memory replay

As the training time revealed excessive, a simpler approach was adopted:

RLImitation folder:
- CarRacing_ImitationPolicy.py: the steering policiy is learned by imitation learning (see also https://github.com/hchkaiban/CarRacingImitationLearning)
- CarRacing_RL_Imitation.py: DDQN applied to throttle and brake 
- Reward_Values_Actions.png: reward in time
- RL_IM_cp500.webm: Demo 

# Result:

The car drives smoothly, at decent speed, both on training and of random tracks. It does not immobilize anymore and is able to recover most of the time from off-road situations. Longer training and more data would keep improving the performances. 

Thanks to Jaara for the usefull repo: https://github.com/jaara.
