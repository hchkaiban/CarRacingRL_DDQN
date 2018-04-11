Double Deep Q Network from Deep Mind's "Playing Atari with Deep Reinforcement Learning" dec 2013 with latest improvements from 2016 (target network periodical update and priorized experience replay).
The algorithm is first validated on pendulum-v0 and then applied to the CarRacing-v0. 

CarRacing-v0: 
Actions: Steering: real valued in [-1, 1] Gas: real valued in [0, 1] Break: real valued in [0, 1] 
Observations: STATE_W = 96 * STATE_H = 96 * 3 : RGB Image 
Reward: +1000/N for every N tile on track else -0,1

### DDQN algorithm first benchmarked on pendulum-v0: 
- Pendulum_RL_DDQN_linearAct.py
![Pendulum-v0_DDQL Reward/Exploration](https://github.com/hchkaiban/CarRacingRL/blob/master/Pendulum_DDQN_Reward.png)
- Pendulum_QL.py: simple QL

### Application of DDQN to CarRacing-v0:
- CarConfig.py: Global configuration 
- CarRacing_RL_DDQN_linearAct.py: main file
- brain_model.png: Keras plot of brain's CNN

![DDQN_Brain_Model](https://github.com/hchkaiban/CarRacingRL/blob/master/brain_model.png)
- SumTree.py: Class for priorized memory replay

### Policy calculation:
As the training time revealed excessive, a simpler approach was adopted: the steering policiy is learned by imitation learning; throttle and braking by reinforcement learning.

RLImitation folder:
- CarRacing_ImitationPolicy.py: the steering policiy by imitation learning (see also https://github.com/hchkaiban/CarRacingImitationLearning)
![Reward_Values_Actions](https://github.com/hchkaiban/CarRacingRL/blob/master/RLImitation/Reward_Values_Actions.png)

Scatter plot of discretized actions function of the steering angle (note that the car accelerates mostly around 0°)
![Actions_Scatter](https://github.com/hchkaiban/CarRacingRL/blob/master/RLImitation/Actions_scatter.png)
- CarRacing_RL_Imitation.py: DDQN applied to throttle and brake 
- Reward_Values_Actions.png: reward in time
- RL_IM_cp500.webm: Video demo 
![Simulation_TrainingAndTestTracks](https://github.com/hchkaiban/CarRacingRL/blob/master/RLImitation/RL_IM_cp500.webm)

# Result:
Three challenging discreet actions were chosen in order to prove that the algorithm learns a meaningful way to synchronize them: strong braking, fast acceleration or free wheeling (no action). As shown by the scatter plot and demo video, it manages to synchronize them with the steering policy: after some training time teh car drives at decent speed, both on training and random tracks, does not immobilize anymore and is able to recover most of the time from off-road situations. Longer training, more data and smoother discrete actions would keep improving the performances. 

Thanks to Jaara for the useful repo: https://github.com/jaara.
