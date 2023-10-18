from env import Environment
from learning import actionValue
import numpy as np


#TODO: Equilibrate a bit the system before training.. <--

carrierMode = 1
n_suckers = 10
sim_shape = (40,)
t_position = 41 


#EPISODIC LEARNING

# episodes =1000
# steps = 5000

# env = Environment(n_suckers,sim_shape,t_position,carrierMode=carrierMode)
# Q =actionValue((env.state_space,env.action_space),nAgents=env._nagents,total_episodes=episodes)
# #OBS: action array is identically formulated for both multiagent and single. What changes is the indexing strategy

# for e in range(episodes):
#     state = env.get_state()
#     if (e%20==0):
#         print(e)
#         print("convergence =", Q.get_conv())
#         print("lr =",Q.lr)
#         print("epsilon =", Q.epsilon)
#     for k in range(steps):
#         action = Q.get_action(state)
#         old_state = state
#         state,reward,terminal  = env.step(action)
#         Q.update(state,old_state,action,reward)
#         # if (e%500==0):
#         #     env.render()
#         if terminal:
#             break
#     Q.makeGreedy()
#     env.reset(exploringStarts=True)

# #############
# #CONTINUOUS LEARNING

# episodes =1000
# steps = 2000
# sim_shape = (40,)
# t_position = 41 #target out of simulation box 
# env = Environment(n_suckers,sim_shape,t_position,carrierMode=carrierMode,omega=0.1)
# Q =actionValue((env.state_space,env.action_space),nAgents=env._nagents,total_episodes=episodes,hiveUpdate=True)
# #OBS: action array is identically formulated for both multiagent and single. What changes is the indexing strategy
# state = env.get_state()
# for e in range(episodes):
#     if (e%20==0):
#         print(e)
#         print("convergence =", Q.get_conv())
#         print("lr =",Q.lr)
#         print("epsilon =", Q.epsilon)
#     for k in range(steps):
#         action = Q.get_action(state)
#         old_state = state
#         state,reward,terminal  = env.step(action)
#         Q.update(state,old_state,action,reward)
#     Q.makeGreedy() #just for scheduling, not linked to an actual episode
#     env.reset_partial() #to avoid abuse of memory DOES NOT reset position 

from tqdm import trange
#Automatic analysis
steps = 2000
episodes=1000
ns =[5,10,15,20,25,30,35]
vel_RLhive =[]
print('number of suckers analysed:', ns)
for n_suckers in ns:
    print('learning for tentacle with '+str(n_suckers)+' suckers\n')
    env = Environment(n_suckers,sim_shape,t_position,omega=0.1,carrierMode=carrierMode,isOverdamped=True)
    env.equilibrate(1000)
    Q =actionValue((env.state_space,env.action_space),nAgents=env._nagents,total_episodes=episodes,hiveUpdate=True) 
    state = env.get_state()
    for e in trange(episodes):
        # if (e%20==0):
        #     print(e)
        #     print("convergence =", Q.get_conv())
        #     print("lr =",Q.lr)
        #     print("epsilon =", Q.epsilon)
        for k in range(steps):
            action = Q.get_action(state)
            old_state = state
            state,reward,terminal  = env.step(action)
            Q.update(state,old_state,action,reward)
        Q.makeGreedy() #just for scheduling, not linked to an actual episode
        env.reset_partial() #to avoid abuse of memory DOES NOT reset position 
    vel_RLhive.append(env.get_averageVel())

print(vel_RLhive())