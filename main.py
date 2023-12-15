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
steps = 6000
episodes=1000
print("Episodes: ",episodes)
print("steps x episode:", steps)
print()
ns =[3,5,8,10,12,15,20,25,30,35]
vel_RLhive =[]
print('number of suckers analysed:', ns)
for n_suckers in ns:
    print("**********\n")
    print('\n\nLearning for tentacle with '+str(n_suckers)+' suckers\n')
    env = Environment(n_suckers,sim_shape,t_position,omega=0.1,carrierMode=carrierMode,isOverdamped=True)
    Q =actionValue((env.state_space,env.action_space),nAgents=env._nagents,total_episodes=episodes,hiveUpdate=True) 
    state = env.get_state()
    print("lr =",Q.lr)
    print("epsilon =", Q.epsilon)
    for e in trange(episodes):
        for k in range(steps):
            action = Q.get_action(state)
            old_state = state
            state,reward,terminal  = env.step(action)
            Q.update(state,old_state,action,reward)
        Q.makeGreedy() #just for scheduling, not linked to an actual episode
        env.reset_partial() #to avoid abuse of memory DOES NOT reset position
    print("lr =",Q.lr)
    print("epsilon =", Q.epsilon) 
    print(Q.get_value())
    env.reset()
    env.deltaT = 0.1 # for higher precision
    env.equilibrate(1000)
    state=env.get_state()
    for k in range(20000):
        action = Q.get_onPolicy_action(state)
        state,reward,terminal  = env.step(action)
    vel_RLhive.append(env.get_averageVel()/env.x0)
    print("average_vel (normalized)=",vel_RLhive[-1])
np.savetxt("periodic tentacle_HIVE_dt%.1f.txt"%env.deltaT,np.column_stack((np.array(ns),np.round(vel_RLhive,6))),fmt='%d\t%.6f',header="n suck\tnormVel\t\ttentacle length%d"%env.tentacle_length)
print(vel_RLhive)

print("NON HIVE UPDATE")
input("proceed?\n")
steps = 8000
episodes=1000
print("Episodes: ",episodes)
print("steps x episode:", steps)
print()
ns =[3,5,8,10,15,20]
vel_RL_noHive =[]
print('number of suckers analysed:', ns)
for n_suckers in ns:
    print("**********\n")
    print('\n\nLearning for tentacle with '+str(n_suckers)+' suckers\n')
    env = Environment(n_suckers,sim_shape,t_position,omega=0.1,carrierMode=carrierMode,isOverdamped=True)
    Q =actionValue((env.state_space,env.action_space),nAgents=env._nagents,total_episodes=episodes,hiveUpdate=False) 
    state = env.get_state()
    print("lr =",Q.lr)
    print("epsilon =", Q.epsilon)
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
    print("lr =",Q.lr)
    print("epsilon =", Q.epsilon) 
    print(Q.get_value())
    env.reset()
    env.deltaT = 0.1 # for higher precision
    env.equilibrate(1000)
    state=env.get_state()
    for k in range(20000):
        action = Q.get_onPolicy_action(state)
        state,reward,terminal  = env.step(action)
    vel_RL_noHive.append(env.get_averageVel()/env.x0)
    print("average_vel (normalized)=",vel_RL_noHive[-1])

print(vel_RL_noHive)
np.savetxt("periodic tentacle_noHIVEdt%.1f.txt"%env.deltaT,np.column_stack((np.array(ns),np.round(vel_RL_noHive,6))),fmt='%d\t%.6f',header="n suck\tnormVel\t\ttentacle length%d"%env.tentacle_length)