import env
from env import Environment
from learning import actionValue
import numpy as np
from tqdm import trange

carrierMode = 1
sim_shape = (20,)
t_position = 41 
omega =0.1
amplFraction = env.x0Fraction
tentacle_length = 10
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


#Automatic analysis
print("PARAMETERS: ")
print("finite tentacle\nLt=%d\tampliteudeFraction=1/%d\tomega=%.3f"%(tentacle_length,amplFraction,omega))
print("\nMULTIAGENT HIVE UPDATE")
steps = 3000
episodes=1500
print("Episodes: ",episodes)
print("steps x episode:", steps)
print()
ns =[3,5,8,10,12,15,20,25,30,35]
vel_RLhive =[]
print('number of suckers analysed:', ns)
input("continue?")
for n_suckers in ns:
    print("**********\n")
    print('\n\nLearning for tentacle with '+str(n_suckers)+' suckers\n')
    env = Environment(n_suckers,sim_shape,t_position,omega=omega,tentacle_length=tentacle_length,carrierMode=carrierMode,isOverdamped=True)
    Q =actionValue(env.info,total_episodes=episodes,hiveUpdate=True) 
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
np.savetxt("finiteT_multiagent_HIVE_omega%.2f.txt"%env.omega,np.column_stack((np.array(ns),np.round(vel_RLhive,6))),fmt='%d\t%.6f',header="nsuck\tnormVel\t\t#tentacle length=%d"%env.tentacle_length)
print(vel_RLhive)

print("\nNON HIVE UPDATE")
input("proceed?\n")
steps = 5000
episodes=1500
print("Episodes: ",episodes)
print("steps x episode:", steps)
print()
ns =[3,5,8,10,12,15,20,25]
# ns =[15,20,25]
vel_RL_noHive =[]
print('number of suckers analysed:', ns)
for n_suckers in ns:
    print("**********\n")
    print('\n\nLearning for tentacle with '+str(n_suckers)+' suckers\n')
    env = Environment(n_suckers,sim_shape,t_position,omega=omega,tentacle_length=tentacle_length,carrierMode=carrierMode,isOverdamped=True)
    # Q =actionValue((env.state_space,env.action_space),nSuckers=env._nsuckers,total_episodes=episodes,hiveUpdate=False) 
    Q =actionValue(env.info,total_episodes=episodes,hiveUpdate=False) 
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
    # print(Q.get_value())
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
np.savetxt("finiteT_multiagent_noHIVE_omega%.2f.txt"%env.omega,np.column_stack((np.array(ns),np.round(vel_RL_noHive,6))),fmt='%d\t%.6f',header="nsuck\tnormVel\t\ttentacle length%d"%env.tentacle_length)

##################
vel_RL_gangliaMin = []
print("\nGanglia: minimal not hive")
input("proceed?\n")
steps = 20000
episodes=1500
print("Episodes: ",episodes)
print("steps x episode:", steps)
print()
ns =[5,8,10,12,15,20,25]

vel_RL_noHive =[]
print('number of suckers analysed:', ns)
for n_suckers in ns:
    print("**********\n")
    print('\n\nLearning for tentacle with '+str(n_suckers)+' suckers\n')
    if ns==20:
        ng=2
    elif(ns==15 or ns==25):
        ng = int(ns/5)
    else:
        ng =1
    env = Environment(n_suckers,sim_shape,t_position,omega=omega,tentacle_length=tentacle_length,carrierMode=carrierMode,isOverdamped=True,is_multiagent=False,nGanglia=ng)
    # Q =actionValue((env.state_space,env.action_space),nSuckers=env._nsuckers,total_episodes=episodes,hiveUpdate=False) 
    Q =actionValue(env.info,total_episodes=episodes,hiveUpdate=False) 
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
    # print(Q.get_value())
    env.reset()
    env.deltaT = 0.1 # for higher precision
    env.equilibrate(1000)
    state=env.get_state()
    for k in range(20000):
        action = Q.get_onPolicy_action(state)
        state,reward,terminal  = env.step(action)
    vel_RL_gangliaMin.append(env.get_averageVel()/env.x0)
    print("average_vel (normalized)=",vel_RL_gangliaMin[-1])

print(vel_RL_gangliaMin)
np.savetxt("finiteT_Ganglia_noHIVE_omega%.2f.txt"%env.omega,np.column_stack((np.array(ns),np.round(vel_RL_gangliaMin,6))),fmt='%d\t%.6f',header="nsuck\tnormVel\t\ttentacle length%d"%env.tentacle_length)