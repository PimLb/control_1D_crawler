from env import Environment
from learning import actionValue
import numpy as np


#TO DO: Equilibrate a bit the system before training..

carrierMode = 1
n_suckers = 8
sim_shape = (30,)
t_position = 29.5



episodes =1000
steps = 500

env = Environment(n_suckers,sim_shape,t_position,carrierMode=carrierMode)
Q =actionValue((env.state_space,env.action_space),nAgents=env._nagents,n_episodes=episodes)
#OBS: action array is identically formulated for both multiagent and single. What changes is the indexing strategy
tstepsTotarget =[]
episode = []
diff =[] #to check convergence
Q_old = Q._Q.copy()
for e in range(episodes):
    state = env.get_state()
    for k in range(steps):
        action = Q.get_action(state)
        old_state = state
        state,reward,terminal  = env.step(action)
        Q.update(state,old_state,action,reward)
        # if (e%500==0):
        #     env.render()
        if terminal:
            # print('touched wall')
            print(Q.get_value())
            break
    if (e%200==0):
        print(e)
        print(Q.get_value())
    diff.append(np.amax(np.abs(Q._Q -Q_old)))
    Q_old = Q._Q.copy()
    Q.makeGreedy()
    tstepsTotarget.append(k)
    episode.append(env._episode)
    env.reset()
# Q.get_policyView()
import matplotlib.pyplot as plt
plt.figure()
fig = plt.subplot(xlabel='episodes', ylabel='time_steps to target')
fig.plot(epoch,tstepsTotarget)
plt.ion()
plt.show()
    # print("episode",env._epoch)
    # Q.makeGreedy() #decrease epsilon


    # rivedi dinamica.. dovrebbe essere instantanea l'evoluzione
    #   a questo punto però c'è una scelta da fare su quando viene evolto ogni agente..
    # update action ogni mezzo periodo? (ma Q ad ogni time step?)
    # reward prop a velocita centro di massa

    #quando pensi funziona, compara con random policy

    #wish list: read papear, learn something, achieve a code that does something, include one goal with a collaboration