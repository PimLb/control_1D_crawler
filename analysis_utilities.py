#Contains accessory funcitons used for testing, comparing, plotting ecc
# More of a workbook note

import numpy as np


def optimum_impulse(t,omega,N,n_suckers):
    n_pulse = int(n_suckers/N)
    k = 2*np.pi/N
    alpha =  np.arctan(omega/(k*k))
    beta = 3/2*np.pi
    n0 = (omega*t-alpha+beta)/k
    #  print("reference",n0)
    ids=[]
    ns=[]
    for i in range(n_pulse):       
        n = (n0+N*i)%n_suckers
        # print(n,i,N*i)
        ns.append(n)
        ids.append(int(np.floor(n)))
    # id = int(np.floor(n))
    # print(ns,ids)
    # input()
    return ns,ids

def u0(s,t,omega,N,amplitude,l0):
    k = 2*np.pi/N
    alpha =  np.arctan(omega/(k*k))
    A = amplitude/k * np.cos(alpha)
    u = A*np.cos(omega*t - k*s/l0 -alpha)
    return u


def anal_vel_l0norm(N,omega):
    k = 2*np.pi/N
    amplitude_fraction = 1/x0Fraction
    phase_vel = omega/k
    alpha = np.arctan(omega/(k*k))
    # reducedOmega = omega/(k*k)
    # cos_alpha = 1/(np.sqrt(1+reducedOmega*reducedOmega))
    return  amplitude_fraction * phase_vel * np.cos(alpha)#cos_alpha





def actionPolicyCounter(policy,env,mulitagent=False):
    """
    For a given policy, establish on average how many suckers are active weighted on the frequency each state has been visited
    """
    stateFreq = getVisits(policy,env)
    #state Freq: each entry is the normalized residency time over a fairly long sample. Never visited states are discarded. Return also number of active states.






def getVisits(policy,env,hiveUpdate=False):
    """
    By implementing the policy on the environment, it counts each entry of the policy how many times is visited.
    Output : 
    """
    print("CAREFUL: constrained action unsupported")
    #policy is a vector each entry representing a state
    learning_space = env.info["learning space"]
    nsuckers = env.info["n suckers"]
    is_multiAgent = env.info["multiagent"]
    if is_multiAgent:
        nAgents = nsuckers
    else:
        nAgents = env.info["n ganglia"]
    state = env.get_state()
    for k in range(10000):
        action = getAction(state,policy,nsuckers,nAgents,is_multiAgent = is_multiAgent, hiveUpdate=hiveUpdate)
        state,_r,_t=env.step(action)



def getAction(state,policy,nsuckers,nAgents,is_multiAgent,hiveUpdate):
    out_action=[]
    if is_multiAgent:
        if hiveUpdate:
            for k in range(nAgents):
                out_action.append(policy[state[k]])
        else:
            out_action.append(policy[k][state[k]])
    else:
        #GANGLIA (CONTROL CENTER) SCENARIO
        #here we need a specific binary decoding for the actions and encoding for states
        encoded_state = [learning.interpret_binary(s) for s in state]
        if hiveUpdate:
            for k in range(nAgents):
                out_action.append(learning.make_binary(policy[encoded_state[k]]),int(nsuckers/nAgents))
        else:
            for k in range(nAgents):
                out_action.append(learning.make_binary(policy[k][encoded_state[k]]),int(nsuckers/nAgents))
    



    # for n_suckers in ns:
    #  ...:     print()
    #  ...:     env = Environment(n_suckers,sim_shape,t_position, carrierMode = 1,omega =omega,isOverdamped=True)
    #  ...:     env.deltaT = 0.1
    #  ...:     env.equilibrate(1000)
    #  ...:     #state = env.get_state()
    #  ...:     #Q =actionValue((env.state_space,env.action_space),nAgents=env._nagents,total_episodes=episodes,hiveUpdate=True)
    #  ...:     #Q._Q = Q_optimum_finite
    #  ...:     for k in range(10000):
    #  ...:         nss,ids = optimum_impulse(env._t,env.omega,env.N,env._nsuckers)
    #  ...:         #action = Q.get_onPolicy_action(state)
    #  ...:         action = [0] * n_suckers
    #  ...:         for s in ids:
    #  ...:             action[s]=1
    #  ...:         state,reward,terminal=env.step(action)
    #  ...:     print("\n** average vel = ",env.get_averageVel())
    #  ...:     norm_vel = env.get_averageVel()/env.x0
    #  ...:     print(norm_vel)
    #  ...:     vel_semiAnal.append(norm_vel)