#Contains accessory funcitons used for testing, comparing, plotting ecc
# More of a workbook note

import numpy as np
import globals
from tqdm import trange

from datetime import datetime


stateMap = {'->|<-':0,'->|->':1,'->|tip':6,'<-|<-':2,'<-|->':3,'<-|tip':7,'base|<-':4,'base|->':5}
stateMap_nonHive = {'->|<-':0,'->|->':1,'->|tip':0,'<-|<-':2,'<-|->':3,'<-|tip':1,'base|<-':0,'base|->':1}
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


def anal_vel_l0norm(N,omega,x0Fraction):
    k = 2*np.pi/N
    amplitude_fraction = 1/x0Fraction
    phase_vel = omega/k
    alpha = np.arctan(omega/(k*k))
    # reducedOmega = omega/(k*k)
    # cos_alpha = 1/(np.sqrt(1+reducedOmega*reducedOmega))
    return  amplitude_fraction * phase_vel * np.cos(alpha)#cos_alpha

###################
########################        POLICY ANALYSIS TOOLS   


def actionMapState_dict(policy,is_ganglia,isHive,n_suckers,nAgents):
    '''
    NOT sure of the interpretation, but it could be a compact number to assign to a policy?
    In principle this is knowable a priori.. A given policy corresponds to a fixed amount of actions for each given state..
    Since I see this as a per tentacle property, I return the overall active suckers per state for the given policy. 
    De facto I'm mapping not hive into hive doing so in terms of action population..
    '''
    internalStates = {'->|<-','<-|->','<-|<-','->|->'}
    actionPerState = {}
    if is_ganglia==False:
        if isHive:
            n_states = len(policy)
            # actionPerState = np.empty(n_states) #each policy in one action, then it should be multiplied by the number of suckers
            for s,a_ind in policy.items():
                if s in internalStates:
                    actionPerState[s] =  (n_suckers-2)*a_ind
                else:
                    actionPerState[s] = a_ind
        else:
            #use same map of hive {'->|<-':0,'->|->':1,'->|tip':6,'<-|<-':2,'<-|->':3,'<-|tip':7,'base|<-':4,'base|->':5}
            # actionPerState = np.zeros(8)
            for pol in policy:
                for s,a_ind in pol.items():
                    if s in actionPerState:
                        actionPerState[s] += a_ind #a_ind is just 1 or 0 for each agent
                    else:
                        actionPerState[s] = a_ind
            # actionPerState = dict(functools.reduce(operator.add,map(collections.Counter, actionPerState)))

    else:
        padding= int(n_suckers/nAgents)
        if isHive:
            n_states = len(policy)
            actionPerState = np.empty(n_states)
            for s,a_ind in policy.items():
                actionPerState[s]= nAgents*sum(globals.make_binary(a_ind,padding))  #=number of anchorings for that policy
        else:
            for pol in policy:
                for s,a_ind in pol.items():
                    if s in actionPerState:
                        actionPerState[s]+= sum(globals.make_binary(a_ind,padding))
                    else:
                        actionPerState[s]= sum(globals.make_binary(a_ind,padding))
            
            # actionPerState = dict(functools.reduce(operator.add,map(collections.Counter, actionPerState)))

    return actionPerState




def getPolicyStats(Q,env,nLastPolicies = 100,runtimeInfo=None):
    """
    Useful if some oscillation present on the last segment (pseudo_plateau) of the triaining. Can gather stats on the different policies the Q matrix jumps in.. 
    """
    
    # I  expect runtimeInfo contains also info on number of steps and episodes with eventual number of convergence cycles

    if Q._ganglia==False:
        if Q._parallelUpdate:
            type="MULTIAGENT_HIVE"
        else:
            type="MULTIAGENT"
    else:
        nAgents = Q._nAgents
        if Q._parallelUpdate:
            type = "%dGANGLIA_HIVE"%nAgents
        else:
            type = "%dGANGLIA"%nAgents
            
    print("out file name:")
    fileName = "Raw_policyMeasuresFor%dsuckers_omega%.2f_%s"%(Q._nsuckers,env.omega,type)
        

    print(fileName)
    input()

    #first establish baseline of the random policy (or the null one)
    # print("Random Policy ANalysis")
    state_freqRandom,visitedRandom = Q.evaluateTrivialPolicy(env)
    # print("visited states:",visitedStates)
    #Loop trough last n learned policies to gather stats

    #performance measure
    polIndx = []
    norm_vels = []
    visitedStates = []
    #policy characterization measures
    relative_state_freqs =[]
    activeS =[] #normalized by total number of suckers in the tentacle
    value =[] #value associated to the policy
    c=0
    print("Gathering properties of last %d policies.."%nLastPolicies)
    for pol_n in trange(1,nLastPolicies+1):
        polIndx.append(c)
        value.append(Q.set_referencePolicy(pol_n))
        # print("")
        vel,state_freq,norm_activeSuckers,visited = Q.evaluatePolicy(env)
        norm_vels.append(vel)
        visitedStates.append(len(visited))
        relative_state_freqs.append({s:(state_freq[s]-state_freqRandom[s])/state_freqRandom[s]*100 for s in visitedRandom})
        activeS.append(norm_activeSuckers)
        c+=1
    norm_vels = np.array(norm_vels)
    max_vel= np.amax(norm_vels)
    bestPolIndx = np.argmax(norm_vels) #Returns first occurrence
    average_normVel = np.average(norm_vels)
    std_normVel = np.std(norm_vels)
    
    outFileName = "SUMMARY_policyMeasuresFor%dsuckers_omega%.2f_%s"%(Q._nsuckers,env.omega,type)
    outFile = open(outFileName,'w')

    line = '***** SUMMARY ****\nn suckers = %d, T length = %d, omega = %.2f, x0= %.3f Type: %s'%(Q._nsuckers,env.tentacle_length,env.omega,env.x0,type)
    outFile.write(line)
    line='\nNumber of policies analyzed : %d'%nLastPolicies
    outFile.write(line)
    line='\n\nNORMALIZED VEL AVERAGE= %.4f +- %.4f'%(np.round(average_normVel,4),np.round(std_normVel,4))
    outFile.write(line)
    line='\nNORMALIZED VEL MAX= %.4f\t Correspondent policy index: %d'%(np.round(max_vel,4),bestPolIndx)
    outFile.write(line)
    if runtimeInfo is not None:
        line = "\n\nRUNTIME: "+runtimeInfo
        outFile.write(line)


    activeS = np.array(activeS)
    visitedStates = np.array(visitedStates)
    value = np.array(value)
    # print(value)
    if value.ndim>1:
        value = np.sum(value,axis=1) #total value for all agents
    #INFO:x0=%.3f, tLengthavailable states/all possible:%d/%d\n %(visitedStates,Q.state_space_dim
    header = "x0=%.3f. Visited states under random_policy:%d/%d\nindx\tnorm vel\tactiveSts[%%]\tvisitedStates\tTotal av_value"%(env.x0,len(visitedRandom),Q.state_space_dim)
    now = datetime.now().ctime()
    if runtimeInfo is not None:
        footer = "runtime for training: "+runtimeInfo+"\nCurrent time: "+now
    footer = "Current time: "+now
    np.savetxt(fileName,np.column_stack((np.array(polIndx),np.round(norm_vels,6),np.round(activeS,3)*100,visitedStates,np.round(value,3))),fmt=' %d\t%.6f\t\t%.1f\t%d\t\t%.3f',header=header,footer=footer)

    Q._refPolicy = Q.getPolicy()


    return bestPolIndx




################
# TOOLS FOR ANALYSIS OF TIME AND SPACE (AUTO)CORRELATION OF THE ACTION MATRIX

def timeCorrelation(A,sucker_index):
    #All averages are time average, therefore along columns (horizontal direction matrix)
    average = np.average(A[sucker_index])
    print(average)
    time_steps = A.shape[1]
    print(time_steps)
    #
    C = np.empty(time_steps)
    for t1 in range(time_steps):
        for t in range(time_steps-t1):
            C[t1] += A[sucker_index,t+t1] * A[sucker_index,t]
    C = C/time_steps/(average*average)
    return C 

def twoSuckerCorrelation(A,sucker_index):
    n_suckers = A.shape[0]
    time_steps = A.shape[0]
    C = np.empty(n_suckers)
    average = np.average(A[sucker_index]) #average over all suckers.. Is that correct?
    # print(average)
    # suckers = set(np.arange(n_suckers))
    # suckers.remove(sucker_index) #all other suckers
    # print(suckers)
    for i1 in range(n_suckers):
        C[i1] = np.average(A[i1,:] * A[sucker_index,:])
    # print(C)
    C = C/(average*average)
    # print(C)
    return C



# def extractResults(infile):
#     '''
#     Read recap file, returns index with best policy, average and standard deviation. 
#     This routine could be coupled with the extraction of the relevant Action Matrix (time serie of the action performed by a policy), and relevant analysis of it.
#     '''

#     out_file = open('SUMMARY_'+ infile +'.out','w')
    




##################
#def actionMapState(policy,is_multiAgent,isHive,n_suckers,nAgents):
#     '''
#     NOT sure of the interpretation, but it could be a compact number to assign to a policy?
#     In principle this is knowable a priori.. A given policy corresponds to a fixed amount of actions for each given state..
#     Since I see this as a per tentacle property, I return the overall active suckers per state for the given policy. 
#     De facto I'm mapping not hive into hive doing so in terms of action population..
#     '''
#     #for not hive give one actionPerState vector for each agent!
#     if is_multiAgent:
#         if isHive:
#             n_states = len(policy)
#             actionPerState = np.empty(n_states) #each policy represents one action, then EXCLUDING TIP AND  BASE it should be multiplied by the number of suckers
#             for s,a_ind in enumerate(policy[0:4]):
#                 actionPerState[s] =  (n_suckers-2)*a_ind
#             for s,a_ind in enumerate(policy[4:]):
#                 actionPerState[s] =  a_ind
#         else:
#             #use same map of hive {'->|<-':0,'->|->':1,'->|tip':6,'<-|<-':2,'<-|->':3,'<-|tip':7,'base|<-':4,'base|->':5}
#             actionPerState = np.zeros(8)
#             for pol in policy[1:n_suckers-1]:
#                 n_states = len(pol)
#                 aPs = np.empty(n_states)
#                 for s,a_ind in enumerate(pol):
#                     aPs[s] =  a_ind
#                 actionPerState[0:4] += aPs
            
#             aPs = np.empty(2)
#             for s,a_ind in enumerate(policy[0]):
#                 aPs[s] =  a_ind
#             actionPerState[4] = aPs[0]
#             actionPerState[5] = aPs[1]
#             for s,a_ind in enumerate(policy[n_suckers-1]):
#                 aPs[s] =  a_ind
#             actionPerState[6] = aPs[0]
#             actionPerState[7] = aPs[1]
#     else:
#         padding= int(n_suckers/nAgents)
#         if isHive:
#             n_states = len(policy)
#             actionPerState = np.empty(n_states)
#             for s,a_ind in enumerate(policy):
#                 actionPerState[s]= nAgents*sum(globals.make_binary(a_ind,padding))
#         else:
#             n_states = len(policy[0])
#             actionPerState = np.zeros(n_states)
#             for pol in policy:
#                 aPs = np.empty(n_states)
#                 for s,a_ind in enumerate(pol):
#                     aPs[s]= sum(globals.make_binary(a_ind,padding))
#                 actionPerState+=aPs

#     return actionPerState

# def convertPolicy(policy,env,hiveUpdate,nGanglia_out =1):
#     """
#     Converts a multi agent policy into a CC one. Being multiagent policy always a subset of CC ones.

#     """
#     # learning_space = env.info["learning space"]
#     # n_states = learning_space[0]
#     # n_actions = learning_space[1]
    
#     nsuckers = env.info["n suckers"]
#     is_multiAgent = env.info["multiagent"]
#     action_on = 2**nsuckers #all on
#     if is_multiAgent ==False:
#         print("nothing to do")
#         return
#     #First parse policies in base, tip and internal ones

#     ganglia_policy = np.empty(nsuckers-1) #all combinations of the above
#     if hiveUpdate ==True:
#         #not interested in base and tip, all info from internal
#         # base_pol = policy[4,5]  #compressed, elongated-->first position is compressed =0,second posiiton is 1
#         # tip_pol = policy[5,6]   #compressed, elongated
#         internal_suck_pol = policy[0:4] #onl
#     else:
#         #no hive
#         base_pol = policy[0]
#         # tip_pol = policy[-1]
#         internal_suck_pol = policy[1:nsuckers-1]
#         # for i in range(nsuckers-1):
        #     spring_


    #COMPLICATED for hive
    #e.g state ('->|<-') [comp,comp] = (...,0,1,..) all shifts of this spring states and for each pos action corresponding according to given pol
   
   #not hive
   






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