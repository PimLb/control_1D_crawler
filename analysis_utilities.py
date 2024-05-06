#Contains accessory funcitons used for testing, comparing, plotting ecc
# More of a workbook note

# import numpy as np
import globals
from tqdm import trange
import numpy as np
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
# def movieFailure(env,policy,isHive,epsilon,failingSuckers=0,epsilonGreedyFail=False):
#     import random
def policyImporter(folder):
    import re
    import glob
    filenames = glob.glob(folder+"*.npy")
    print(filenames)
    input()
    policies = []
    for filename in filenames:
        print(filename)
        match_Ganglia = re.search("(\d)(GANGLIA)",filename)  
        match_Hive = re.search("HIVE",filename)  
        if match_Hive:
            print("hive")
            isHive=True
        else:
            print("not hive")
            isHive=False
        if match_Ganglia:
            nGanglia = int(match_Ganglia.group(1))
            print(nGanglia,"Ganglia")
        else:
            print("multiagent")
            nGanglia = 0
        policy = np.load(filename,allow_pickle=True)
        pol={"policy":policy,"ganglia":nGanglia,"hive":isHive}
        # print(nGanglia,isHive)
        policies.append(pol)
    
    return policies

def policyRobustnessStudy(policies,suckerCentric=True,plot=True,n_suckers=12):
    from env import Environment
    sim_shape = (20,)
    t_position = 100
    results=[]
    if plot ==True :
        import matplotlib.pyplot as plt
        plt.figure()
        plt.ion()
        if suckerCentric:
            epsilon = 1 #100% failure probability of n failing suckers randomly chosen
            title = "Random sucker failure\nRandom choice prob = %d %%"%(epsilon*100)
            xlabel = "# failing suckers" #but not always the same, is always randomly chosen
        else:
            title = "Agent epsilon failure"
            xlabel = "epsilon"
        fig = plt.subplot(xlabel=xlabel, ylabel='velocity (normalized)')
        fig.set_title(label=title)
    
    for pol in policies:
        vels =[]
        policy = pol["policy"]
        n_ganglia = pol["ganglia"]
        isHive= pol["hive"]
        if n_ganglia>0:
            ganglia=True
            if isHive:
                architecture = "%d Ganglia HIVE"%n_ganglia
            else:
                architecture = "%d Ganglia"%n_ganglia
        else:
            ganglia=False
            if isHive:
                architecture = "Multiagent HIVE"
            else:
                architecture = "Multiagent"
        print("\n\n** %s **\n"%architecture)
        env = Environment(n_suckers,sim_shape,t_position,omega =0.1,isOverdamped=True,is_Ganglia=ganglia,nGanglia=n_ganglia)
    #A. SUCKER CENTRIC: Epsilon 100% robustness with respect to n suckers
        if suckerCentric:
            failing_suckers = []
            for fs in range(env._nsuckers):
                vel = robustnessAnal(env,policy,isHive,epsilon,failingSuckers=fs,epsilonGreedyFail=False)
                vels.append(vel)
                failing_suckers.append(fs+1)
            
            out = (failing_suckers,vels)
        #B. AGENT CENTRIC: Robustness with respect to increasing epsilon
        else:
            epsilons = np.linspace(0,1,10)
            for epsilon in epsilons:
                vel = robustnessAnal(env,policy,isHive,epsilon,epsilonGreedyFail=True)
                vels.append(vel)
            out = (epsilons,vels)
        results.append(out)
        if plot==True:
            fig.plot(out[0],out[1],'-o',lw=5,label = architecture)
            fig.legend()
    input()
    return results
    

def robustnessAnal(env,policy,isHive,epsilon,failingSuckers=0,epsilonGreedyFail=False,doMovie = False):
    '''
    Returns a plot/data on decay of velocity as a function of the #suckers failing when playing the given policy.
    Usage: loop externally to extract correspondent velocity. 
    2 MODES: a) AGENT CENTRIC: All agent do a random action with epsilon probability (identical to playing a epsilon greedy policy).
                1 parameter: epsilon= prob of random action.
             b) SUCKER CENTRIC (more comparable among architectures): n suckers at random fail with given probability. 
                2 parameters: n failing suckers, epsilon= prob of taking a random action
    INPUT: failingSucker = only for epsilonGreedy false: how many suckers fail
           epsilon = probability of failure (valid for both modes)
    OBS.: If in the sucker centric I give prob of taking the OPPOSITE action the lower boundary (all suckers fail) will be a negative velocity. 
        I want rather that the lower boundary is the random policy, as in the epsilon greedy mode
    '''
    #Planning to use it on 12s. Might be interesting to consider 20s-->save best policies!!

    #TODO
    #Select n suckers failing with prob 1 EXCLUDING BASE AND TIP (too influencial I think)
    #Randomly shuffle which sucker is failing
    #Failing = do the opposite of the prescription
    #Second modality of failure: epsilonGreedy fashon: 
    #   PROBLEM: more exxtractions for multi vs ganglia--> difficult to compare fairly

    #Consider creating a dedicate plot function which also loops this funciton for different # sucker failures
    # and also makes a small video
    import random



    steps = 20000

    nsuckers = env._nsuckers
    info = env.info
    isGanglia = info["isGanglia"]
    if epsilonGreedyFail==False:
        print("<INFO> : Sucker centric perturbation analysis")
        print("n faling suckers = ",failingSuckers)
        print("Probability of failure = %d%%"%(epsilon*100))
    else:
        print("<INFO> : Agent centric (epsilon greedy) perturbation analysis")
        print("Probability of random action = %d%%"%(epsilon*100))
    # if epsilonGreedyFail==False and failingSuckers ==0:
    #     print("<WARNING> : no failing suckers.. ")
    #     input("continue?")
    
    if isGanglia:
        nGanglia = info["n ganglia"]
        nAgents = nGanglia
        if nAgents ==1:
         #the way in which policies are saved consider the format of 1 ganglion as a single element array of policies
            isHive = False
        if isHive:
            ag,gindxs  = ([0]*nGanglia,range(nGanglia))
        else:
            ag,gindxs  = (range(nGanglia),range(nGanglia))
    else:
        #MULTIAGENT
        if isHive:
            nAgents = 1
            ag,sindxs  = ([0]*env._nsuckers,range(env._nsuckers))
        else:
            nAgents = nsuckers
            ag,sindxs  = (range(nAgents),range(env._nsuckers))
    if isHive:
        policy = np.array([policy.item()])#just for the way they were saved, and to allow zip loop
    

    env.reset(equilibrate = True)
    state = env.get_state()
    # print(state)
    # ----------------- GANGLIA SCENARIO ---------
    if env.isGanglia:
        padding= int(nsuckers/nAgents)
        print("Ganglia")
        print("n Gagnlia = ",nGanglia)
        print("IS HIVE =",isHive)

        # nRandom =0
        for t in trange(steps): 
            encoded_state = [globals.interpret_binary(s) for s in state]
            action = []
            # if isHive:
                # for k in range(env._nsuckers):
                #     #get on-policy action
                #     action.append(policy[encoded_state[k]])
            # else:
            for a,gind in zip(ag,gindxs):#agent,ganglion index
                if epsilonGreedyFail:
                    if np.random.random() < (1 - epsilon):
                        # randomChoice = 0
                        action.append(globals.make_binary(policy[a][encoded_state[gind]],padding))
                    else:
                        # randomChoice=1
                        action.append(globals.make_binary(np.random.randint(0,env.action_space),padding))
                else:
                    action.append(globals.make_binary(policy[a][encoded_state[gind]],padding))
            # nRandom+=randomChoice
            
            action_flattened = [a for al in action for a in al] #or list of list with first index on agent (ganglia)
            if not epsilonGreedyFail:
                #Generalize for any number of failing suckers
                # suckers = list(np.arange(1,nsuckers-1)) #EXCLUDE BASE AND TIP
                suckers = list(np.arange(0,nsuckers))
                for i in range(failingSuckers):
                    devil = random.choice(suckers)
                    # print(devil)
                    if np.random.random() < (1 - epsilon):
                        pass
                    else:
                        # action_flattened[devil] = abs(action_flattened[devil]-1) #does exaclty the contrary of what prescibed
                        action_flattened[devil] =np.random.randint(0,2)
                    suckers.remove(devil)
                # input()
            state,r,_t=env._stepOverdamped(action_flattened)

            if doMovie and t%10==0:
                env.render()

        # print("check random choice=",nRandom)
        # ----------- MULTIAGENT SCENARIO ------------
                
    else:
        print("Multiagent")
        print("IS HIVE =",isHive)
        for t in trange(steps): 
            action = []
            for a,sid in zip(ag,sindxs): #agent,sucker index
                #get on policy action
                if epsilonGreedyFail:
                    if np.random.random() < (1 - epsilon):
                        action.append(policy[a][state[sid]])
                    else:
                        action.append(np.random.randint(0,env.action_space)) #OBS: in this case there's a 50% probability to do the right choice..
                else:
                    action.append(policy[a][state[sid]])
            if not epsilonGreedyFail:
                #Generalize for any number of failing suckers
                # suckers = list(np.arange(1,nsuckers-1)) #EXCLUDE BASE AND TIP
                suckers = list(np.arange(0,nsuckers))
                for i in range(failingSuckers):
                    devil = random.choice(suckers)
                    # print(devil)
                    if np.random.random() < (1 - epsilon):
                        pass
                    else:
                        # action[devil] = abs(action[devil]-1) #opposite action
                        action[devil] =np.random.randint(0,2)
                    suckers.remove(devil) #pick another sucker to fail

                    #OBS can achieve the same with random.sample(suckers,failingSuckers) but cannot re-extract for every sucker if epsilon different from 1
             
            state,r,_t=env.step(action)
            if doMovie and t%10==0:
                env.render()

   
    print("(Norm) Velocity=",env.get_averageVel()/env.x0)

    return env.get_averageVel()/env.x0


def onPolicyStateActionVisit(env,policy,isHive):
    '''
    Returns for each sucker the frequency each multiagent state is played.
    In mind I have that I can do 4 color plots over the tentacle for the 4 different internal states (while always same for base and tip)
    We can use as bottom line the hive policy
    Return also average active suckers of analyzed policy
    '''
    # internalStates = {'->|<-','<-|->','<-|<-','->|->'}
    # multiState = env._get_state_multiagent()
    
    #   PLAY THE POLICY
    #looop the following over integration steps 
    
    # suckerActFreq= {'->|<-': 0 ,'->|->':0,'->|tip':0,'<-|<-':0,'<-|->':0,'<-|tip':0,'base|<-':0,'base|->':0}

    steps = 20000
    actionFreqPerSucker = []
    stateFreqPerSucker = []
    n_activeSuckers = 0
    nsuckers = env._nsuckers
    info = env.info
    isGanglia = info["isGanglia"]
    
    if isGanglia:
        nGanglia = info["n ganglia"]
        nAgents = nGanglia
        if nAgents ==1:
         #the way in which policies are saved consider the format of 1 ganglion as a single element array of policies
            isHive = False
        if isHive:
            ag,gindxs  = ([0]*nGanglia,range(nGanglia))
        else:
            ag,gindxs  = (range(nGanglia),range(nGanglia))
    else:
        #MULTIAGENT
        if isHive:
            nAgents = 1
            ag,sindxs  = ([0]*env._nsuckers,range(env._nsuckers))
        else:
            nAgents = nsuckers
            ag,sindxs  = (range(nAgents),range(env._nsuckers))
    if isHive:
        policy = np.array([policy.item()])#just for the way they were saved, and to allow zip loop
        

    for n in range(env._nsuckers):
        actionFreqPerSucker.append({'->|<-': 0 ,'->|->':0,'->|tip':0,'<-|<-':0,'<-|->':0,'<-|tip':0,'base|<-':0,'base|->':0})
        stateFreqPerSucker.append({'->|<-': 0 ,'->|->':0,'->|tip':0,'<-|<-':0,'<-|->':0,'<-|tip':0,'base|<-':0,'base|->':0})
    #PLAY THE ACTION
    env.equilibrate(1000)
    state = env.get_state()
    # print(state)
    # ----------------- GANGLIA SCENARIO ---------
    if env.isGanglia:
        padding= int(nsuckers/nAgents)
        print("Ganglia")
        print("n Gagnlia = ",nGanglia)
        print("IS HIVE =",isHive)
        for t in trange(steps): 
            encoded_state = [globals.interpret_binary(s) for s in state]
            action = []
            # if isHive:
                # for k in range(env._nsuckers):
                #     #get on-policy action
                #     action.append(policy[encoded_state[k]])
            # else:
            for a,gind in zip(ag,gindxs):#agent,ganglion index
                action.append(globals.make_binary(policy[a][encoded_state[gind]],padding))
            action_flattened = [a for al in action for a in al]
            n_activeSuckers += sum(action_flattened)
            multiState = env._get_state_multiagent() #getting states in term of sucker rather than spring
        #UPDATE FREQ
            for indx,s in enumerate(multiState):
                stateFreqPerSucker[indx][s] +=1
                actionFreqPerSucker[indx][s] += action_flattened[indx]
            state,r,_t=env.step(action)
        # ----------- MULTIAGENT SCENARIO ------------
                
    else:
        print("Multiagent")
        print("IS HIVE =",isHive)
        for t in trange(steps): 
            action = []
            for a,sid in zip(ag,sindxs): #agent,sucker index
                #get on policy action
                action.append(policy[a][state[sid]])
            n_activeSuckers += sum(action)
            for indx,s in enumerate(state):
                stateFreqPerSucker[indx][s] +=1
                actionFreqPerSucker[indx][s] += action[indx]
            state,r,_t=env.step(action)

    averageActiveSuckers = n_activeSuckers/(t+1)/nsuckers
    #FINALIZE STATS
    #normalization
    print("Analysis over\nAVERAGE ACTIVE SUCKERS:", averageActiveSuckers)
    print("Velocity analyzed policy:",env.get_averageVel()/env.x0)
    # print(stateFreqPerSucker)
    # print(actionFreqPerSucker)
    for sF,aF in zip(stateFreqPerSucker,actionFreqPerSucker):
        sF.update((key, val/(t+1)) for key, val in sF.items())  
        aF.update((key, val/(t+1)) for key, val in aF.items())  
    normActFreq = []
    for sF,aF in zip(stateFreqPerSucker,actionFreqPerSucker):
        normActFreq.append({k: (aF[k]/sF[k] if aF[k]!=0 else 0) for k in aF.keys() })
    #SAVE and also return

    return stateFreqPerSucker,actionFreqPerSucker,normActFreq
    

def plotTSvisits(actionFreq,refActionfreq=None):
    import matplotlib.pyplot as plt
    """
    If ref is given, the color plot is normalized by the reference (standard choice would be to use the standard hive policy..)
    """
    intermediateKeys = set(['->|<-','->|->','<-|<-','<-|->'])
    baseKeys = ['base|<-','base|->']
    tipKeys = ['->|tip','<-|tip']
    #colorplot y-axis 4 states x-axis sucker position 
    
    #Read plot tile from key and value from item. 
    #First I have to gather per key all suckers
    #Keep it general for easier adaptation (be agnostic about key names..)
    nsuckers = len(actionFreq)
    keys = set(actionFreq[0].keys())
    tentacleState = {}
    if refActionfreq is not None:
        print("NORMALIZING WITH REFERENCE..")
        for k in keys:
            freqPerSucker = []
            print(k)
            for ns in range(nsuckers):
                # print(actionFreq[ns][k],refActionfreq[ns][k])
                # if refActionfreq[ns][k] !=0:
                if actionFreq[ns][k]==0:
                    freqPerSucker.append(0)
                else:
                    try:
                        freqPerSucker.append(actionFreq[ns][k]/refActionfreq[ns][k])
                    except ZeroDivisionError:
                        freqPerSucker.append(np.inf)
                    #     print(actionFreq[ns][k],refActionfreq[ns][k])
                    #     exit()
                # else:
                    # freqPerSucker.append(actionFreq[ns][k])
            print(freqPerSucker)
            tentacleState[k] = np.array(freqPerSucker)
    else:
        for k in keys:
            freqPerSucker = []
            print(k)
            for ns in range(nsuckers):
                freqPerSucker.append(actionFreq[ns][k])
            print(freqPerSucker)
            tentacleState[k] = np.array(freqPerSucker)
    # print(tentacleState)
    
    #prepare what I plot--> combine base|<- with <-|<-  with <-|tip ecc..
    # stateKeys = set(['->|<-','<-|->','base|<- <-|<- <-|tip', 'base|-> ->|-> ->|tip'])
    # stateKeys = {'->|<-':0,'<-|->':1,'<-|<-':2, '->|->':3}
    
    tentacleState['<-|<-'] = tentacleState['base|<-'] + tentacleState['<-|<-'] + tentacleState['<-|tip']
    tentacleState['->|->'] = tentacleState['base|->'] + tentacleState['->|->'] + tentacleState['->|tip']
    for k in baseKeys + tipKeys:
        del tentacleState[k]
        keys.remove(k)
    print(tentacleState)
    # print(np.array([tentacleState[k] for k in tentacleState]))
    
    plt.figure()
    fig = plt.subplot(xlabel='sucker', ylabel='')
    fig.set_yticks([0,1,2,3],list(keys))
    fig.set_xticks(list(np.arange(0,nsuckers)),['base']+list(np.arange(2,nsuckers))+['tip'])
    # fig.set_xlim([-1,nsuckers+1])
    # fig.set_ylim([-1,4])
    plt.ion()
    plt.show()
    # X,Y = np.meshgrid(np.arange(0,nsuckers),stateKeys.items())
    Z = np.array([tentacleState[k] for k in tentacleState])
    print(Z)
    # if refActionfreq is not None:
    #     Znorm = np.round(Z,2)
    # else:
    Zmax = np.nanmax(Z[np.abs(Z) != np.inf])
    Znorm = np.round(Z/Zmax,2)
    print(Zmax)
    # print(Z[2])
    # fig.pcolor(X, Y, Z)
    fig.imshow(Z)
    for i in range(len(keys)):
        for j in range(nsuckers):
            text = fig.text(j, i, Znorm[i, j],
                       ha="center", va="center", color="w")
        
        
        


def actionMapState_dict(policy,is_ganglia,isHive,n_suckers,nAgents):
    '''
    Returns a number reoresenting frequency ogf anchoring action per state.
    NOT sure of the interpretation, but it could be a compact number to assign to a policy?
    In principle this is knowable a priori.. A given policy corresponds to a fixed amount of anchoring actions for each given state..
    Since I see this as a per tentacle property, I return the overall active suckers per state for the given policy. 
    De facto I'm mapping not hive into hive doing so in terms of action population..

    CAREFUL: In practice many staes are never visited. Needs to be weighted by an on-policy state visits frequency
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




def getPolicyStats(Q,env,nLastPolicies = 100,runtimeInfo=None,outFolder="./",info=None):
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
            
    # print("out file name:")
    fileName = outFolder+"Raw_policyMeasuresFor%dsuckers_omega%.2f_%s"%(Q._nsuckers,env.omega,type)
    

    # print(fileName)
    # input()

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
    # for pol_n in trange(1,nLastPolicies+1):
    for pol_n in range(1,nLastPolicies+1):
        polIndx.append(c)
        value.append(Q.set_referencePolicy(pol_n))
        # print("")
        vel,state_freq,norm_activeSuckers,visited = Q.evaluatePolicy(env)
        norm_vels.append(vel)
        visitedStates.append(len(visited))
        relative_state_freqs.append({s:(state_freq[s]-state_freqRandom[s])/state_freqRandom[s]*100 for s in visitedRandom}) #WARNING: unused
        activeS.append(norm_activeSuckers)
        c+=1
    norm_vels = np.array(norm_vels)
    max_vel= np.amax(norm_vels)
    bestPolIndx = np.argmax(norm_vels) +1 #Returns first occurrence
    average_normVel = np.average(norm_vels)
    std_normVel = np.std(norm_vels)

    #best 100 policies
    sorted_norm_vels = np.sort(norm_vels)[::-1]
    avergeSorted = np.average(sorted_norm_vels[0:int(nLastPolicies/2)])
    stdSorted = np.std(sorted_norm_vels[0:int(nLastPolicies/2)])
    
    outFileName = "SUMMARY_policyMeasuresFor%dsuckers_omega%.2f_%s.txt"%(Q._nsuckers,env.omega,type)
    outFile = open(outFileName,'w')

    line = '***** SUMMARY ****\nn suckers = %d, T length = %d, omega = %.2f, x0= %.3f Type: %s'%(Q._nsuckers,env.tentacle_length,env.omega,env.x0,type)
    outFile.write(line)
    line='\nNumber of policies analyzed : %d.'%(nLastPolicies)
    outFile.write(line)
    if info is not None:
        line = '\nConvergence criterion (tollerance) = %.3f\nPlateau exploration parameters: lr = %.4f\tepsilon =%.3f\tsteps = %d'%(info['convergence'],info['lr'],info['eps'],info['steps'])
        outFile.write(line)
    line='\n\nNORMALIZED VEL AVERAGE= %.4f +- %.4f'%(np.round(average_normVel,4),np.round(std_normVel,4))
    outFile.write(line)
    line='\nNORMALIZED VEL BEST%d POLICIES= %.4f +- %.4f'%(int(nLastPolicies/2),np.round(avergeSorted,4),np.round(stdSorted,4))
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
    tentacleInfo = 'n suckers = %d, T length = %d, omega = %.2f, x0= %.3f Type: %s\n'%(Q._nsuckers,env.tentacle_length,env.omega,env.x0,type)
    paramInfo = 'Convergence criterion (tollerance) = %.3f\nPlateau exploration parameters: lr = %.4f\tepsilon =%.3f\tsteps = %d\n'%(info['convergence'],info['lr'],info['eps'],info['steps'])
    polInfo = "Visited states under random_policy:%d/%d\nindx\tnorm vel\tactiveSts[%%]\tvisitedStates\tTotal av_value"%(len(visitedRandom),Q.state_space_dim)
    header = tentacleInfo + paramInfo + polInfo
    now = datetime.now().ctime()
    if runtimeInfo is not None:
        footer = "runtime for training: "+runtimeInfo+"\nCurrent time: "+now
    else:
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