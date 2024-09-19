#Contains accessory funcitons used for testing, comparing, plotting ecc
# More of a workbook note

# import numpy as np
import globals
from tqdm import trange
import numpy as np
from datetime import datetime
import copy
import matplotlib.pyplot as plt
import env

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


def anal_vel_l0norm(N,omega):
    k = 2*np.pi/N
    amplitude_fraction = 1/env.x0Fraction
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
    directoryName = filenames[-1].split("/")[-2]
    print(directoryName)
    n_suckers = int(re.match("(\d+)",directoryName).group(1))
    print(n_suckers)
    input()
    policies = []
   
    for filename in filenames:
        pol = {}
        print(filename)
        match_Ganglia = re.search("(\d)(GANGLIA)",filename)  
        match_Hive = re.search("HIVE",filename)  
        match_period = re.search("(\d+)N",filename)
        if match_period:
            print("considering tentacle with different wave lengths")
            N  = int(match_period.group(1))
            print("periodicity=",N)
            pol["periodicity"] = N
        if match_Hive:
            print("hive")
            isHive=True
            pol["hive"] = isHive
        else:
            print("not hive")
            isHive=False
            pol["hive"] = isHive
        if match_Ganglia:
            nGanglia = int(match_Ganglia.group(1))
            print(nGanglia,"Ganglia")
            pol["ganglia"] = nGanglia
        else:
            print("multiagent")
            nGanglia = 0
            pol["ganglia"] = nGanglia
        policy = np.load(filename,allow_pickle=True)
        pol["policy"] = policy
        pol["n_suckers"] = n_suckers
        # pol={"policy":policy,"ganglia":nGanglia,"hive":isHive,"n_suckers":n_suckers}
        # print(nGanglia,isHive)
        policies.append(pol)
    
    return policies

def loadPolicy(filename):
    policy = np.load(filename,allow_pickle=True)
    return policy


def suckerImportance(env,policy,secondOrder = True):
    '''In this case i pass the environment myself.
        All info retrieved from policy and environment
    '''
    try:
        nPolicies = len(policy)
        isHive = False
        print("NOT HIVE")
        print("number of independent agents (not Hive) = ",nPolicies)
    except TypeError:
        print("HIVE")
        isHive = True
    
    periodicity = env.N
    n_suckers = env.info["n suckers"]
    periodicityInfo = "pulse period = "+str(np.round(periodicity/n_suckers,2)) + " lengths"
    learning_space = env.info["learning space"]
    state_space_dim = learning_space[0]
    action_space_dim = learning_space[1] 
    isGanglia = env.info["isGanglia"]
    if isGanglia:
        n_ganglia = env.info["n ganglia"]
    else:
        n_ganglia = 0

    # INFO 

    if n_ganglia>0:
        ganglia=True
        if isHive:
            label = "%dGanglia HIVE"%n_ganglia
            architecture = "%dGanglia_HIVE_%dsuckers"%(n_ganglia,n_suckers)
        else:
            label = "%dGanglia"%n_ganglia
            architecture = "%dGanglia_%dsuckers"%(n_ganglia,n_suckers)
    else:
        ganglia=False
        if isHive:
            label = "Multiagent HIVE"
            architecture = "Multiagent_HIVE_%dsuckers"%n_suckers
        else:
            label = "Multiagent"
            architecture = "Multiagent_%dsuckers"%n_suckers
    print("\n\n** %s **\n"%label)
    title = architecture

    # ---------------

    # SET UP FIGURE
    plt.figure()
    plt.ion()
    fig = plt.subplot(xlabel='sucker ID', ylabel='importance 1st order')
    fig.set_xticks(np.arange(n_suckers))
    fig.set_title(label="sucker importance "+periodicityInfo)


    # ---------------

    nTrialSteps = 20000
    pdf = np.zeros(n_suckers)
    env.reset(equilibrate = True)
    state = env.get_state()
    for t in trange(nTrialSteps):
        action = getAction(env.info,policy,state,isHive)
        _devils_ranked,main_devil = selectRelevantSucker(env,action)
        pdf[main_devil] +=1
        state,_r,_t = env._stepOverdamped(action)
    #normalize
    pdf = pdf/nTrialSteps
    print(pdf)
    ranked_averaged_devils = np.argsort(pdf)[::-1]
    print("ranked importance: ",ranked_averaged_devils)
    print("1st Most important: ",ranked_averaged_devils[0])
    fig.bar(np.arange(pdf.size),pdf)
    plt.show()

    if secondOrder:
        fixID = ranked_averaged_devils[0]

        plt.figure()
        plt.ion()
        fig = plt.subplot(xlabel='sucker ID', ylabel='importance 2nd order')
        xlabels = [str(s)for s in np.arange(n_suckers)]
        xlabels[fixID] = "FIXED"
        fig.set_xticks(np.arange(n_suckers),labels=xlabels)
        fig.set_title(label="sucker importance "+periodicityInfo)  



        pdf = np.zeros(n_suckers)
        env.reset(equilibrate = True)
        state = env.get_state()
        for t in trange(nTrialSteps):
            action = getAction(env.info,policy,state,isHive)
            _devils_ranked,second_devil = selectRelevantSucker(env,action,fixID=fixID)
            pdf[second_devil] +=1
            state,_r,_t = env._stepOverdamped(action)
        #normalize
        pdf = pdf/nTrialSteps
        print(pdf)
        ranked_averaged_devils = np.argsort(pdf)[::-1]
        print("2nd Most important: ",ranked_averaged_devils[0])
        ranked_averaged_devils = np.delete(ranked_averaged_devils, np.where(ranked_averaged_devils == fixID)[0][0])
        ranked_averaged_devils = np.insert(ranked_averaged_devils,0,fixID)
        print("2nd order list of devils: ", ranked_averaged_devils)
        fig.bar(np.arange(pdf.size),pdf,color='orange')
        plt.show()
    return ranked_averaged_devils

def policyRobustnessStudy(policies,suckerCentric=True,plot=True,normalize=False,randomSuckerSel =True):
    instantaneusWorseSucker = False
    goTohigherOrder = True
    average = False
    importantSuckers={}
    from env import Environment
    sim_shape = (20,)
    t_position = 100
    results=[]
    n_suckers = policies[0]["n_suckers"]
    if plot ==True :
        plt.figure()
        plt.ion()
        if suckerCentric:
            epsilon = 1 #100% failure probability of n failing suckers randomly chosen
            if randomSuckerSel:
                title = "%dSUCKERS Tentacle  Random sucker failure\nRandom choice prob = %d %%"%(n_suckers,epsilon*100)
            else:
                if goTohigherOrder:
                    title = "%dSUCKERS Tentacle 2nd order more influential suckers failure \nRandom choice prob = %d %%"%(n_suckers,epsilon*100)
                    if average:
                        title = "%dSUCKERS Tentacle 2nd order more influential suckers failure \nTrajectory assesment\nRandom choice prob = %d %%"%(n_suckers,epsilon*100)
                else:
                    title = "%dSUCKERS Tentacle 1st order more influential suckers failure \nRandom choice prob = %d %%"%(n_suckers,epsilon*100)
                    if average:
                        title = "%dSUCKERS Tentacle 1st order more influential suckers failure \nTrajectory assesment\nRandom choice prob = %d %%"%(n_suckers,epsilon*100)
            xlabel = "failing suckers" #but not always the same, is always randomly chosen
            mode = "suckerCentric"
            fmt="%d\t\t%.4f"
        else:
            title = "%dSUCKERS Tentacle  Agent epsilon failure"%n_suckers
            xlabel = "epsilon"
            mode = "epsilonGreedy"
            fmt="%.1f\t\t%.4f"
        if not normalize:
            fig = plt.subplot(xlabel=xlabel, ylabel='v/x0')
        else:
            fig = plt.subplot(xlabel=xlabel, ylabel='(v/x0)/vMAX')
        fig.set_title(label=title)
    
    

    fig_allPdfs = None
    fig_allPdfs2 = None
    width = 0.8
    width2 = 0.8
    for pol in policies:
        vels =[]
        policy = pol["policy"]
        n_ganglia = pol["ganglia"]
        isHive= pol["hive"]
        if n_ganglia>0:
            ganglia=True
            if isHive:
                label = "%dGanglia HIVE"%n_ganglia
                architecture = mode+"Robustness_%dGanglia_HIVE_%dsuckers"%(n_ganglia,n_suckers)
            else:
                label = "%dGanglia"%n_ganglia
                architecture = mode+"Robustness_%dGanglia_%dsuckers"%(n_ganglia,n_suckers)
        else:
            ganglia=False
            if isHive:
                label = "Multiagent HIVE"
                architecture = mode+"Robustness_Multiagent_HIVE_%dsuckers"%n_suckers
            else:
                label = "Multiagent"
                architecture = mode+"Robustness_Multiagent_%dsuckers"%n_suckers
        print("\n\n** %s **\n"%label)
        try:
            period = pol["period"]
            print("policy learned on wavelength different from tentacle length")
            print("N = ",period)
        except:
            period=None
        env = Environment(n_suckers,sim_shape,t_position,omega =0.1,isOverdamped=True,is_Ganglia=ganglia,nGanglia=n_ganglia,period = period)
    #A. SUCKER CENTRIC: Epsilon 100% robustness with respect to n suckers
        
        if suckerCentric:
            ranked_averaged_devils = None
            if (not instantaneusWorseSucker) and (not randomSuckerSel):
                    if fig_allPdfs is None:
                        plt.figure()
                        plt.ion()
                        fig_allPdfs = plt.subplot(xlabel='sucker ID', ylabel='importance 1st order')
                        fig_allPdfs.set_xticks(np.arange(n_suckers))
                        fig_allPdfs.set_title(label="sucker importance")
                    else:
                        pass
                    #interested to gather stat over worse sucker in time. From that i establish how often a sucker is the worse
                    print("assessing most influentials suckers")
                    
                    if not average:
                        nTrialSteps = 20000
                        pdf = np.zeros(n_suckers)
                        env.reset(equilibrate = True)
                        state = env.get_state()
                        for t in trange(nTrialSteps):
                            action = getAction(env.info,policy,state,isHive)
                            _devils_ranked,main_devil = selectRelevantSucker(env,action)
                            pdf[main_devil] +=1
                            state,_r,_t = env._stepOverdamped(action)
                        #normalize
                        pdf = pdf/nTrialSteps
                        print(pdf)
                        ranked_averaged_devils = np.argsort(pdf)[::-1]
                        print(ranked_averaged_devils)
                        print("1st Most important: ",ranked_averaged_devils[0])
                        
                        # print("Plotting analysis of average most influential sucker")
                        
                    else:
                        print("\n<NEW>: asssessing sucker importance over sever runs of failures of same sucker\n")
                        ranked_averaged_devils,pdf = selectRelevantSucker_average(env,policy,isHive)
                        print(ranked_averaged_devils)
                        print("1st Most important: ",ranked_averaged_devils[0])
                    importantSuckers [label] = [ranked_averaged_devils[0]]
                    plt.figure()
                    plt.ion()
                    # fig2 = plt.subplot(xlabel='sucker ID', ylabel='1st order importance')
                    # fig2.set_title(label = label)
                    # fig2.set_xticks(np.arange(n_suckers))
                    # fig2.bar(np.arange(pdf.size),pdf)

                    fig_allPdfs.bar(np.arange(pdf.size),pdf,label=label,width=width)
                    fig_allPdfs.legend()
                    width -=0.1
                    if goTohigherOrder: 
                        fixID = ranked_averaged_devils[0]
                        labels = [str(s)for s in np.arange(n_suckers)]
                        if not average:
                            #From direct observation it is always sucker 11 when I check instantaneus devil at each time step (not over the whole trajectory as for "average" option..)
                            labels[fixID]="fixed"
                        print("COMPUTING SECOND ORDER FIXING SUCKER",fixID)
                        if fig_allPdfs2 is None:
                            plt.figure()
                            plt.ion()
                            fig_allPdfs2 = plt.subplot(xlabel='sucker ID', ylabel='importance 2nd order')
                            # print("labels",labels)
                            fig_allPdfs2.set_xticks(np.arange(n_suckers),labels=labels)
                            fig_allPdfs2.set_title(label="sucker importance")

                        if not average:
                            pdf = np.zeros(n_suckers)
                            env.reset(equilibrate = True)
                            state = env.get_state() 
                            
                            for t in trange(nTrialSteps):
                                action = getAction(env.info,policy,state,isHive)
                                _devils_ranked,second_devil = selectRelevantSucker(env,action,fixID)
                                pdf[second_devil] +=1
                                state,_r,_t = env._stepOverdamped(action)
                            pdf = pdf/nTrialSteps
                            print(pdf)
                            ranked_averaged_devils = np.argsort(pdf)[::-1] 
                            
                            
                            print(ranked_averaged_devils)
                            print("2nd Most important: ",ranked_averaged_devils[0])
                            
                            #Insert at correct position what found previously
                            ranked_averaged_devils = np.delete(ranked_averaged_devils, np.where(ranked_averaged_devils == fixID)[0][0])
                            ranked_averaged_devils = np.insert(ranked_averaged_devils,0,fixID)
                            print("2nd order list of devils: ", ranked_averaged_devils)
                            # input("continue ?")
                        else:
                            ranked_averaged_devils,pdf = selectRelevantSucker_average(env,policy,isHive,fixID)
                            print("2nd order list of devils: ", ranked_averaged_devils)
                            print("2nd Order Most important: ",ranked_averaged_devils[1])
                        # plt.figure()
                        # plt.ion()
                        # fig2 = plt.subplot(xlabel='sucker ID', ylabel='2nd order importance')
                        # fig2.set_title(label = label)
                        # fig2.set_xticks(np.arange(n_suckers),labels = labels)
                        # fig2.bar(np.arange(pdf.size),pdf)
                        fig_allPdfs2.bar(np.arange(pdf.size),pdf,label=label,width=width2)
                        fig_allPdfs2.legend()
                        width2-=0.1
                        importantSuckers [label].append([ranked_averaged_devils[1]])
            # input("continue?")
            failing_suckers = []
            if randomSuckerSel:
                max_failingSuckers = env._nsuckers+1 #all suckers
            else:
                max_failingSuckers = env._nsuckers+1 #all suckers
            for fs in range(max_failingSuckers):
                vel = robustnessAnal(env,policy,isHive,epsilon,failingSuckers=fs,epsilonGreedyFail=False,randomSuckerSel = randomSuckerSel,whichDevils = ranked_averaged_devils)
                vels.append(vel)
                failing_suckers.append(fs)
            
            out = [failing_suckers,vels]
        #B. AGENT CENTRIC: Robustness with respect to increasing epsilon
        else:
            epsilons = np.linspace(0,1,10,endpoint=False)
            for epsilon in epsilons:
                vel = robustnessAnal(env,policy,isHive,epsilon,epsilonGreedyFail=True)
                vels.append(vel)
            out = [epsilons,vels]
        if normalize:
            maxv= vels[0]
            out[1] = [v/maxv for v in out[1]]
        results.append(out)
        if normalize:
            np.savetxt("results/robustness/"+architecture+"_NORM.txt",np.column_stack((out[0],np.round(out[1],4))),fmt = fmt,header=xlabel+"\tvel/velMAX")
        else:
            np.savetxt("results/robustness/"+architecture+".txt",np.column_stack((out[0],np.round(out[1],4))),fmt = fmt,header=xlabel+"\tvel")
        if plot==True:
            fig.plot(out[0],out[1],'-o',lw=5,label = label)
            fig.axhline(0,ls='--',c='black')
            fig.legend()
            plt.show()
    input()
    return importantSuckers
def selectRelevantSucker_average(env,policy,isHive,fixID=None,descending=False):
    """
    relevant sucker estimated making it play wrong action over several evolution steps
    """
    trialSteps = 10000
    n_suckers = env._nsuckers
    
    suckers = set([s for s in range(n_suckers)])
    pdf = np.zeros(n_suckers)
    vels = np.empty(len(suckers)) 
    if fixID is not None:
        suckers =suckers-{fixID}
        vels[fixID] = 100 #dummy to not put fixed sucker index index ahead

    
    
    for suckerID in suckers:
        env.reset(equilibrate = True)
        state = env.get_state()
        avVel = 0
        for s in trange(trialSteps):
            onPolActions = getAction(env.info,policy,state,isHive)
            actions = onPolActions.copy()
            if fixID is not None:
                actions[fixID] = abs(actions[fixID]-1)
            # actions[suckerID] = abs(actions[suckerID] -1)
            actions[suckerID] = np.random.randint(0,2)
            # print(actions)
            state,_r,_t = env._stepOverdamped(actions)
            avVel += env.get_averageVel()
            
        


        avVel = avVel/trialSteps
        pdf[suckerID] = avVel
        vels[suckerID] = avVel
        print(suckerID,avVel)
    ids = np.argsort(vels) #sorted indices from smaller to larger velocity
    if fixID is not None:
        # ids = np.insert(ids,0,fixID)
        # main=ids[1]
        ids = np.delete(ids, np.where(ids == fixID)[0][0])
        ids = np.insert(ids,0,fixID)
        # print(vels)
        # input()
    else:
        pass
        # main = ids[0]
    if descending:
        ids=ids[::-1]
    # print(ids)
    # if np.unique(ids).size != ids.size:
    #     print(vels)
    #     print(ids)
    #     input("PROBLEM")
    
    print(vels)
    #normalization
    pdf = (pdf/sum(abs(pdf))) #smaller more influential
    print(pdf)
    return ids,pdf

def selectRelevantSucker(env,onPolActions,fixID=None,descending=False):
    '''
    Returns sorted indexes of the less (or most) important sucker in the movement by making all sucker play wrong action and check which was the most impactful.
     -- NOT SURE   UPDATE TO DO : If step provided averages over a longer time the obtained velocity to assess most impactful sucker
    INPUT: environment,on policy actions
    ATTENCTION: This analysis is not precise especially when considering more than one failing sucker since cannot correlate the effect of several suckers together in this form..
    '''
    #PROBLEM I DON'T WANT TO PERTURB original tentacle..
    #Need to copy the object..
    # initialEnv = copy.deepcopy(env)
    #Need to do it for each sucker
    # vels=[]
    
    n_suckers = env._nsuckers
    actions = onPolActions.copy()
    # if fixID is not None:
    #     print("original actions", onPolActions)
    # print("########")
    # print(actions)
    suckers = set([s for s in range(n_suckers)])
    vels = np.empty(len(suckers)) 
    if fixID is not None:
        suckers =suckers-{fixID}
        actions[fixID] = abs(actions[fixID]-1)

    refActions = actions.copy()
    # if fixID is not None:
    #     print(refActions)
    #     print(suckers)
    #     input()
    
    for id in suckers:
        actions[id] = abs(actions[id]-1) #play the contrary move of what prescribed
        # print(actions)
        # print(env._t)
        #state,r,_t=env._stepOverdamped(onPolActions)
        avVel=0

        instVel=env._stepOverdampedVIRTUAL(actions) #avoid actual update of positions and observables, returns only instanteneous velocity
        vels[id] = instVel
        # env.render()
        # time.sleep(0.1)
        # vels.append(instVel)   
        # actions = onPolActions.copy() #RESET
        actions = refActions.copy()
    
    ids = np.argsort(vels) #sorted indices from smaller to larger velocity
    if fixID is not None:
        ids = np.insert(ids,0,fixID)
        main=ids[1]
        # print(vels)
        # input()
    else:
        main = ids[0]
    if descending:
        ids=ids[::-1]
    # print(ids)
    # if np.unique(ids).size != ids.size:
    #     print(vels)
    #     print(ids)
    #     input("PROBLEM")
    return ids,main

def getAction(info,policy,state,isHive,epsilon=None):
    '''
    Returns action formatted for the whole tentacle (not divided into agent clusters)
    '''
    if isHive:
        policy = np.array([policy.item()])#just for the way they were saved, and to allow zip loop
    isGanglia = info["isGanglia"]
    nsuckers = info["n suckers"]
    action_space = info["learning space"][1]

    action = []
   ############                GANGLIA          ###################
    if isGanglia:
        state = [globals.interpret_binary(s) for s in state]
        nGanglia = info["n ganglia"]
        nAgents = nGanglia
        padding= int(nsuckers/nAgents)
        if nAgents ==1:
         #the way in which policies are saved consider the format of 1 ganglion as a single element array of policies
            isHive = False
        if isHive:
            ag,gindxs  = ([0]*nGanglia,range(nGanglia))
        else:
            ag,gindxs  = (range(nGanglia),range(nGanglia))
        
        for a,gind in zip(ag,gindxs):#agent,ganglion index
            if epsilon is not None:
                if np.random.random() < (1 - epsilon):
                    action.append(globals.make_binary(policy[a][state[gind]],padding))
                else:
                    # randomChoice=1
                    action.append(globals.make_binary(np.random.randint(0,action_space),padding))
            else:
                action.append(globals.make_binary(policy[a][state[gind]],padding))
        action_flattened = [a for al in action for a in al]
        return action_flattened
    else:
############                MULTIAGENT          ###################
        if isHive:
            nAgents = 1
            ag,sindxs  = ([0]*nsuckers,range(nsuckers))
        else:
            nAgents = nsuckers
            ag,sindxs  = (range(nAgents),range(nsuckers))
        for a,sid in zip(ag,sindxs): #agent,sucker index
            #get on policy action
            if epsilon is not None:
                if np.random.random() < (1 - epsilon):
                    action.append(policy[a][state[sid]])
                else:
                    action.append(np.random.randint(0,action_space)) #OBS: in this case there's a 50% probability to do the right choice..
            else:
                action.append(policy[a][state[sid]])

        return action
     

    
            
    


def robustnessAnal(env,policy,isHive,epsilon,failingSuckers=0,epsilonGreedyFail=False,doMovie = False, steps = 20000, randomSuckerSel = True,whichDevils = None):
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
    instantaneusWorseSucker = False
    import random

    nsuckers = env._nsuckers
    info = env.info
    isGanglia = info["isGanglia"]
    if epsilonGreedyFail==False:
        print("<INFO> : Sucker centric perturbation analysis")
        print("n faling suckers = ",failingSuckers)
        print("Probability of failure = %d%%"%(epsilon*100))
        if not randomSuckerSel:
            print("<WARNING> Random Selection = FALSE")
    else:
        print("<INFO> : Agent centric (epsilon greedy) perturbation analysis")
        print("Probability of random action = %d%%"%(epsilon*100))

    if isGanglia:
        nGanglia = info["n ganglia"]
        print("Ganglia")
        print("n Gagnlia = ",nGanglia)
    else:
        print("Multiagent")
        
    print("IS HIVE =",isHive)
    
    

    # env.reset(equilibrate = True)
    env.reset()
    env.equilibrate(1000)
    state = env.get_state()
    print(state)
    print(env.deltaT)
    currentT=[]
    mostImportantID=[]


        
    # print("here",whichDevils)
    for t in trange(steps): 
        currentT.append(t)
        if not epsilonGreedyFail:
            action = getAction(env.info,policy,state,isHive) 
            #Generalize for any number of failing suckers
            # suckers = list(np.arange(1,nsuckers-1)) #EXCLUDE BASE AND TIP
            suckers = list(np.arange(0,nsuckers))
            devil_suckers = set()
            gotDevil=False
            if not randomSuckerSel:
                if whichDevils is None:
                #at each time step according to current situation (action played before), establish which are most impactful suckers (neglecting correlations)
                    devils,main = selectRelevantSucker(env,action)
                    mostImportantID.append(main)
                else: 
                    devils = whichDevils
            for i in range(failingSuckers):
                if randomSuckerSel: #default
                    devil = random.choice(suckers)
                else:
                    devil = devils[i] #here not important removal from sucker list. Index from more to less impactful
                # print(devil)
                # input()
                if np.random.random() < (1 - epsilon):
                    pass
                else:
                    # action_flattened[devil] = abs(action_flattened[devil]-1) #does exaclty the contrary of what prescibed  
                    gotDevil=True
                    action[devil] =np.random.randint(0,2)
                
                suckers.remove(devil)
                if gotDevil:
                    devil_suckers.add(devil)
        else:
              action = getAction(env.info,policy,state,isHive,epsilon)
        
        # print(action)
        # input()
        state,r,_t=env._stepOverdamped(action)
        
        
        if doMovie and t%10==0:
            if not epsilonGreedyFail:
                env.render(colored_suckers=devil_suckers)
            else:
                env.render()
            
        
    if randomSuckerSel==False and failingSuckers ==1 and (instantaneusWorseSucker):
        plt.figure()
        plt.ion()
        fig = plt.subplot(xlabel='time step', ylabel='suckerID')
        if isGanglia:
            if isHive:
                fig.set_title(label='%dSuckers %dGanglia HIVE\nMost influencial sucker analysis'%(nsuckers,nGanglia))
            else:
                fig.set_title(label='%dSuckers %dGanglia\nMost influencial sucker analysis'%(nsuckers,nGanglia))
        else:
            if isHive:
                fig.set_title(label='%dSuckers Multiagent HIVE\nMost influencial sucker analysis'%nsuckers)
            else:
                fig.set_title(label='%dSuckers Multiagent \nMost influencial sucker analysis'%nsuckers)
        fig.plot(currentT,mostImportantID,'o',ms=4)


   
    print("(Norm) Velocity=",env.get_averageVel()/env.x0)

    return env.get_averageVel()/env.x0

def onPolicyStateActionVisit(env,policy,isHive):
    '''
    Returns for each sucker the frequency each multiagent state is played.
    In mind I have that I can do 4 color plots over the tentacle for the 4 different internal states (while always same for base and tip)
    We can use as bottom line the hive policy
    Return also average active suckers of analyzed policy
    Better output is the one normalized per state frequency
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
        actionFreqPerSucker.append({'->|->':0, '<-|->':0,'->|<-': 0 ,'->|tip':0,'<-|<-':0,'<-|tip':0,'base|<-':0,'base|->':0})
        stateFreqPerSucker.append({'->|->':0,'<-|->':0,'->|<-': 0 ,'->|tip':0,'<-|<-':0,'<-|tip':0,'base|<-':0,'base|->':0})
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
    

def plotTSvisits(actionFreq,refActionfreq=None,maxNorm = False):
    import matplotlib.pyplot as plt
    """
    If ref is given, the color plot is normalized by the reference (standard choice would be to use the standard hive policy..)
    """
    intermediateKeys = ['->|->','<-|->','->|<-','<-|<-']
    baseKeys = ['base|<-','base|->']
    tipKeys = ['->|tip','<-|tip']
    #colorplot y-axis 4 states x-axis sucker position 
    
    #Read plot tile from key and value from item. 
    #First I have to gather per key all suckers
    #Keep it general for easier adaptation (be agnostic about key names..)
    nsuckers = len(actionFreq)
    # keys = set(actionFreq[0].keys())
    keys = intermediateKeys + baseKeys + tipKeys 
    print(keys)
    tentacleState = {}
    if refActionfreq is not None:
        print("NORMALIZING WITH REFERENCE..")
        for k in keys:
            freqPerSucker = []
            # print(k)
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
            # print(freqPerSucker)
            tentacleState[k] = np.array(freqPerSucker)
    else:
        for k in keys:
            freqPerSucker = []
            # print(k)
            for ns in range(nsuckers):
                freqPerSucker.append(actionFreq[ns][k])
            # print(freqPerSucker)
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
    # print(tentacleState)
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
    if maxNorm:
        Zmax = np.nanmax(Z[np.abs(Z) != np.inf])
        Znorm = np.round(Z/Zmax,2)
        print(Zmax)
    else:
        Znorm = np.round(Z,2)
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
    TODO : value like this not very meaningful.. always different number since precise Q are all different...
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
       
        #below the policy is evaluated
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

    #best 250 policies
    sorted_norm_vels = np.sort(norm_vels)[::-1]
    avergeSorted = np.average(sorted_norm_vels[0:int(nLastPolicies/2)])
    stdSorted = np.std(sorted_norm_vels[0:int(nLastPolicies/2)])

    sortedIndexes = np.argsort(norm_vels)[::-1]
    bestPolIndexes = sortedIndexes[0:int(nLastPolicies/2)]
    #NEW COUNT NUMBER OF DISTINT POLICIES EXPLORED AMONG ALL AND AMONG THE BEST 250
    npolicies = countPolicies(Q._lastPolicies,Q._parallelUpdate)
    npoliciesBest = countPolicies(Q._lastPolicies[bestPolIndexes],Q._parallelUpdate)
    
    outFileName = "SUMMARY_policyMeasuresFor%dsuckers_omega%.2f_%s.txt"%(Q._nsuckers,env.omega,type)
    outFile = open(outFileName,'w')

    line = '***** SUMMARY ****\nn suckers = %d, T length = %d, omega = %.2f, x0= %.3f Type: %s'%(Q._nsuckers,env.tentacle_length,env.omega,env.x0,type)
    outFile.write(line)
    line='\nNumber of policies analyzed : %d.'%(nLastPolicies)
    outFile.write(line)
    if info is not None:
        line = '\nConvergence criterion (tollerance) = %.3f\nPlateau exploration parameters: lr = %.4f\tepsilon =%.3f\tsteps = %d'%(info['convergence'],info['lr'],info['eps'],info['steps'])
        outFile.write(line)
    line='\n\nNORMALIZED VEL AVERAGE= %.4f +- %.4f\t number of distint policies = %d'%(np.round(average_normVel,4),np.round(std_normVel,4),npolicies)
    outFile.write(line)
    line='\nNORMALIZED VEL BEST%d POLICIES= %.4f +- %.4f\t number of distint policies = %d'%(int(nLastPolicies/2),np.round(avergeSorted,4),np.round(stdSorted,4),npoliciesBest)
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


def countPolicies(policies,isHive):
    #n is the agent t is the policy counter
    # before = np.array(list(self._lastPolicies[-t-1][n].values()))
    # after = np.array(list(self._lastPolicies[-t][n].values()))
    # difference =np.sum(before-after)
    # isHive = Q._parallelUpdate
    # policies = Q._lastPolicies
    #nAgents = Q._nAgents
    nPolicies = len(policies)
    
    if isHive:
        axis = 0 
        nAgents = 1
        #here I have one single policy F
        pol = []
        for t in range(nPolicies):
            pol.append(list(policies[t].values()))  

    else:
        axis = 1 
        #ATTENZIONE: devi comparare agente per agente (non ha senso comparare tra suckers o ganglia diversi..)
        nAgents = len(policies[0])
        # print(nAgents)
        pol = []
        for a in range(nAgents):
            polAgent = []
            for t in range(nPolicies):
                polAgent.append(list(policies[t][a].values()))
            pol.append(polAgent)
    pol = np.array(pol) #pol[nAgent,nSavedPolicy]
    _unique,countsIdentical = np.unique(pol,axis=axis,return_counts = True) # counts is an array that tells you for any elements how many times
    countDifferent = nPolicies-len(countsIdentical) #since counts identical is an array indicating at each position number of identical columns in the unique matrix
    # or countDifferent = nPolicies - unique.shape[1]
        
    return countDifferent
    # return counter,counterIndicized

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