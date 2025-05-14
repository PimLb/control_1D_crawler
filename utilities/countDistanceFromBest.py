import os
import inspect
import numpy as np 
import sys
from tqdm import trange
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print(currentdir)
parentdir = currentdir.split('/')[:-4]
parentdir = '/'.join(parentdir)
print(parentdir)

sys.path.insert(0, parentdir) 

#COMMENT: for control number of states is the sum of the states of each Q matrix (if multi agent) rather than the product. 
# This since in control when I follow a policy each per agent policy can at most traverse all its states, its like combining all in a single matrix (sum).
# Is a policy point of view normalization non all vaible tentacle states which wiuld imply all possible combinations.

from env import Environment
from learning import actionValue

# HERE UNIQUE FUNCTION

def countPolicies(policies,isHive,distributed):
    nPolicies = len(policies)
    axis = 0 
    pol_values = []
    pol=[]
    if isHive:
        #here I have one single policy F
        for t in range(nPolicies):
            pol_values.append(list(policies[t].values()))  
            pol.append(policies[t])
    else:
        #ATTENZIONE: devi comparare agente per agente (non ha senso comparare tra suckers o ganglia diversi..)
        nAgents = len(policies[0])
        for t in range(nPolicies):
            polAgent_values = []
            polAgent = []
            for a in range(nAgents):
                if distributed:
                    if (a==0 or a == (nAgents-1)):
                        policies[t][a]["dummy1"] = -1
                        policies[t][a]["dummy2"] = -1
                polAgent_values.append(list(policies[t][a].values()))
                polAgent.append(policies[t][a]) #row is the time column the agent. I have to compare each row to establish identity
            pol_values.append(polAgent_values)
            pol.append(polAgent)
    pol_values = np.array(pol_values) #pol[nAgent,nSavedPolicy]
    pol = np.array(pol)
    _unique,indx,countsIdentical = np.unique(pol_values,axis=axis,return_counts = True,return_index=True) # counts is an array that tells you for any elements how many times
    uniquePol = pol[indx]
    # countDifferent = len(countsIdentical) #since counts identical is an array indicating at each position number of identical columns or rows in the unique matrix
    # pol = list(pol)
    # uniquePol = list(uniquePol)
    index = np.argsort(countsIdentical)[::-1]
    #reordering from most to less frequent
    uniquePol = uniquePol[index]

    policyDistribution = np.sort(countsIdentical)[::-1] 
    policyDistribution = policyDistribution/np.sum(policyDistribution)

    #conversion necessary for good parsing to Q in the following analysis
    uniquePol = uniquePol.tolist()

    print(pol_values[0].size)
    
    return policyDistribution,uniquePol




#-----
n_suckers = 12
t_position = 111
sim_shape = (20,)



filename = sys.argv[1]
allPolicies = np.load(filename,allow_pickle=True)
print(len(allPolicies[0]))
isHive = int(input("is hive?"))



distributed = False
if isHive == False:
    n_states = len(allPolicies[0][0].keys()) 
    nAgents = len(allPolicies[0])
    if nAgents == n_suckers:
        print("Distributed standard")
        print(allPolicies[0][0].keys())
        print(allPolicies[0][1].keys())
        distributed = True
        name = str(n_suckers)+"_distributed_standard"
    elif nAgents == 1:
        print("Centralized 1G")
        name = str(n_suckers)+"_1G"
    elif nAgents == 2:
        print("Centralized 2G")
        name = str(n_suckers)+"_2G_standard"
else:
    n_states = len(allPolicies[0].keys())
    if n_states==8:
        print("Distributed HIVE")
        name = str(n_suckers)+"_distributed_hive"
        distributed = True
        nAgents = 1
    elif n_states==(2**5):
        print("Centralized 2G Hive")
        name = str(n_suckers)+"_2G_hive"
        nAgents = 2

print("number of states=", n_states)
print("number of agents =",nAgents)
# ESTABLISH UNIQUE
print("\n ASSESSING UNIQUE POLICIES\n")
_policyDistribution,uniquePolicies = countPolicies(allPolicies,isHive,distributed)
nPolicies = len(uniquePolicies)
print("number unique policies = ",nPolicies)
print("\n")
#####


#COMPUTE VELOCITIES AND SORT -- > GET REF POLICY

# establish env and Q 
env = Environment(n_suckers,sim_shape,t_position,omega =0.1,isOverdamped=True,is_Ganglia= (not distributed),nGanglia=nAgents)
Q = actionValue(env.info,hiveUpdate=isHive)

######## GET number visited states random policy:

_vel,_state_freq,visitedRandom = Q.evaluateTrivialPolicy(env,isRandom=True)
RPvisits = len(visitedRandom)


###
n_DRP = 200
print("\nEvaluating Deterministic Random policy")
print("Trying %d policies"%n_DRP)

DRPvisits =[]
for i in trange(n_DRP):
    _vel,_state_freq,_norm_activeSuckers,visitedDRP_agent,visitedDRP_tentacle,_orderedAll = Q.evaluateRandomDeterministic(env,returnOrderedStates = True)
    DRPvisit = len(visitedDRP_agent)
    # DRPvisit = len(visitedDRP_tentacle)
    DRPvisits.append(DRPvisit)
DRPvisits = np.array(DRPvisits)
average_DRPvisits = np.average(DRPvisits)
average_DRPvisits_std = np.std(DRPvisits)
print("RANDOM DETERMINISTIC POL STATES: ",DRPvisits)
print("average = %.2f +- %.2f"%(average_DRPvisits,average_DRPvisits_std))
print("\n Back to trained policies..: COMPUTING VELOCITIES\n")

# input("Continue?")

normVelUnique = []
n_visitedStates = []
visitedStatesAgent =[]
visitedStatesTentacle = []
visitesStatesTentacleAll = []

for k in trange(nPolicies):
    unPol= uniquePolicies[k]
    Q.loadPolicy(unPol)
    vel,_state_freq,_norm_activeSuckers,visited,orderedVisits,orderedVisitsALL = Q.evaluatePolicy(env,returnOrderedStates = True)
    normVelUnique.append(vel)
    visitedStatesAgent.append(visited)
    # print(len(orderedVisits),orderedVisits)
    # input()
    n_visitedStates.append(len(visited)) #visited is [(agent1,state1), (agent2,state2) ecc..] unique --> so doesn't care if other agent is doing something else:counts one
    # n_visitedStates.append(len(orderedVisits)) # here is [(stateAgent1,stateAgent2) , ecc..]
    visitedStatesTentacle.append(orderedVisits)
    visitesStatesTentacleAll.append(orderedVisitsALL)
    # print(visited)
    # input()

normVelUnique = np.array(normVelUnique)
normVelUniqueRanked = np.sort(normVelUnique)[::-1]
indxRanked = np.argsort(normVelUnique)[::-1]

n_visitedStates = np.array(n_visitedStates)
n_visitedStates = n_visitedStates[indxRanked]
# visitedStatesALL = np.array(visitedStatesALL)
# visitedStatesALL = visitedStatesALL[indxRanked]
visitedStatesBestUnique = visitedStatesTentacle[indxRanked[0]]
visitedStatesBestAll = visitesStatesTentacleAll[indxRanked[0]]



print("CHECK: number visited tentacle states best=",len(visitedStatesBestUnique))
print("tentacle states:",visitedStatesBestUnique)
#new 
print("CHECK: number visited agent states best=",len(visitedStatesAgent[indxRanked[0]]))
print("agent states",visitedStatesAgent[indxRanked[0]])
###



np.save("statesBestPol_"+name+".npy",visitedStatesBestAll)
np.save("unique_statesBestPol_"+name+".npy",visitedStatesBestUnique)
#####
## SAVE BEST POLICY
print("Saving best observed policy")
bestPol = uniquePolicies[indxRanked[0]]
np.save("bestPol_"+name+".npy",bestPol)


# -------------- DATA PREPARATION FOR TREATABILITY AS MATRIX-------------

pol_values = [] #values of the ditionary, discard keys not needed
if isHive:
     #here I have one single policy F
    for t in range(nPolicies):
        pol_values.append(list(uniquePolicies[t].values()))  
else:
    for t in range(nPolicies):
        polAgent_values = []
        for a in range(nAgents):
            if distributed:
                if (a==0 or a == (nAgents-1)):
                    uniquePolicies[t][a]["dummy1"] = -1
                    uniquePolicies[t][a]["dummy2"] = -1
            polAgent_values.append(list(uniquePolicies[t][a].values()))
        pol_values.append(polAgent_values)

pol_values = np.array(pol_values) 

#in the multiagent case I have to consider the overall policy on the tentacle.
#So at each time stamp (saved policy) I considere 

# --------------------------------------

print("\n ESTABLISH DISTANCE FROM REFERENCE\n")
# Establish distance to ref policy: the highest vel one
#PROBLEM: actions for centralized are not anymore 0-1 but binary code-->
#         I have to covnert to correspondent action (option1) but then think about norm..
#         Better: just count 1 if different
pol_values = pol_values[indxRanked]
refPol_values = pol_values[0] 
print(pol_values[0])

distance =[]
matrixDiff = []
for pol in pol_values:
    diff = np.abs(refPol_values - pol)
    if not distributed:
        diff[diff!=0] = 1
    matrixDiff.append(diff)
    distance.append(np.sum(diff))
matrixDiff = np.array(matrixDiff)
distance = np.array(distance)
norm = refPol_values.size
if isHive == False and distributed:
    print("correcting norm which  was including the dummy states of base and tip for homogeneous dimensions")
    norm-=4

distanceNorm = distance/norm
distanceNorm = np.round(100*distanceNorm,2)

print("norm = ",norm)
print("check: agents x states (not valid for distributed non hive) = ",nAgents*n_states)
#I need to save one matrix per policy
outName = "distanceMatrixes_"+name+"_uniquePolDistanceFromMax.npy"
np.save(outName,matrixDiff)



outName = name+"_uniquePolDistanceFromMax.txt"
np.savetxt(outName,np.column_stack((normVelUniqueRanked,distance,distanceNorm,n_visitedStates)),fmt='%.6f\t%d\t\t%.2f\t\t%d',header = "norm vel\tdistance\tnorm distance[%%]\tvisited states", footer="total number of unique policies=%d\nAbsolute normalization = %d\nVisited states under random policy = %d\nVisited states under RDP = %.2f +- %.2f"%(nPolicies,norm,RPvisits,average_DRPvisits,average_DRPvisits_std))