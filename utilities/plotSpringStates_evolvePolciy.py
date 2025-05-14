import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os
import inspect


randomAction = False

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print(currentdir)
parentdir = currentdir.split('/')[:-4]
parentdir = '/'.join(parentdir)
print(parentdir)

sys.path.insert(0, parentdir) 

from env import Environment
from learning import actionValue

n_suckers = 12
n_springs = 11
t_position = 111
sim_shape = (20,)

def make_binary(baseTen_input:int,padding:int):
    '''
    Padding adds digits with 0 in front, for a readable action instruction
    '''
    # print(padding)
    binary_num = [int(i) for i in bin(baseTen_input)[2:]]
    out = [0]*(padding-len(binary_num)) + binary_num
    return out

def readSpring(suckerStates:list,nsuckers):
    #input dictionary states
    #returns a list of 0 - 1 representing spring elongation 
    out = []
    #skip last one and always read right spring
    # print(suckerStates)
    for s in suckerStates[0:nsuckers-1]:
        # print(s)
        if s == "base|->":
            out.append(1)
        elif s == "base|<-":
            out.append(0)
        elif s=="<-|->" or s== "->|->":
            out.append(1)
        elif s=="<-|<-" or s== "->|<-":
            out.append(0)
    #     print(out)
    # input()
    return out
filename = sys.argv[1]
inputPolicy = np.load(filename,allow_pickle=True).tolist()
print(inputPolicy)
isHive = int(input("is hive?"))
 
distributed = False
if isHive == False:
    n_states = len(inputPolicy[0].keys()) 
    nAgents = len(inputPolicy)
    if nAgents == n_suckers:
        print("Distributed standard")
        print(inputPolicy[0].keys())
        print(inputPolicy[1].keys())
        distributed = True
        name = str(n_suckers)+"_distributed_standard"
    elif nAgents == 1:
        print("Centralized 1G")
        name = str(n_suckers)+"_1G"
    elif nAgents == 2:
        print("Centralized 2G")
        name = str(n_suckers)+"_2G"
else:
    n_states = len(inputPolicy.keys())
    if n_states==8:
        print("Distributed HIVE")
        name = str(n_suckers)+"_distributed_hive"
        distributed = True
        nAgents = n_suckers
    elif n_states==(2**5):
        print("Centralized 2G Hive")
        name = str(n_suckers)+"_2G_hive"
        nAgents = 2

print("number of states=", n_states)
print("number of agents =",nAgents)
env = Environment(n_suckers,sim_shape,t_position,omega =0.1,isOverdamped=True,is_Ganglia= (not distributed),nGanglia=nAgents)

if randomAction==False:
    
    
    Q = actionValue(env.info,hiveUpdate=isHive)


    Q.loadPolicy(inputPolicy)
    vel,visitedSpringState,orderedSpringStates,orderedSpringStatesALL= Q.evaluatePolicy(env,returnSpringState=True)
else:
    print("RANDOM Policy")

    Q = actionValue(env.info,hiveUpdate=isHive)
    answ = int(input("1 for Random Action, 0 for random deterministic policy\n"))

    if answ == 1:
        vel,visitedSpringState,orderedSpringStates,orderedSpringStatesALL = Q.evaluateTrivialPolicy(env,returnSpringState=True)
    elif answ == 0:
        n_checks=1
        print("averaging %d random deterministic (only for time series)"%n_checks)
        
        av_vel = []
        # list_orderedSpringStates =[]
        list_orderedSpringStatesALL = []
        for n in range(n_checks):
            vel,visitedSpringState,orderedSpringStates,orderedSpringStatesALL = Q.evaluateRandomDeterministic(env,returnSpringState=True)
            av_vel.append(vel)
            # list_orderedSpringStates.append(orderedSpringStates)
            list_orderedSpringStatesALL.append(orderedSpringStatesALL)
        # print(list_orderedSpringStates)
        # orderedSpringStates = np.average(np.array(list_orderedSpringStates),axis = 1)
        orderedSpringStates_ALL = np.average(np.array(list_orderedSpringStatesALL),axis = 1)

    else:
        print("Invalid option")
        exit()


env.plot_CM()
print("\nCHECKS:")
print("norm vel=",vel)


print("UNIQUE STATES PLOT")
states = orderedSpringStates
n_states = len(states)
print("Number of unique (all tentacle) spring states visited (by considered policy) = ", n_states)

# plot of unique traversed spring states
plt.figure()
plt.ion()
plt.xticks([0,1,2,3,4,5,6,7,8,9,10],["1","2","3","4","5","6","7","8","9","10","11"])
plt.xlabel("spring id")
plt.ylabel("unique traversed tentacle states")
plt.title("Unique spring states")

yMaxsize = 41

Z = np.empty(n_springs)
for s in states:
    Z=np.vstack((Z,s))

# To have comparable sizes..
for i in range(yMaxsize-n_states):
    dummy = np.empty(Z.shape[1])
    dummy[:]=np.nan
    Z=np.vstack((Z,dummy))
aspect = 11/n_states*1.5
Z = Z[1:] #discard first random entry
print(Z.shape)

cmap = mpl.colormaps.get_cmap('viridis')  # viridis is the default colormap for imshow
cmap.set_bad(color='white')
plt.imshow(Z,aspect=aspect)
plt.show()

print("TIME PLOT")
states = orderedSpringStatesALL
n_states = len(states)
print("Number (all tentacle) spring states visited (by considered policy) = ", n_states)

# plot of unique traversed spring states
plt.figure()
plt.ion()
plt.xticks([0,1,2,3,4,5,6,7,8,9,10],["1","2","3","4","5","6","7","8","9","10","11"])
plt.xlabel("spring id")
plt.ylabel("All traversed tentacle states")
plt.title("Time plot spring states")

Z = np.empty(n_springs)
for s in states:
    Z=np.vstack((Z,s))

aspect = 11/n_states*1.5
Z = Z[1:] #discard first random entry
print(Z.shape)
plt.imshow(Z,aspect=aspect)#,interpolation="none")

input()