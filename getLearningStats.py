#User prompts NS, omega, n_ganglia, Hive 
#--> produce files with detailed stats:  - average vel +-variance
#                                        - plots of convergence, in the title corresponding average vel + colors oscillating policies
#                                        - save on file convergence plot: both picture and numbers
#       
#                                  - save on file other policy properties: 1. how many suckers on in average ecc..

import os
import sys
import inspect

import numpy as np
import time
from datetime import datetime
from datetime import timedelta
from tqdm import tqdm
from tqdm import trange

readPeriod = False


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print(currentdir)
parentdir = currentdir.split('/')[:-1]
parentdir = '/'.join(parentdir)
print(parentdir)

sys.path.insert(0, parentdir) 



import env
from env import Environment
from learning import actionValue

from analysis_utilities import getPolicyStats
from globals import ReadInput



def train(env,Q,steps):
    # pbar =tqdm(total=max_episodes)
    state = env.get_state()
    while(1):
        for k in range(steps):
            action = Q.get_action(state)
            old_state = state
            state,reward,_t = env.step(action)
            Q.update(state,old_state,action,reward)
        convergence,maxEpisodes = Q.makeGreedy()  #--> TODO: check where make sense to save previous policies if I want to do a later analysis
        env.reset_partial() #to save memory
        # pbar.update(1)
        if convergence or maxEpisodes:
            break
    # pbar.close()
    return convergence

def plateauExploration(Q,env,steps,lr_plateauExpl,eps_plateauExpl,n_episodes=200): 
    #NOW EXPLORATION OF PLATEAU AREA WITH HIGHER EPSILON TO GET POLICIES
    Q.lr = lr_plateauExpl
    if Q._parallelUpdate:
        Q.epsilon = eps_plateauExpl
    else:
        Q.epsilon[:] = eps_plateauExpl
    print("lr and epsilon for policy exploration:",Q.lr,Q.epsilon)
    env.reset()
    state = env.get_state()
    # for e in trange(n_episodes):
    for e in range(n_episodes):
        for k in range(steps):
            action = Q.get_action(state)
            old_state = state
            state,reward,_t = env.step(action)
            Q.update(state,old_state,action,reward)
        Q.updateObs()#here I keep trace of last policies and values
    print("Plateau exporation over")
    return Q

def saveData(Q,name,getPics=True,outFolder = './'):
    '''
    Saves on file average value time series.
    Optionally images of each agent average value evolution with color for policy changes
    '''
    now = datetime.now().ctime()
    
    av_values = np.array(Q._av_value)
    if Q._parallelUpdate:
        nAgents = 1
        episodes = [e for e in range(av_values.size)]
        fileName = "av_value_"+name+".txt"
        convInfo = "%d"%Q.nConv
        footer = "Number of episodes for convergence (HIVE)= "+convInfo+"\nCurrent time "+now
        
        np.savetxt(outFolder+fileName,np.column_stack((episodes,av_values)),header="episodes\tav_value",footer=footer)
        # if getPics:
        #     Q.plot_av_value(saveFig=True,labelPolicyChange = True,outFolder=outFolder)
    else:
            nAgents = av_values.shape[1]
            for n in range(nAgents):
                avValue = av_values[:,n]
                episodes = [e for e in range(avValue.size)]
                fileName = "av_value_"+name+"_agent"+str(n)+".txt"
                convInfo = "%d"%Q.nConv[n]
                footer = "Number of episodes for convergence (SINGLE AGENT)= "+convInfo+"\nCurrent time "+now
                np.savetxt(outFolder+fileName,np.column_stack((episodes,avValue)),header="episodes\tav_value",footer=footer)

    if getPics:
        Q.plot_av_value(saveFig=True,labelPolicyChange = True,outFolder=outFolder)

###############
######
            
#           ************** MAIN **************

outFolderFigures = "figures/"
outFolderRawData = "raw/"

#CREATE FOLDERS
if not os.path.exists(outFolderFigures):
    os.makedirs(outFolderFigures)
    print("creating figures folder")

if not os.path.exists(outFolderRawData):
    os.makedirs(outFolderRawData)
    print("creating raw data folder")



# input_fileName =  sys.argv[1]
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--infile', type=argparse.FileType('r', encoding='UTF-8'), required=True)
args = parser.parse_args()
try:
    infile = args.infile
except:
     exit("Could not open input file. EXIT")

if readPeriod:
    inputParam = ReadInput(infile,adaptPeriod=False)
    period = inputParam.period
else:
    inputParam = ReadInput(infile,adaptPeriod=True)
    period = None

#CONVERGENCE PARAMETERS
max_attempts = 5
plateau_conv = inputParam.convergence
explore_plateau = True
# n_suckers = int(input("insert number of suckers\n"))
n_suckers = inputParam.ns

# answ = int(input("insert 1 for multiagent, 0 for control center\n"))
# answ2 = int(input("Insert 1 for HIVE UPDATE, 0 otherwise\n"))

#default Parameters:
sim_shape = inputParam.sim_shape
t_position = inputParam.t_position
tentacle_length = inputParam.t_length

####

omega=inputParam.omega
N_max = 2*np.pi/np.sqrt(omega)
print("max expected around (periodic tentacle limit) Ns =",N_max)

is_Ganglia = inputParam.isGanglia
max_episodes = inputParam.max_episodes
min_epsilon = inputParam.min_epsilon
min_lr = inputParam.min_lr
lrPlateau = inputParam.lr_plateau
epsPlateau = inputParam.epsilon_plateau
n_episodesPlateau = inputParam.polExplEpisodes
isHive = inputParam.isHive

if not is_Ganglia:
    print("Each sucker is an agent")
    # is_Ganglia = False
    
    nGanglia = 0
    if isHive ==1:
        # isHive = True
        steps = 6000
        typename="MULTIAGENT_HIVE"
    else:
        steps = 6000
        typename="MULTIAGENT"
else:
    # is_Ganglia = True
    steps = 20000
    # nGanglia=int(input("Insert number of control centers"))
    nGanglia = inputParam.nGanglia
    if isHive ==1 and nGanglia>1:
        # isHive = True
        typename = "%dGANGLIA_HIVE"%nGanglia
    else:
        # isHive = False
        typename = "%dGANGLIA"%nGanglia


# answ= input("Plateau expoloration? ")
# if answ:

# else:
#     explore_plateau = False

#######################
##################
    
## Set up output file where to repeat all these on screen message

print("Setting up universe")
print("Max episodes: ",max_episodes)
print("steps x episode exploration:", steps)

#INITIALIZATIONS:

# create out folders



env = Environment(n_suckers,sim_shape,t_position,omega=omega,tentacle_length=tentacle_length,is_Ganglia=is_Ganglia,nGanglia=nGanglia,period=period)
Q =actionValue(env.info,max_episodes=max_episodes,hiveUpdate=isHive,min_epsilon = min_epsilon,min_lr=min_lr,adaptiveScheduling = True,plateau_conv=plateau_conv) 

elapsed_time=[]
default_steps = steps

if is_Ganglia:
    if nGanglia ==1 :
        steps = 1000000
    if nGanglia ==2:
        if isHive:
            steps = 60000
        else:
            steps = 100000
if typename=="MULTIAGENT":
    steps = 54000

print("steps x episode training:", steps)

for attempts in range(max_attempts):
    starttime = time.perf_counter()
    convergence = train(env,Q,steps)
    endtime = time.perf_counter()
    elapsed_time.append(str(timedelta(seconds=(endtime-starttime))))
    if convergence:
        print("Convergence reached" )
        print(elapsed_time)
        #CHECK
        # Q.plot_av_value(labelPolicyChange=True)
        break
    else:
        if attempts == (max_attempts-1):
            print("\n <WARNING> max attempts for convergence reached..")
            print(elapsed_time)
            break
        steps += default_steps #int(default_steps/2)
        print("did not converge--> increasing steps: ", steps)
        #CHECK
        # Q.plot_av_value(labelPolicyChange=True)
        ####
        print("Resetting Q")
        #ATTENZIONE TODO: discuss--> shall I reset or keep last obtained values?
        Q =actionValue(env.info,max_episodes=max_episodes,hiveUpdate=isHive,min_epsilon = min_epsilon,min_lr=min_lr,adaptiveScheduling = True,plateau_conv=plateau_conv) 

#just to check..
if is_Ganglia==False and isHive==True:
    print(Q.get_value())
# GET RUNTIME INFO WITH NUMBER OF STEPS FOR EACH AGENT
if isHive:
    convInfo = str(Q.nConv)
else: 
    convInfo = str(np.amax(Q.nConv))
RuntimeInfo = "Run time ="+ elapsed_time[-1]  + " Number steps per episode (final iteration): %d"%steps+" Episodes to converge (max for multi): "+convInfo

if explore_plateau:
    print("Gather policies after convergence: Plateau exploration with %d steps"%default_steps)
    plateauExploration(Q,env,default_steps,lrPlateau,epsPlateau,n_episodes=n_episodesPlateau)
else:
    pass

#if is_Ganglia or isHive:
saveData(Q,typename,outFolder=outFolderFigures)
   # for n in numberOfTrainings..
#here other runs just to get stats on sub otpimal policies by exporing a bit in the plateau region

    #not saving for multiagent.. too many..
info = {'lr':lrPlateau,'eps':epsPlateau,'steps':default_steps,'convergence':plateau_conv}
print("analysis of last # policies..")
bestPolIndx = getPolicyStats(Q,env,info = info,runtimeInfo=RuntimeInfo,outFolder = outFolderRawData,nLastPolicies = n_episodesPlateau+1)

print("Saving best observed policy action time series (Action Matrix)")
Q.set_referencePolicy(bestPolIndx)
A = Q.getOnpolicyActionMatrix(env)
#numpy binary format
filename = "actionTimeSeriesOnPolicy_"+typename+".npy"
np.save(filename,A)
filename = "bestPolicy_"+typename+".npy"
np.save(filename,Q._refPolicy)
#Now some filtering of best policies to extract for instance action time series and characterize it. In any case interesting to save at least the best policy and relative A matrix

#TODO : 1. create figures and raw folders autonomously if they dont't exist!!
#       2. save Q matrix best policy




# DOMANDE AGNESE:
# 1. numero policies da considerare per average, uguale a quanto esploro plateau?
# 2. Per multiagent hive sembra probleamatico lr grande..