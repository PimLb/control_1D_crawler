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

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
print(currentdir)
print(parentdir)
sys.path.insert(0, parentdir) 

import env
from env import Environment
from learning import actionValue

from analysis_utilities import getPolicyStats

# Leggi  da input!!!! così posso lanciare sul cluster..
#Implementa batteria di scheduling diversi come fatto da francesco? O media su stesso varie volte. 
# Oppure controlla condizioni per raggiungere pseudo plateau caso per caso?


#PLANNINFG FINALE:
# Fai convergenza con parametri stabili: epsilon min 0.001 e lr_min = 0.001:
#   Sempre scheduling di 1k episodes ma fase di plateau adattiva e (solo per i casi non hive) selettiva (chi converge prima non updata più)
#   IF non converge oltre max episodes (credo molto raro), THEN rifare con più steps
#  HP > Una volta a plateau seconda fase di esplorazione near to convergence. Dove updato a epsilon (e lr?) alti e popolo lista tutte policies trovate. --> need some code adaptation in the way old policies saved now..
#
# ATTENZIONE: l'hp di lavoro sopra potrebbe non funzionare per ganglion che sembrano più stabili del multi-agent. Se così fosse devo ripetere vari runs di apprendimento con stesso o diverso scheduling per cascare via via su policies diverse..


#PLANNING CODICE
# 1. read from input
# 2. finish setting up analysis tool on action matrix 

outFolderFigures = "figures/"
outFolderRawData = "raw/"



def train(env,Q,steps):
    pbar =tqdm(total=max_episodes)
    state = env.get_state()
    while(1):
        for k in range(steps):
            action = Q.get_action(state)
            old_state = state
            state,reward,_t = env.step(action)
            Q.update(state,old_state,action,reward)
        convergence,maxEpisodes = Q.makeGreedy()  #--> TODO: check where make sense to save previous policies if I want to do a later analysis
        env.reset_partial() #to save memory
        pbar.update(1)
        if convergence or maxEpisodes:
            break
    pbar.close()
    return convergence

def plateauExploration(Q,env,steps,lr_plateauExpl,eps_plateauExpl,n_episodes=200): 
    #NOW EXPLORATION OF PLATEAU AREA WITH HIGHER EPSILON TO GET POLICIES
    Q.lr = lr_plateauExpl
    if Q._parallelUpdate:
        Q.epsilon = eps_plateauExpl
    else:
        Q.epsilon[:] = eps_plateauExpl
    print(Q.lr,Q.epsilon)
    env.reset()
    state = env.get_state()
    for e in trange(n_episodes):
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
        if getPics:
            Q.plot_av_value(saveFig=True,labelPolicyChange = True)
    else:
        nAgents = av_values.shape[1]
        for n in nAgents:
            avValue = av_values[:,n]
            episodes = [e for e in range(avValue.size)]
            fileName = "av_value_"+name+"_agent"+str(n)+".txt"
            convInfo = "%d"%Q.nConv[n]
            footer = "Number of episodes for convergence (SINGLE AGENT)= "+convInfo+"\nCurrent time "+now
            np.savetxt(outFolder+fileName,np.column_stack((episodes,avValue)),header="episodes\tav_value",footer=footer)
        if getPics and Q.isGanglia:
            Q.plot_av_value(saveFig=True,labelPolicyChange = True,outFolder=outFolder)

###############
######
            
#           ************** MAIN **************


#######
### ALl of the following should be read from dedicated input file
sim_shape = (20,)
t_position = 41 
omega = 0.1
amplFraction = env.x0Fraction
tentacle_length = 10



max_attempts = 3
#learning param
max_episodes = 1500
min_epsilon = 0.001
min_lr = 0.001

#FINAL POLICY EXPLORATION PARAMETERS
lrPlateau = 0.0025
epsPlateau = 0.1


N_max = 2*np.pi/np.sqrt(omega)


print("max expected around (periodic tentacle limit) Ns =",N_max)

n_suckers = int(input("insert number of suckers"))

answ = int(input("insert 1 for multiagent, 0 for control center"))
answ2 = int(input("Insert 1 for HIVE UPDATE, 0 otherwise"))

if answ == 1:
    print("Each sucker is an agent")
    is_Ganglia = False
    steps = 6000
    nGanglia = 1
    if answ2 ==1:
        isHive = True
        typename="MULTIAGENT_HIVE"
    else:
        isHive=False
        typename="MULTIAGENT"
else:
    is_Ganglia = True
    steps = 20000
    nGanglia=int(input("Insert number of control centers"))
    if answ2 ==1:
        isHive = True
        typename = "%dGANGLIA_HIVE"%nGanglia
    else:
        typename = "%dGANGLIA"%nGanglia


# answ= input("Plateau expoloration? ")
# if answ:
explore_plateau = True
# else:
#     explore_plateau = False

#######################
##################
    
## Set up output file where to repeat all these on screen message

print("Setting up universe")
print("Max episodes: ",max_episodes)
print("steps x episode:", steps)

#INITIALIZATIONS:

# create out folders




env = Environment(n_suckers,sim_shape,t_position,omega=omega,tentacle_length=tentacle_length,is_Ganglia=is_Ganglia,nGanglia=nGanglia)
Q =actionValue(env.info,max_episodes=max_episodes,hiveUpdate=isHive,min_epsilon = min_epsilon,min_lr=min_lr,adaptiveScheduling = True) 

elapsed_time=[]
default_steps = steps

for attempts in range(max_attempts):
    starttime = time.perf_counter()
    convergence = train(env,Q,steps)
    endtime = time.perf_counter()
    elapsed_time.append(str(timedelta(seconds=(endtime-starttime))))
    if convergence:
        print("Convergence reached" )
        print(elapsed_time)
        #CHECK
        Q.plot_av_value(labelPolicyChange=True)
        break
    else:
        steps += int(default_steps/2)
        print("did not converge--> increasing steps: ", steps)
        #CHECK
        Q.plot_av_value(labelPolicyChange=True)
        ####
        print("Resetting Q")
        #ATTENZIONE TODO: discuss--> shall I reset or keep last obtained values?
        Q =actionValue(env.info,max_episodes=max_episodes,hiveUpdate=isHive,min_epsilon = min_epsilon,min_lr=min_lr,adaptiveScheduling = True) 
if attempts == (max_attempts-1):
    print("\n <WARNING> max attempts for convergence reached..")
    print(elapsed_time)
        
print(Q.get_value())
# GET RUNTIME INFO WITH NUMBER OF STEPS FOR EACH AGENT
if isHive:
    convInfo = str(Q.nConv)
else: 
    convInfo = str(np.amax(Q.nConv))
RuntimeInfo = elapsed_time[-1] + " Number steps per episode (final iteration): %d"%steps+" Episodes to converge (max for multi): "+convInfo

if explore_plateau:
    print("Gather policies after convergence: Plateau exploration")
    plateauExploration(Q,env,default_steps,lrPlateau,epsPlateau)
else:
    pass

if is_Ganglia or isHive:
    saveData(Q,typename,outFolder=outFolderFigures)
   # for n in numberOfTrainings..
#here other runs just to get stats on sub otpimal policies by exporing a bit in the plateau region

    #not saving for multiagent.. too many..

print("analysis of last # policies..")
bestPolIndx = getPolicyStats(Q,env,runtimeInfo=RuntimeInfo,outFolder = outFolderRawData)

print("Saving best observed policy action time series (Action Matrix)")
Q.set_referencePolicy(bestPolIndx)
A = Q.getOnpolicyActionMatrix(env)
#numpy binary format
filename = "actionTimeSeriesOnPolicy_"+typename+".npy"
np.save(filename,A)


#Now some filtering of best policies to extract for instance action time series and characterize it. In any case interesting to save at least the best policy and relative A matrix






