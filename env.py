from globals import *

# import pygame
# import numpy as np

state_space_name = {(0,0):'->|<-',(0,1):'->|->',(1,0):'<-|<-',(1,1):'<-|->',('base',0):'base|<-',('base',1):'base|->' ,(0,'tip'):'->|tip',(1,'tip'):'<-|tip'}
action_space_name = {1:'anchoring', 0:'not anchoring'}
np.seterr(invalid='ignore')
# import math
# import random
#CODES TODO:
# Think about optimizing ifs 
# EASY WAY: eliminate boundary conditions
# Drop storage of some quantities, especially when getting to more complex situations (2d, multi-tentacle ecc)
# --> nice: Avoid ifs by selecting a function at once (like overdamped or not, for instance)

#IMPORTANT TO DO:
# Plot action pulse against carrier to check if peak of pulse corresponds to peak of l0 (optimal friction).
#Keep in mind we are discrete..
#Could be nice to reduce omega accordingly whem more agents to impose fixed phase velocity. PROBLEM: good delta t depends on omega
# We could show higher the number of agents--> higher velocity because we approah the continuum limit?


#OBSERVATION and HOT QUESTIOMS:
#1. In a multiagent setting and with a simple Q matrix linked to elongation/compression states there's no way of getting paper vel (?) 
#  Could this be different in single agent? (if reward adequate to drive velocity maximization)
#2. Can I distinguish crawling to grabbing by tuning reward on wall(prey) reach in a training scenario with prey close enough?


# QUESTIONS:
# In dynamics shall I pick an agent at random to evolve, or perform each time all actions for each sucker?
# In a genuine multi agent what would be the choice? 
# IMPORTANT DIFFERENCE:
#    In a single agent framework I whoud choose an action at once which corresponds to do something on a single sucker..


#PHYSICAL PARAMETERS
#scaling --> keep zeta =1 and k same order of zeta, k = zeta

zeta = 1
elastic_constant = 1 

mass = 10 # mass < 0,1 overdamped limit

x0Fraction = 4



reduced_m_inv = zeta/mass
reduced_k = elastic_constant/zeta #overdsmped limit if k>>m but dt must be 
dt = 0.1

#NEW: ASSUME k and zeta same order both in overdamped and damped

#------------



minDistance = 0.5



FPS = 20





def sign0(x)->int:
    '''
    Returns 0 for negative, 1 for positive
    CAREFUL 0 tension is given state 1. Like this state space is 2 dimensional
    '''
    # print(x)
    return int(0.5*np.sign(x) + 1)



class Box(object):
    '''
    Contains all info on universe.
    --> Box origin in the left bottom corner <--
    Provides a binning (if needed?)
        Keep in mind: 
        - bins assigned to lower boundary
    '''
    DEFAULT_BINS = 100
    MAXBINS = 10000
    @property
    def n_bins(self):
        return self._n_bins
    @n_bins.setter
    def n_bins(self,new_value):
        # print("setter")
        if self._n_bins != new_value:
            self._n_bins = new_value
            self._setBinSize()

        else:
            pass
    def _setBinSize(self):
        if self._n_bins > self.MAXBINS:
            raise ValueError ("too many bins..")
        self.dx = np.round((self.boundary[0])/self._n_bins,2)
    

        if self.dimensions == 2:
            self.dy = np.round((self.boundary[1])/self._n_bins,2)

  
    def __init__(self,in_shape,nbins = None):
        nbins = self.DEFAULT_BINS
        #TODO check shape is a tuple
        self._n_bins = nbins
        self.boundary = []
        max_x = in_shape[0]
        self.boundary.append(max_x)
        if len(in_shape) == 2:
            max_y = in_shape[1]
            self.dimensions = 2
            self.boundary.append(max_y)
            print("\nSetting up a 2D universe")
        else:
             self.dimensions = 1
             print("\nSetting up a 1D universe")
        self.boundary = np.array(self.boundary)
        self.bsize = self._n_bins * self._n_bins
        self.shape = in_shape
        print(self.boundary)
        self._setBinSize()

    # COMMENTO: più rapido un conto o un accesso memoria? (Potrei direttamente salvarmi la mappa..)
    def periodicB(self,coordinate):
        #more clever way for box with left bottom corner in the origin:
        for k,b in enumerate(self.boundary):
            coordinate[k]= coordinate[k] - math.floor(coordinate[k]/b)*b
        return coordinate
    def get_index(self,coordinate):
        """
         From coordinate assign index in the box (flattened array). 
        """
        if self.dimensions ==1:
            index = int(coordinate/self.dx)
        elif self.dimensions ==2:
            index = int(coordinate[0]/self.dx) + self._n_bins*int(coordinate[1]/self.dy)

        return index
    
    def get_position(self,index):
        """
        From flattened index gets coordinate
        """
        coordinate = np.array((self.dx * (index%self._n_bins), self.dy * index/self._n_bins))
        return np.round(coordinate,2)
    

######################
######
class Agent(object):
    def __init__(self,box:Box,coordinate,infoText = " ", left = None, right = None) -> None:
            
        self._box = box
        self.index_position= None
        # could be useful
        self.leftNeighbor = left
        self.rightNeighbor = right
        self.lastAction = 0
        self._id = None
        self._velocity_old =0
        self._acceleration_old = None
        ####
        
        self.position = np.array(coordinate)
        self._position_old = self._position.copy()
        self._abslutePosition = self._position.copy() #Store positions without boundaries effect..
        self._abslutePosition_old = self._position.copy()
        if self._position.size == self._box.dimensions:
            pass
        else:
            raise ValueError ("Incompatibility between coordinare given and universe dimension!")
        # self.index_position = box.get_index(self.position) 
        
        #raise error if out of the box
        
        # if np.any([self._position[k]>=b for k,b in enumerate(self._box.boundary.values())]):
        if np.any([self._position>=self._box.boundary]):
            raise ValueError("Out of simulation box!")
        
        if infoText:
            self.info = infoText

    def assignPointer(self,id, left =None, right = None,infoText = " ") -> None:
        self.leftNeighbor = left
        self.rightNeighbor = right
        self.info = infoText
        self._id = id
        return
    

    def reset(self,infoText = None):
        # self.index_position= None
        self._postion = None
        # self._postion_old = None
        if infoText:
            self.info = infoText

    #periodic boudaries automatically enforced when passing position
    @property
    def position(self):
        return self._position
    @position.setter
    def position(self,coordinate):
        self._position = self._box.periodicB(np.array(coordinate))

            

def build_tentacle(n_suckers,box,l0,x0,amplitude, exploringStarts = False):
    '''
    build tentacle with some randomicity
    '''
    # Info on space from box objec
    A = []
    offset_x = box.boundary[0]/(n_suckers+1) #box.boundary[0]/n_suckers
    # print("offset= ",offset_x, box.boundary[0]-offset_x)
    #fare in modo di avere None dove indice non esiste
    # old_position = offset_x
    old_position = box.boundary[0]-offset_x
    if exploringStarts:
        random.seed()#uses by default system time
        if box.dimensions == 1:
            for k in range(n_suckers):
                # position = offset_x + k * x0 + amplitude * random.random()
                position = old_position - l0(0,n_suckers-1-k) + amplitude * random.uniform(-1,1)
                #position = old_position + rest_position + amplitude * random.random()
                old_position = position
                A.append(Agent(box,[position]))
            A = A[::-1]
        elif box.dimensions == 2:
            offset_y = box.boundary[1]/2
            for k in range(n_suckers):
                position_x = offset_x + k * x0 + amplitude * random.uniform(-1,1)
                #position_x = old_position + rest_position + amplitude * random.random()
                #old_position = position_x
                position_y = offset_y #+ k + dt * random.random()
                A.append(Agent(box,(position_x,position_y)))
        else:
            raise ReferenceError("Simulation box badly or not initialized ?")
    else:
        if box.dimensions == 1:
            for k in range(n_suckers):
                # position = offset_x + k * x0
                position = old_position - l0(0,n_suckers-1-k)
                # print(position)
                old_position = position
                #position =  old_position - rest_position
                #old_position = position
                A.append(Agent(box,[position]))
            A = A[::-1]
        elif box.dimensions == 2:
            offset_y = box.boundary[1]/2
            # print("offset y",offset_y)
            for k in range(n_suckers):
                position_x = offset_x + k * x0
                # position_x = old_position + rest_position
                # old_position = position_x
                position_y = offset_y #+ k + dt * random.random()
                A.append(Agent(box,(position_x,position_y)))
        else:
            raise ReferenceError("Simulation box badly or not initialized ?")
    #Point to neighbors
    A[0].assignPointer(0,right = A[1],infoText = "I'm the base")
    A[n_suckers-1].assignPointer(n_suckers-1,left = A[n_suckers-2],infoText = "I'm the tip")
    for k in range(1,n_suckers-1):
        A[k].assignPointer(k,left= A[k-1],right=A[k+1],infoText = "I'm intermediate sucker n " + str(k))
    return A



class   Environment(object):
    def __init__(self,n_suckers,sim_shape,t_position,tentacle_length = 10,carrierMode = 1,omega=0.1,is_Ganglia = False,isOverdamped = True, nGanglia =1,period=None): 
         #control cluster refers to the number of suckers, the total number of spring involved is n_suckers-1
        
        if period is not None:
            self.N = period
            x0=tentacle_length/n_suckers
        else:
            self.N = n_suckers
            x0 = tentacle_length/self.N
        amplitude = x0/x0Fraction
        self.amplitude = amplitude
        self.x0 = x0
        self.wavelength = self.N*self.x0
        self._nsuckers=n_suckers
        self.omega = omega 
        self.tentacle_length = tentacle_length
        print("FINITE TENTACLE")
        print('x0\tamplitude\twavelength\ttentacle length\tomega')
        print('%.3f\t%.4f\t\t%.1f\t\t%.2f\t\t%.2f'%(x0,amplitude,self.wavelength,tentacle_length,omega))
        print('n suckers\tperiodicity')
        print("%d\t%d"%(self._nsuckers,self.N))
        
        self.isGanglia = is_Ganglia #need to characterize type of state returned..

        
        self._isOverdamped = isOverdamped

        self.carrierMode = carrierMode
        
       
        # print("Carrier modes= ",carrierMode)
        box = Box(sim_shape)
        self._box = box
        
        self._t= 0
        
        self._nsteps = 0
        # self._episodeSteps = totalSteps
        self._episode = 1
        self._universe = {"agents":[],"target":[]}
        self._suckers = self._universe["agents"]
        self._tposition = self._universe["target"]
        self._tposition.append(np.array([t_position]))
        # if np.any([self._tposition[k]>=b for k,b in enumerate(self._box.boundary.values())]):
        # if np.any([self._tposition>=self._box.boundary]):
        #     raise ValueError("Target out of simulation box!")
        self._suckers.extend(build_tentacle(n_suckers,box,self.l0,self.x0,self.amplitude)) #doing so self.universe mirrors the content

        
        if self._isOverdamped:
            print("OVERDAMPED DYNAMICS")
            self.step=self._stepOverdamped
            self._deltaT = dt
            print("delta t =", self.deltaT)
        else:
            print("NON OVERDAMPED:")
            print("m/zeta = ",mass/zeta)
            for sucker in self._suckers:
                sucker._acceleration_old = self._get_acceleration(sucker)
            self.step = self._stepDamped
            # self.deltaT = dt/2.
            self._deltaT = dt
            print("delta t =", self.deltaT)

        if self._box.dimensions == 2:
            raise NameError ("2D dynamics not implemented yet")
            

        
        self.inv_DeltaT = 1./self.deltaT

        self._tip_positions= []
        self._CM_position = []
        self._vel =[]
        self._telapsed = []

        self._length =[]


        self._telapsed.append(self._t)
        self._CM_position.append(self.get_CM())
        self._tip_positions.append(self.get_tip())
        self._length.append(self.get_tentacle_length())


        self.cumulatedReward = 0

        #rendering data
        self._currentPlotCM = []
        self._currentPlotTip = []
        self._figTip = None
        self._figCM = None
        self._figVel = None
        self.window = None
        self.window_size = 800
        self.clock = None
        self.metadata = { "render_fps": FPS}
        

        self.info ={}
        if is_Ganglia == False:
            # self._nagents = n_suckers
            self.get_state=self._get_state_multiagent
            print("**SUCKER AGENT**")
            # self.action_space = 2 # sucker can turn on friction or turn it off
            # self.state_space = 8#4
            # self.action_space_name = {1:'anchoring', 0:'not anchoring'} # sucker can turn on friction or turn it off
            self.action_space= 2
            # self.state_space_name = {(0,0):'->|<-',(0,1):'->|->',(1,0):'<-|<-',(1,1):'<-|->',('base',0):'base|<-',('base',1):'base|->' ,(0,'tip'):'->|tip',(1,'tip'):'<-|tip'}#4 internal + 2 tip + 2 base
            self.state_space = 8
        else:
            print("** CONTROL CENTER **")
            self.step = self._step_ganglia
            controlClusterSize = int(n_suckers/nGanglia)
            if ((n_suckers)%controlClusterSize) != 0:
                raise ValueError("number of suckers cannot be contained an integer amount of times in the control center. Choose another ganglia size!")
            print("number of suckers per ganglion =", controlClusterSize)
            if controlClusterSize< 2:
                raise ValueError("minimum control center size must be 3 corresponding to n_springs = 2")
            n_springs = controlClusterSize-1 
            self.get_state = self._get_state_controlCluster
            self.action_space = np.power(2,controlClusterSize)
            self.state_space = np.power(2,n_springs) # eg. ns =5 and 1 ganglion --> 4  springs --> 4 digit binary number to represent state
            # self.action_space_name = {1:'anchoring', 0:'not anchoring'}
            # self.state_space_name = {1:'elongated', 0:'compressed'}
            # self._nagents = int((n_suckers - 1)/controlClusterSize)
            self._nGanglia = nGanglia
            
            print("\n**Control centers mode**\n")
            print("Number of ganglia: ",self._nGanglia)
            self.info["n ganglia"] =  self._nGanglia
            self._n_springs = n_springs#springs per ganglia
            print("Springs per ganglia (--> states) = ",self._n_springs)
            #states here become spring states. Which are binary: either elongated or compressed
            #therefore there is a straigthforward conveersion into a binary mapping

        self.info["learning space"]=(self.state_space,self.action_space)
        self.info["n suckers"]=self._nsuckers
        self.info["isGanglia"] = self.isGanglia
        self.info["isPeriodic"] = False
    @property
    def isOverdamped(self):
        return self._isOverdamped
    @isOverdamped.setter
    def isOverdamped(self,overdamped):
        self._isOverdamped = overdamped
        if overdamped:
            self.step=self._stepOverdamped
        else:
            self.step=self._stepDamped
    @property
    def deltaT(self):
        return self._deltaT
    @deltaT.setter
    def deltaT(self,deltaT):
        self._deltaT=deltaT
        self.inv_DeltaT = 1./self.deltaT
    @property
    def omega(self):
        return self._omega
    @omega.setter
    def omega(self,omega):
        self._omega = omega
        k = 2*np.pi/self.N
        vp = omega/k
        #CAREFULL if carrier mode not 1 needs correction
        alpha = math.atan(self._omega/(k*k))
        vCMtheory = vp*self.amplitude*math.cos(alpha)/self.x0
        print("Optimal normalized analitical velocity periodic tentacle OVERDAMPED= ", vCMtheory)
    
    def reset(self,equilibrate = False,exploringStarts = False,fps = FPS):

        #maybe useless. I'm afraid of memory leaks..
        for s in self._suckers:
            del s

        
        t_position = self._tposition#keep same target
        self._universe = {"agents":[],"target":[]}
        self._suckers = self._universe["agents"]
        self._tposition = self._universe["target"]
        self._tposition.extend(t_position) 
        self._suckers.extend(build_tentacle(self._nsuckers,self._box,self.l0,self.x0,self.amplitude,exploringStarts=exploringStarts))
        self._episode += 1
        

        self._t = 0 #current time
        self._nsteps = 0
        self._telapsed =[]
        

        self._length =[]
        
        self._tip_positions = []
        self._CM_position = []

        self._telapsed.append(self._t)
        self._CM_position.append(self.get_CM())
        self._vel =[]
        self._tip_positions.append(self.get_tip())
        self._length.append(self.get_tentacle_length())

        
        self.cumulatedReward = 0 #total reward per episode

        if fps != FPS:
            self.window = None
        if not self._isOverdamped:
            for sucker in self._suckers:
                sucker._acceleration_old = self._get_acceleration(sucker)
        
        if equilibrate:
            self.equilibrate(1000)
        
    def reset_partial(self):
        #To reduce memory consumption for continuous problem
        for s in self._suckers:
            del s

        self._nsteps = 0
        self._telapsed = self._telapsed[-100:]
        self._episode += 1
        self._tip_positions = self._tip_positions[-100:]
        self._CM_position = self._CM_position[-100:]
        self._vel =self._vel[-100:]

        self._length = self._length[-100:]

        self.cumulatedReward = 0 #total reward per episode

    def equilibrate(self,steps):
        if self.isGanglia==False:
            action = [0]*self._nsuckers
        else:
            action = [[0]*self._nsuckers]
        for k in range(steps):
            self.step(action)
        self._episode = 1
        self._vel = []
        self._nsteps = 0
        self._telapsed =[]
        self._tip_positions = []
        self._CM_position = []
        self._length =[]
        self._telapsed.append(self._t)
        self._CM_position.append(self.get_CM())
        self._length.append(self.get_tentacle_length())

        self._tip_positions.append(self.get_tip())
    
    def l0(self,t:float,k:int) -> float:
        '''
        N = number of suckers
        '''
        # the k dependent term mimics some time delay in the propagation 
        # print (wavelengthFraction)
        # print(x0,amplitude)
        return self.x0 + self.amplitude*math.sin(self.omega*t - 2*math.pi/self.N * (k+1))
    
    
    def _get_state_controlCluster(self):
        """
        Here the states are represented by the compression state of each spring, therefore 2 states per spring.
        The whole state can be easily interpreted in binary code.
        Given a number of suckers contained in the ganglion, the number of state is ns-1.
        We consider the righthand spring to each sucker in the ganglion except last one
        """
    
        #0 compressed, 1 elongated
        #n_springs = (ns -1)/nGanglia
        clustered_springs = []
        for i in range(self._nGanglia):
            springs = []
            for sucker in  self._suckers[i*self._n_springs +i : (i+1)*self._n_springs +i ]:
                k = sucker._id
                pright = sucker.rightNeighbor._abslutePosition
                dright =  pright -sucker._abslutePosition
                right_tension = sign0(dright-self.l0(self._t,k)) # 0 = negative right force --> compressed, 1= elongated
                springs.append(right_tension)
            clustered_springs.append(springs)

        return clustered_springs

    def _get_state_multiagent(self):
        '''
            4 states for intermediary suckers, 2 for tip and base
        '''

        #BASE
        states = []
        dright = -self._suckers[0]._abslutePosition +self._suckers[0].rightNeighbor._abslutePosition
        # if dright<0:
        #     # print('here state',dright)
        #     dright +=  self._box.boundary
            # print(dright)
        right_tension = sign0(dright-self.l0(self._t,0))
        states.append(state_space_name[('base',right_tension)])

        #update old posiiton
        # self._suckers[0]._position_old = self._suckers[0].position.copy()
        # self._suckers[0]._abslutePosition_old = self._suckers[0]._abslutePosition.copy()
        #Intermediate suckers
        # for k in range(1,self._nsuckers-1):
        for sucker in self._suckers[1:self._nsuckers-1]:
            #more compact boundary enforcing
            k = sucker._id
            pright = sucker.rightNeighbor._abslutePosition
            pleft = sucker.leftNeighbor._abslutePosition
            dright = -sucker._abslutePosition + pright
            # if dright<0:
            #     dright +=  self._box.boundary
            right_tension = sign0(dright-self.l0(self._t,k))  #negative argument = pushing left (compressed)
            dleft = sucker._abslutePosition - pleft
            # if dleft<0:
            #     dleft += self._box.boundary
            left_tension = sign0(dleft-self.l0(self._t,k-1)) #negative argument = pushing right (compressed)
            states.append(state_space_name[(left_tension,right_tension)])

            # #update old posiitons
            # sucker._position_old = sucker.position.copy()
            # sucker._abslutePosition_old = sucker._abslutePosition.copy()

        #TIP
        dleft = self._suckers[self._nsuckers-1]._abslutePosition - self._suckers[self._nsuckers-1].leftNeighbor._abslutePosition
        # if dleft<0:
        #     dleft += self._box.boundary
        left_tension = sign0(dleft-self.l0(self._t,self._nsuckers-2))
        states.append(state_space_name[(left_tension,'tip')])

        #update old posiiton
        # self._suckers[self._nsuckers-1]._position_old=self._suckers[self._nsuckers-1].position.copy()
        # self._suckers[self._nsuckers-1]._abslutePosition_old = self._suckers[self._nsuckers-1]._abslutePosition.copy()


        # if multiagent:
            # if not humanR:
        return states
            # else:
            #     return [ ( "elongated" if s[0]==1 else "compressed" , "elongated" if s[1] ==1 else "compressed") for s in states]
        # else:
        #     S =[]
        #     #CHECK/THINK MORE
        #     # [s_agent_i sagent_j ] S = 2^nagents :  S = s[agent][left_tension/right_tension]s[diffagent][left/right]
        #     for i in range(self._nagents):
        #         for j in  range(i+1,self._nagents):
        #             i_rev= self._nagents - (i+1)
        #             j_rev = self._nagents -(j+1)
        #             (states[i_rev][0]*states[j_rev][0])+S
        #             S.append(states[i][1]*states[j][1])
        #     return S
        
    # def get_humandR_state(self):
    #     state = self.get_state()
    #     return [ ( "elongated" if s[0]==1 else "compressed" if s[0]==0 else "base" , "elongated" if s[1] ==1 else "compressed" if s[1]==0 else "tip") for s in state]

    # def get_stateDebug(self):
    #     #BASE
    #     states = []
    #     dright = -self._agents[0].position +self._agents[0].rightNeighbor.position
    #     right_tension = dright-self.l0(self._t,0)
    #     states.append((2,right_tension))


    #     #Intermediate suckers
    #     # for k in range(1,self._nsuckers-1):
    #     for sucker in self._agents[1:self._nsuckers-1]:
    #         #more compact boundary enforcing
    #         k = sucker._id
    #         pright = sucker.rightNeighbor.position
    #         pleft = sucker.leftNeighbor.position
    #         dright = -sucker.position + pright
    #         if dright<0:
    #             dright +=  self._box.boundary
    #         right_tension = dright-self.l0(self._t,k)  #negative argument = pushing left (compressed)
    #         dleft = sucker.position - pleft
    #         if dleft<0:
    #             dleft += self._box.boundary
    #         left_tension = dleft-self.l0(self._t,k-1) #negative argument = pushing right (compressed)
    #         states.append((left_tension,right_tension))

    #     #TIP
    #     dleft = self._agents[self._nsuckers-1].position - self._agents[self._nsuckers-1].leftNeighbor.position
    #     left_tension = dleft-self.l0(self._t,self._nsuckers-1-1)
    #     states.append((left_tension,2))

    #     return states

    def get_tip(self):
        return self._suckers[-1].position[0]
    
    def get_CM(self):
        return np.average([a._abslutePosition for a in self._suckers])
        #return 0.5*(self._agents[0].position+self._agents[-1].position)

    def get_instVel(self):
        vel = self.inv_DeltaT*(self._CM_position[-1]-self._CM_position[-2])
        
        return vel

    def get_averageVel(self):
        '''
        Returns average tentacle velocity in the current episode
        '''
        return np.average(self._vel)

    def get_tentacle_length(self):
        return (self._suckers[-1]._abslutePosition[0]-self._suckers[0]._abslutePosition[0])

    def get_observation(self):
        '''returns a human readable observation (position of each sucker)'''
        CM = self.get_CM()
        tip = self.get_tip()
        return self._universe | {"Center of mass":CM,"tip position":tip,"sim_time":self._t,"episode":self._episode}
    

    def _get_velocity(self,sucker,current_a):
        return sucker._velocity_old + 0.5*self.deltaT*(current_a+sucker._acceleration_old)


    def _get_acceleration(self,sucker):
        #.position is the current position
        #TODO remove try except which constitute additional if, but trerat explicirtly base and tip

        k = sucker._id
        try:
            pleft = sucker.leftNeighbor._position_old
            dleft = sucker._position_old - pleft
            if dleft<0:
                dleft += self._box.boundary
            left_tension = dleft-self.l0(self._t,k-1) #negative argument = pushing right (compressed)
        except:
            #BASE
            # print("base")
            left_tension =0
        try:
            pright = sucker.rightNeighbor._position_old
            dright = pright - sucker._position_old
            if dright<0:
                dright +=  self._box.boundary
            right_tension = dright-self.l0(self._t,k)  #negative argument = pushing left (compressed)
        except:
            #TIP
            right_tension = 0
        return reduced_m_inv*(reduced_k*(right_tension - left_tension)-sucker._velocity_old) 
    

    def _getReward(self):
        '''
        Computes reward and checks terminal condition 
        '''
        #velocity = 0.5/dt*(self._CM_position[-1] - self._CM_position[-2])#need several orders to be distinguishible from advancement
        # velocity2 = 1./6(self._CM_position[-1] + self._CM_position[-2] -self._CM_position[-3] - self._CM_position[-4])
        # velocityn = sum(self._CM_position[-int(len(self._CM_position)/2):]) - sum(self._CM_position[:int(len(self._CM_position)/2)])


        #TERMINAL CONDITION
        terminal = False 
        touching = [(abs(a.position - self._tposition[0])<= minDistance)[0] for a in self._suckers]


        #CM BASED
        #advancing = self._CM_position[-1]-self._CM_position[-2]
        vel = self.get_instVel()
        self._vel.append(vel)
        #TIP BASED
        # try:
        #     advancing = self._tip_positions[-1]-self._tip_positions[-2]
        # except IndexError:
        #     advancing = 0
        
        #reward = vel #numerically more stable scheme since magnitude of reward always consistent

        #ALTERNATIVE: only give -1 reward for backward and make wall reachable in training.. (so that less negative reward if episode ends)
        # reward = abs(vel)
        if vel>0:
            reward = vel #to promote higher speed..
        else:
            reward = -1
        
        # if touching[-1]:
        #     print(touching)
        #     terminal = True
            # reward = 0
            # reward = (self._episodeSteps - self._nsteps) * self.get_averageVel()
            # # reward = self.get_averageVel() * self._box.boundary[0]/self._phase_velocity
            # self.cumulatedReward += reward
            #Could it be  different in single and muilti agent?
        
        
         #cumulated reward in the episode
        self.cumulatedReward += reward
        return reward,terminal

    def _stepDamped(self,action):
        '''
        NEW: implementing damped dynamics. --> TODO: consider finite friction
        IMPORTANT: old position update upon call to get state function <--
        '''

        #TODO treat explicitly base and tip for efficiency

        #raise NameError ("Non overdamped dynamics is not implemented ")
        for sucker in self._suckers: 
            k = sucker._id
            # print(k,action[k])
            sucker.lastAction = action[k]
            # print(sucker._velocity_old,sucker._acceleration_old)
            if action[k] ==1:
                sucker._velocity_old =0
                sucker._acceleration_old =0
                continue
                #self._agents[k].position = self._agents[k]._position_old
            else:
                acceleration = self._get_acceleration(sucker) #acceleration on old positions
                sucker._acceleration_old = acceleration
                delta_x = self.deltaT*sucker._velocity_old +0.5*self.deltaT*self.deltaT* sucker._acceleration_old
                sucker.position =  sucker._position_old + delta_x
                sucker._abslutePosition = sucker._abslutePosition_old + delta_x
                
                velocity = self._get_velocity(sucker,acceleration) #new velocity
                
                sucker._velocity_old = velocity
                
       

        self._tip_positions.append(self.get_tip())
        self._CM_position.append(self.get_CM())

        self._length.append(self.get_tentacle_length())


        reward,terminal = self._getReward()
        newState = self.get_state()
        
        self._telapsed.append(self._t)
        self._t += self.deltaT
        self._nsteps +=1

        for sucker in self._suckers:
            sucker._position_old = sucker.position.copy()
            sucker._abslutePosition_old = sucker._abslutePosition.copy()
        

        return  newState,reward,terminal 

    #better to do the decoding on the side of learning
    def _step_ganglia(self,action):
        '''
        Needs to flatten all actions per ganlia in single list to pass to usual method
        Here we also explicitly update old positions. This could do more efficiently in the dedicated get state method. 
        However some thought is needed, since the loop there skips some suckers..
        '''
        action_flattened = [a for al in action for a in al]
        # print(action_flattened)
        return_value = self._stepOverdamped(action_flattened)

        #update old positions
        # for sucker in self._suckers:
        #     sucker._position_old = sucker.position.copy()
        #     sucker._abslutePosition_old = sucker._abslutePosition.copy()
        
     
        return return_value

        #Working in progress.. here the difficulty is to include scenarios with several control centers  = clustered multiagent 
        
    def _stepOverdampedVIRTUAL(self,action):
        '''
        Same as below excluding update of old positions and observables.
        Sufficient to consider absolute positions for the purpose.
        '''
        
        # #GET ALL CURRENT POSITIONS
        # positions =[]
        # for sucker in self._suckers:
        #     position.append() 
        
        if action[0] == 0:
            pright = self._suckers[0].rightNeighbor._abslutePosition_old
            dist = pright -self._suckers[0]._abslutePosition_old

            inst_vel = (dist  - self.l0(self._t,0))
            delta_x=self.deltaT * inst_vel
            # self._suckers[0].position = self._suckers[0]._position_old + delta_x
            self._suckers[0]._abslutePosition = self._suckers[0]._abslutePosition_old + delta_x
        else:
            self._suckers[0]._abslutePosition = self._suckers[0]._abslutePosition_old.copy()
        # self._suckers[0].lastAction=action[0]
        #INTERMEDIATE
        for sucker in self._suckers[1:self._nsuckers-1]:
            k = sucker._id
            # sucker.lastAction = action[k]
            if action[k] == 1:  
                sucker._abslutePosition = sucker._abslutePosition_old.copy()
            else:
                pleft = sucker.leftNeighbor._abslutePosition_old
                pright = sucker.rightNeighbor._abslutePosition_old
                me = sucker._abslutePosition_old

                dist =  pright -me
                right_force = (dist  - self.l0(self._t,k))

                dist = me - pleft
          
                left_force = -(dist - self.l0(self._t,k-1))

                inst_vel = right_force + left_force
                delta_x =  self.deltaT * inst_vel
                # sucker.position = sucker._position_old + delta_x
                sucker._abslutePosition = sucker._abslutePosition_old + delta_x#here by setter method includes boundaries
        #TIP
        if action[self._nsuckers-1] == 0:
            pleft = self._suckers[self._nsuckers-1].leftNeighbor._abslutePosition_old
            dist = self._suckers[self._nsuckers-1]._abslutePosition_old -pleft
            inst_vel = -(dist - self.l0(self._t,self._nsuckers-2))
            delta_x = self.deltaT * inst_vel
            # self._suckers[self._nsuckers-1].position = self._suckers[self._nsuckers-1]._position_old + delta_x
            self._suckers[self._nsuckers-1]._abslutePosition = self._suckers[self._nsuckers-1]._abslutePosition_old + delta_x
        else:
            self._suckers[self._nsuckers-1]._abslutePosition = self._suckers[self._nsuckers-1]._abslutePosition_old.copy()
        # self._suckers[self._nsuckers-1].lastAction=action[self._nsuckers-1]

        instVel = self.inv_DeltaT*(self.get_CM()-self._CM_position[-1]) #needs only absolute positions

        #NOT updating time

        return  instVel
    


    def _stepOverdamped(self,action):
        '''
        Update rule in overdamped strictly should be instantaneous--> a choice was made over updating from base to tip not completely correct
        NEW: imoplementing syncronous version of overdamped based on evolving previous position from finite velocity (finite friction)
        #IMPORTANT: Consider that any update on .position, enforces automatically boundary conditions
        '''
        
        if action[0] == 0: #NOT ANCHORING
            pright = self._suckers[0].rightNeighbor._abslutePosition_old
            dist = pright -self._suckers[0]._abslutePosition_old
            inst_vel = (dist  - self.l0(self._t,0))
            delta_x=self.deltaT * inst_vel
            self._suckers[0].position = self._suckers[0]._position_old + delta_x
            self._suckers[0]._abslutePosition = self._suckers[0]._abslutePosition_old + delta_x
        else: #ANCHORING
            self._suckers[0].position = self._suckers[0]._position_old.copy()
            self._suckers[0]._abslutePosition = self._suckers[0]._abslutePosition_old.copy()
        self._suckers[0].lastAction=action[0]
        #INTERMEDIATE
        for sucker in self._suckers[1:self._nsuckers-1]:
            k = sucker._id
            sucker.lastAction = action[k]
            if action[k] == 1:  #ANCHORING
                #could skip and do nothing, but want to avoid strange memory leaks by manipulating for instance stepOverdampedVIRTUAL
                sucker.position = sucker._position_old.copy()
                sucker._abslutePosition = sucker._abslutePosition_old.copy()
            else: # NOT ANCHORING
                pleft = sucker.leftNeighbor._abslutePosition_old 
                pright = sucker.rightNeighbor._abslutePosition_old
                me = sucker._abslutePosition_old

                dist =  pright -me
                right_force = (dist  - self.l0(self._t,k))

                dist = me - pleft
                left_force = -(dist - self.l0(self._t,k-1))

                inst_vel = right_force + left_force
                delta_x =  self.deltaT * inst_vel
                sucker.position = sucker._position_old + delta_x
                sucker._abslutePosition = sucker._abslutePosition_old + delta_x#here by setter method includes boundaries
        #TIP
        if action[self._nsuckers-1] == 0: #NOT ANCHORING
            pleft = self._suckers[self._nsuckers-1].leftNeighbor._abslutePosition_old
            dist = self._suckers[self._nsuckers-1]._abslutePosition_old -pleft
            inst_vel = -(dist - self.l0(self._t,self._nsuckers-2))
            delta_x = self.deltaT * inst_vel
            self._suckers[self._nsuckers-1].position = self._suckers[self._nsuckers-1]._position_old + delta_x
            self._suckers[self._nsuckers-1]._abslutePosition = self._suckers[self._nsuckers-1]._abslutePosition_old + delta_x
        else: # ANCHORING
            self._suckers[self._nsuckers-1].position = self._suckers[self._nsuckers-1]._position_old.copy()
            self._suckers[self._nsuckers-1]._abslutePosition = self._suckers[self._nsuckers-1]._abslutePosition_old.copy()

        self._suckers[self._nsuckers-1].lastAction=action[self._nsuckers-1]


        #REWARD AND TERMINAL STATE
        
        self._tip_positions.append(self.get_tip())
        self._CM_position.append(self.get_CM())

        self._length.append(self.get_tentacle_length())


        reward,terminal = self._getReward()
        newState = self.get_state()
        
        #update time

        self._telapsed.append(self._t)
        self._t += self.deltaT
        self._nsteps +=1

        #update old positions
        for sucker in self._suckers:
            sucker._position_old = sucker.position.copy()
            sucker._abslutePosition_old = sucker._abslutePosition.copy()

        return  newState,reward,terminal 

    

    
    # RENDERING ROUTINES
    def plot_tip(self):
        return self._plot_tip()
    
    def _plot_tip(self):
        if self._figTip is None:
            plt.figure()
            print("initializing matplotlib plot")
            self._figTip = plt.subplot(xlabel='time steps', ylabel='tip position') #fig,ax
            
            
        
        # for l in self._currentPlotTip:
        #     l.remove()
        
        self._figTip.set_title(label='Tip position, episode '+str(self._episode))
        self._currentPlotTip = self._figTip.plot(self._telapsed,self._tip_positions,linewidth=2)


        # if self._box.dimensions==1:
        #         print(self._tposition[0])
        #         self._figTip.hlines(self._tposition[0][0],xmin=0,xmax=self._telapsed[-1],ls='--',color='red')
        plt.ion()
        plt.show()

    def plot_CM(self):
        return self._plot_CM()
    def _plot_CM(self):
        if self._figCM is None:
            plt.figure()
            print("initializing matplotlib plot")
            self._figCM = plt.subplot(xlabel='time steps', ylabel='CM position') #fig,ax
            
        
        # self._figCM.cla()
        # print(self._telapsed[:10],self._tip_positions[:10])
        # for l in self._currentPlotCM:
        #     l.remove()
        self._figCM.set_title(label='CM position, episode '+str(self._episode))
        self._currentPlotCM=self._figCM.plot(self._telapsed,self._CM_position,linewidth=2)
        
        plt.ion()
        plt.show()
    def plot_instVel(self):
        return self._plot_instVel()
    def _plot_instVel(self):
        if self._figVel is None:
            plt.figure()
            print("initializing matplotlib plot")
            self._figVel = plt.subplot(xlabel='time steps', ylabel='vel') #fig,ax
            
        
        # self._figCM.cla()
        # print(self._telapsed[:10],self._tip_positions[:10])
        # for l in self._currentPlotCM:
        #     l.remove()
        self._figVel.set_title(label='instantaneous velocity, episode '+str(self._episode))
        self._currentPlotVel=self._figVel.plot(self._telapsed[1:],self._vel,linewidth=2)
        
        plt.ion()
        plt.show()

    def _plot_length(self):
        if self._figLength is None:
            plt.figure()
            print("initializing matplotlib plot")
            self._figLength= plt.subplot(xlabel='time steps', ylabel='CM position')
        self._figLength.set_title(label='tentacle length, episode '+str(self._episode))
        self._figLength.plot(self._telapsed,self._length,linewidth=2)
        
        plt.ion()
        plt.show()
    def plot_length(self):
        return self._plot_length()
    




    def render(self,colored_suckers=None,special_message = None):
        
    #     #calls by default observation
    #     if self.render_mode == "rgb_array":
        # print("rendering")
        if colored_suckers is None:
            colored_suckers = {}
        return self._render_frame(special_message,colored_suckers)

    def _render_frame(self,special_message,colored_suckers):
        
        if self.window is None:
            npixels = np.amax(self._box.boundary)
            print(npixels)
            print("Setting up rendering")
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.pix_square_size = (
            self.window_size / npixels
            )
            print("Pixel size: ",self.pix_square_size)
            if self.clock is None:
                self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
    
        if special_message is None:
            pygame.display.set_caption('T= '+ str(np.round(self._t,2)) +' episode= '+ str(self._episode))
        else: 
            pygame.display.set_caption(special_message)

        for target_location in self._tposition:
            
            try:
                target_location[1]
                pygame.draw.circle(
                canvas,
                green,
                self.pix_square_size*target_location, #rescaling to window size
                self.pix_square_size/3
                )
            except IndexError:
                target_location = np.array([target_location[0],0])
                target_location = self.pix_square_size*target_location#rescaling to window size
                pygame.draw.rect(
                canvas,
                green,
                pygame.Rect(
                    target_location[0], target_location[1],
                    self.pix_square_size, self._box.boundary[0]*self.pix_square_size,
                ),
                )
            
            
#         # Now we draw the agent
#         # loop on all suckers
            n = 0
            nagents = self._nsuckers
            for indx,agent in enumerate(self._suckers):
                n +=1
                a = agent.position
                if agent.lastAction == 0:
                    color = violet
                else:
                    color = blue
                    if indx in colored_suckers:
                        color = red
                try:
                    a[1]
                    pygame.draw.circle(
                    canvas,
                    (0, 0, 255),
                    a[0],a[1],
                    self.pix_square_size/3
                    )
                    if agent.rightNeighbor.position is not None:
                        pygame.draw.line(canvas,0,a,agent.rightNeighbor.position * self.pix_square_size)
                except IndexError:
                    yrepr = self._box.boundary[0]/2
                    a = np.array([a[0],yrepr])
                    
                    # print(a)
                    # pygame.draw.rect(
                    #     canvas,
                    #     color,
                    #     pygame.Rect(
                    #     a[0],a[1]-self.pix_square_size/2,
                    #     self.pix_square_size/2,self.pix_square_size
                    # )
                    # )
                    #draw l0 position
                    
                    l = np.array([a[0]+self.l0(self._t,n-1),yrepr-1])*self.pix_square_size
                    
                    a = a*self.pix_square_size
                    pygame.draw.circle(
                        canvas,
                        color,
                        a,
                        self.pix_square_size/2,
                        draw_bottom_right=True,draw_bottom_left = True
                    )
                    if agent.rightNeighbor is not None:
                        # print(agent.rightNeighbor.position)
                        if (agent.rightNeighbor.position[0] * self.pix_square_size) < a[0]:
                            pass
                        else:
                            # print(n-1)
                            #plot rest position of spring
                            pygame.draw.circle(
                                canvas,
                                red,
                                l,
                                self.pix_square_size/3,
                            )
                    
                            right = np.array([agent.rightNeighbor.position[0],yrepr])* self.pix_square_size
                            lenght = math.ceil((right - a)[0])
                            # pygame.draw.line(canvas,0,a,right)
                            size = self.pix_square_size/2 *(1-n/nagents) + self.pix_square_size/5
                            pygame.draw.rect(
                                canvas,
                                dark_violet,
                                pygame.Rect(
                                a[0],a[1]-size,
                                lenght,size
                            )
                            )

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the framerate stable.
        self.clock.tick(self.metadata["render_fps"])
    #         else:  # rgb_array
    #             return np.transpose(
    #                 np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
    #             )

    # def get_optimalAction(self):
    #     #OBS: still dependency on binning. I have to pick closer bin
    #     self.l0(self._t,k)
    # def optimalStep(self):
    #     action = get_optimalAction()




   