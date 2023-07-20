import numpy as np
np.seterr(invalid='ignore')
import math
import random

import pygame
import matplotlib.pyplot as plt

#some optimization..
from numba import jit





# QUESTIONS:
# In dynamics shall I pick an agent at random to evolve, or perform each time all actions for each sucker?
# In a genuine multi agent what would be the choice? 
# In a single agent framework I whoud choose an action at once which corresponds to do something on a single sucker..

red = (220, 20, 60)
blue = (30, 144, 255)
green = (0,128,0)
brown = (128,0,0)
dark_violet = (199,21,133)
violet = (219,112,147)

dt = 0.1
minDistance = 0.5
omega = 1
x0=2
amplitude = 0.8


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
        self._postion = None
        # self._position_old = None
        # could be useful
        self.leftNeighbor = left
        self.rightNeighbor = right
        self.lastAction = 0
        ####
        
        self._position = np.array(coordinate)
        # self._position_old = np.array(coordinate)

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

    def assignPointer(self, left =None, right = None,infoText = " ") -> None:
        self.leftNeighbor = left
        self.rightNeighbor = right
        self.info = infoText
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
            


def build_tentacle(n_suckers,box, exploringStarts = False):
    '''
    build tentacle with some randomicity
    '''
    # Info on space from box objec
    A = []
    offset_x = box.boundary[0]/n_suckers
    #fare in modo di avere None dove indice non esiste
    # old_position = offset_x
    if exploringStarts:
        random.seed()#uses by default system time
        if box.dimensions == 1:
            for k in range(n_suckers):
                position = offset_x + k * x0 + amplitude * random.random()
                #position = old_position + rest_position + amplitude * random.random()
                # old_position = position
                A.append(Agent(box,[position]))
        elif box.dimensions == 2:
            offset_y = box.boundary[1]/2
            for k in range(n_suckers):
                position_x = offset_x + k * x0 + amplitude * random.random()
                #position_x = old_position + rest_position + amplitude * random.random()
                #old_position = position_x
                position_y = offset_y #+ k + dt * random.random()
                A.append(Agent(box,(position_x,position_y)))
        else:
            raise ReferenceError("Simulation box badly or not initialized ?")
    else:
        if box.dimensions == 1:
            for k in range(n_suckers):
                position = offset_x + k * x0
                #position =  old_position - rest_position
                #old_position = position
                A.append(Agent(box,[position]))
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
    A[0].assignPointer(right = A[1],infoText = "I'm the base")
    A[n_suckers-1].assignPointer(left = A[n_suckers-2],infoText = "I'm the tip")
    for k in range(1,n_suckers-1):
        A[k].assignPointer(left= A[k-1],right=A[k+1],infoText = "I'm intermediate sucker n " + str(k))
    
    return A



class   Environment(object):
    def __init__(self,n_suckers,sim_shape,t_position,carrierFraction = 1,is_multiagent = True): 
        wavelengthFraction = carrierFraction
         #shape in a tuple in the form (nx,ny)
         # now t_position is only rightwall or leftwall
         # in future, target --> list of targets
        self.carrierFraction = carrierFraction
        print("Carrier modes= ",carrierFraction)
        box = Box(sim_shape)
        self._box = box
        self.isMultiagent = is_multiagent
        self._t= 0
        self._nsteps = 0
        self._episode = 1
        self._universe = {"agents":[],"target":[]}
        self._agents = self._universe["agents"]
        self._tposition = self._universe["target"]
        self._tposition.append(np.array([t_position]))
        # if np.any([self._tposition[k]>=b for k,b in enumerate(self._box.boundary.values())]):
        if np.any([self._tposition>=self._box.boundary]):
            raise ValueError("Target out of simulation box!")
        self._agents.extend(build_tentacle(n_suckers,box)) #doing so self.universe mirrors the content

        self._nsuckers=n_suckers

        if is_multiagent:
            self._nagents = n_suckers
            print("**Multiagent**")
        else:
            self._nagents = 1

        # descriptors of the dynamics and environment

        self._tip_positions= []
        self._CM_position = []
        self._telapsed = []
        self.cumulatedReward = 0

        #rendering data
        self._currentPlotCM = []
        self._currentPlotTip = []
        self._figTip = None
        self._figCM = None
        self.window = None
        self.window_size = 800
        self.clock = None
        self.metadata = { "render_fps": FPS}
        

        ###3

        # self.universe = {"agents":build_tentacle(n_suckers,box),"target":t_position}
        

        if is_multiagent == True:
            self.action_space = 2 # sucker can turn on friction or turn it off
            self.state_space = 4
        else:
            self.action_space = np.power(2,n_suckers)
            self.state_space = np.power(4,n_suckers) # Qmatrix --> self.state_space* self.action_space
    
    def reset(self,exploringStarts = True,fps = FPS):
        self._t = 0 #current time
        self._nsteps = 0
        self._telapsed =[]
        self._episode += 1
        t_position = self._tposition#keep same target
        self._universe = {"agents":[],"target":[]}
        self._agents = self._universe["agents"]
        self._tposition = self._universe["target"]
        self._tposition.extend(t_position) 
        self._agents.extend(build_tentacle(self._nagents,self._box,exploringStarts=exploringStarts))
        self._tip_positions = []
        self._CM_position = []
        self._figTip = None
        self._figCM = None
        self.cumulatedReward = 0 #total reward per episode

        if fps != FPS:
            self.window = None

        for a in self._agents:
            a.lastAction = 0
    
    
    def l0(self,t:float,k:int,N) -> float:
        # the k dependent term mimics some time delay in the propagation 
        wavelengthFraction = self.carrierFraction
        # print (wavelengthFraction)
        return x0 + amplitude*math.sin(omega*t - 2*math.pi*wavelengthFraction/N * k)
    

    def debugState(self):
        reducedStates=[]
        for k in range(1,self._nagents-1):
            dright = self._agents[k].position-self._agents[k].rightNeighbor.position
            dleft = self._agents[k].position-self._agents[k].leftNeighbor.position
            left_tension = (dleft-self.l0(self._t,k-1,self._nagents))
            right_tension = (self._agents[k].position - self._agents[k].rightNeighbor.position+self.l0(self._t,k,self._nagents))
            reducedStates.append((left_tension,right_tension))
        return reducedStates
    

    


    
    def get_state(self):
        '''
            2 states per agent depending on tension state of neighboring springs
        '''
         #returns the state of the sistem and can be framed single agent or multiagent
        multiagent = self.isMultiagent

        


        # states = []
        # for k in range(1,self._nagents-1):
            
        #     dright = self._agents[k].position-self._agents[k].rightNeighbor.position #should always be negative
        #         # print(dright)
        #     if dright>0:
        #             # print('here')
        #         dright = dright -  self._box.boundary
        #             # print(dright)
        #         #right_tension = sign(-self._agents[k].position + self._agents[k].rightNeighbor.position-l0(self._t,k,self._nagents))
        #         #to uniform the concept of compression and elongation
        #     right_tension = sign0(dright+l0(self._t,k,self._nagents))
        #     dleft = self._agents[k].position-self._agents[k].leftNeighbor.position #should always be positive
        #     if dleft<0:
        #         dleft = dleft + self._box.boundary
        #     left_tension = sign0(dleft-l0(self._t,k-1,self._nagents))
 
        #     states.append((left_tension,right_tension))





        states = []
        for k in range(self._nagents):
            if self._agents[k].leftNeighbor is None:
                #BASE
                dright = self._agents[k].position - self._agents[k].rightNeighbor.position #should always be negative
                # print(dright)
                if dright>0:
                    # print('here')
                    dright = dright -  self._box.boundary
                    # print(dright)
                #right_tension = sign(-self._agents[k].position + self._agents[k].rightNeighbor.position-self.l0(self._t,k,self._nagents))
                #to uniform the concept of compression and elongation
                right_tension = sign0(dright+self.l0(self._t,k,self._nagents))

                left_tension = 1-right_tension #not sure SEE WITH AGNESE


                states.append((left_tension,right_tension))

                continue
            else:
                dleft = self._agents[k].position - self._agents[k].leftNeighbor.position #should always be positive
                if dleft<0:
                    dleft = dleft + self._box.boundary
                left_tension = sign0(dleft-self.l0(self._t,k-1,self._nagents))
            if self._agents[k].rightNeighbor is None:
                # TIP
                right_tension = 1-left_tension #not sure SEE WITH AGNESE
            else:
                # right_tension = sign(-self._agents[k].position + self._agents[k].rightNeighbor.position-self.l0(self._t,k,self._nagents))
                #to uniform the concept of compression and elongation
                dright = self._agents[k].position - self._agents[k].rightNeighbor.position
                right_tension = sign0(dright+self.l0(self._t,k,self._nagents))
            states.append((left_tension,right_tension))
        if multiagent:
            # if not humanR:
            return states
            # else:
            #     return [ ( "elongated" if s[0]==1 else "compressed" , "elongated" if s[1] ==1 else "compressed") for s in states]
        else:
            S =[]
            #CHECK/THINK MORE
            # [s_agent_i sagent_j ] S = 2^nagents :  S = s[agent][left_tension/right_tension]s[diffagent][left/right]
            for i in range(self._nagents):
                for j in  range(i+1,self._nagents):
                    i_rev= self._nagents - (i+1)
                    j_rev = self._nagents -(j+1)
                    (states[i_rev][0]*states[j_rev][0])+S
                    S.append(states[i][1]*states[j][1])
            return S
        
    def get_humandR_state(self):
        state = self.get_state()
        return [ ( "elongated" if s[0]==1 else "compressed" , "elongated" if s[1] ==1 else "compressed") for s in state]
                
    def get_tip(self):
        return self._agents[-1].position[0]
    def get_CM(self):
        return np.average([a.position for a in self._agents])
    def get_observation(self):
        '''returns a human readable observation (position of each sucker)'''
        CM = self.get_CM()
        tip = self.get_tip()
        return self._universe | {"Center of mass":CM,"tip position":tip,"sim_time":self._t,"episode":self._episode}
    
    def step(self,action,isOverdamped = True):
        '''
        Update rule in overdamped strictly should be instantaneous--> a choice is made over updating from base to tip
        '''

        #REWRITE AVOIDING IFS ON TIP AND BASE

        elastic_constant = 0.1 #does not matter for overdamped..
        # oldState = self.get_state()
        self._telapsed.append(self._t)
        self._t += dt
        self._nsteps +=1
        #compute needed l0 values

        #must contemplate a single agent action and multiagent one.
        #To do so action is a dictionary where the key is the sucker ida
        #ATTENZIONE : EVOLUZIONE SCRITTA COSI VALE SOLO IN 1D!!
        if not isOverdamped:
            raise NameError ("Non overdamped dynamics is not implemented ")
        if self._box.dimensions == 2:
            raise NameError ("2D dynamics not implemented yet")
        
        for k in range(self._nsuckers): 
            #UPDATE FROM BASE TO TIP
            if action[k] == 1:  
                pass #Do nothing.. position unchanged
                # self._agents[k]._position = self._agents[k]._position_old #useless to check boundary conditions
            elif action[k] == 0:
                if self._agents[k].leftNeighbor is None:
                    self._agents[k].position = self._agents[k].rightNeighbor._position - self.l0(self._t,k,self._nagents)
                elif self._agents[k].rightNeighbor is None:
                    self._agents[k].position = self._agents[k].leftNeighbor._position +self.l0(self._t,k-1,self._nagents)
                else:
                    pleft = self._agents[k].leftNeighbor._position
                    pright = self._agents[k].rightNeighbor._position
                    #boundary on distances: NOT 100% sure yet..
                    if pright - pleft <0: #can happen only if boundary crossed but i expect episode to end before!
                        pright = pright + self._box.boundary
                    
                    #update on position automatically embeds boundary conditions
                    self._agents[k].position = 0.5*(pright + pleft + self.l0(self._t,k-1,self._nagents) - self.l0(self._t,k,self._nagents)) 
                 
        for k in range(self._nsuckers):
            self._agents[k].lastAction = action[k]
            # self._agents[k]._position_old = self._agents[k]._position #set old positions to new position

        terminal = False 

        touching = [(abs(a.position - self._tposition[0])<= minDistance)[0] for a in self._agents]
        #any agent touches
        # if np.any(touching):
        #only the tip touches
        
        if touching[-1]:
            print(touching)
            reward =1
            terminal = True
        try:
            advancing = self._CM_position[-1]>self._CM_position[-2]
        except IndexError:
            advancing = False
        if advancing:
            reward = 1
        else:
            reward = -1
        
        self._tip_positions.append(self.get_tip())
        self._CM_position.append(self.get_CM())

        newState = self.get_state()
        # self._newState = self.get_state()
        # self._current_reward = reward
        self.cumulatedReward += reward
        

        return  newState,reward,terminal 

    

    
    # RENDERING ROUTINES
    def plot_tip(self):
        return self._plot_tip()
    
    def _plot_tip(self):
        if self._figTip is None:
            plt.figure()
            print("initializing matplotlib plot")
            self._figTip = plt.subplot(xlabel='time steps', ylabel='tip position',
            title='Tip position, episode '+str(self._episode)) #fig,ax
            
            
        
        for l in self._currentPlotTip:
            l.remove()
        # self._figTip.cla() # erases also axis name
        # print(self._telapsed[:10],self._tip_positions[:10])
        self._currentPlotTip = self._figTip.plot(self._telapsed,self._tip_positions,linewidth=2)
        if self._box.dimensions==1:
                print(self._tposition[0])
                self._figTip.hlines(self._tposition[0][0],xmin=0,xmax=self._telapsed[-1],ls='--',color='red')
        plt.ion()
        plt.show()

    def plot_CM(self):
        return self._plot_CM()
    def _plot_CM(self):
        if self._figCM is None:
            plt.figure()
            print("initializing matplotlib plot")
            self._figCM = plt.subplot(xlabel='time steps', ylabel='CM position',
            title='CM position, episode'+str(self._episode)) #fig,ax
            
        
        # self._figCM.cla()
        # print(self._telapsed[:10],self._tip_positions[:10])
        for l in self._currentPlotCM:
            l.remove()

        self._currentPlotCM=self._figCM.plot(self._telapsed,self._CM_position,linewidth=2)
        
        plt.ion()
        plt.show()


    def get_policyView(self):
        '''This function represents the spatial behavior of the policy across the tentacle
            That is actions position instateonously
        '''
        if self._figPolicy is None:
            plt.figure()
            print("initializing matplotlib plot")
            self._figPolicy = plt.subplot(xlabel='time steps', ylabel='Friction',
            title=''+str(self._episode))


        return
        

    def render(self,special_message = None):
        
    #     #calls by default observation
    #     if self.render_mode == "rgb_array":
        # print("rendering")
        return self._render_frame(special_message)

    def _render_frame(self,special_message):
        
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
            nagents = self._nagents
            for agent in self._agents:
                n +=1
                a = agent.position
                if agent.lastAction == 0:
                    color = violet
                else:
                    color = blue
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
                    
                    l = np.array([a[0]+self.l0(self._t,n-1,self._nagents),yrepr-1])*self.pix_square_size
                    
                    a = a*self.pix_square_size
                    pygame.draw.circle(
                        canvas,
                        color,
                        a,
                        self.pix_square_size/3,
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
                                self.pix_square_size/5,
                            )
                    
                            right = np.array([agent.rightNeighbor.position[0],yrepr])* self.pix_square_size
                            lenght = math.ceil((right - a)[0])
                            # pygame.draw.line(canvas,0,a,right)
                            size = self.pix_square_size/4 *(1-n/nagents) + self.pix_square_size/5
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


    