
from globals import *
import numpy as np
# import copy 

max_lr = 0.1# was 0.3 learning rate
min_lr = 0.001 # was 0.05 then 0.01, for hive 0.0025
gamma = 0.9#discount 0.9 to reflect upon..
max_epsilon = 0.9
min_epsilon =0.01

stateName =['->|<-','->|->','->|tip','<-|<-','<-|->','<-|tip','base|<-','base|->']
stateMap_base = {('base',0):'base|<-',('base',1):'base|->'}
stateMap_tip = {(0,'tip'):'->|tip',(1,'tip'):'<-|tip'}
stateMap_intermediate = {(0,0):'->|<-',(0,1):'->|->',(1,0):'<-|<-',(1,1):'<-|->'}
actionState=[' not anchoring',' anchoring']

def interpret_binary(s:tuple):
    return int("".join(str(ele) for ele in s), 2)

def interpret_thernary(s:tuple):
    return int("".join(str(ele) for ele in s), 3)


def make_binary(index:int):
    return [int(i) for i in bin(index)[2:]]

class actionValue(object):
    def __init__(self,learning_space:tuple,nAgents,total_episodes,hiveUpdate = True) -> None:
        self._state_space = learning_space[0]
        self._action_space = learning_space[1]
        self._state_space_dim = len(self._state_space)
        self._action_dim = len(self._action_space)
        self.dim = (len(self._state_space),len(self._action_space))
        self._nAgents = nAgents
        self.epsilon = max_epsilon
        self.lr = max_lr
        self.discount = gamma

        scheduling_steps = total_episodes - int(total_episodes/4) #total_episodes
        print("scheduling steps =", scheduling_steps)
        print("greedy steps =", total_episodes - scheduling_steps)
        self._greedySteps = total_episodes - scheduling_steps
        self._upgrade_e = (max_epsilon-min_epsilon)/scheduling_steps
        self._upgrade_lr = (max_lr-min_lr)/scheduling_steps

        self.n_episodes = 0

        self.scheduled_epsilon = [max_epsilon-self._upgrade_e*i for i in range(scheduling_steps)] + [min_epsilon] * (total_episodes-scheduling_steps)
        self.scheduled_lr = [max_lr-self._upgrade_lr*i for i in range(scheduling_steps)] + [min_lr] * (total_episodes-scheduling_steps)

        # print(len(self.scheduled_epsilon))
        if nAgents>1:
            self._multiAgent = True
            if hiveUpdate:
                self._parallelUpdate = True
                print("\n** HIVE UPDATE **\n")
                self.update = self._update_Q_parallel
                self.get_action = self._get_action_hiveUpdate
                Q={}
                #k could be whatsover even directly the tuple of states
                #   --> Can drop the interpreter but need a function producing all possible states
                for k in self._state_space.values():
                    Q[k] = np.random.random(self.dim[1])
                # Q = np.random.random(self.dim)
                self._Q = Q
                #convergence observables:
                self._oldQ = copy.deepcopy(self._Q)
                self._convergence = []
                #VALUE
                # second index to contain correspondent policy. For 2 action space, 0 or 1 (= argmax) to be associated to color in plot
                self._value = []
                self._av_value = []
                self.get_value = self._get_value_hive
                self.get_av_value = self._get_av_value_hive
                self._get_diff = self._get_diff_hive
                self._value.append(self.get_value())
                self._av_value.append(self.get_av_value())
                #plots
                self.plot_value = self._plot_value_hive
                self.plot_convergence = self._plot_convergence_hive
                self.plot_av_value = self._plot_av_value_hive
            else:
                self._parallelUpdate = False 
                self.update = self._update_Q_single
                self.get_action = self._get_action_not_hiveUpdate
                self._Q = []
                print("\n** NOT hive update **\n")
                print("A Q matrix per agent")
                print("(however epsilon, lr and gamma are universal)")
                self._value =[]
                self._av_value = []
                self._oldQ =[]
                self._convergence = []
                
                #BASE
                Q={}
                # Q[self._state_space[4]] = np.random.random(self.dim[1])
                # Q[self._state_space[5]] = np.random.random(self.dim[1])
                base_states = stateMap_base.values()
                for k in base_states:
                    Q[k] = np.random.random(self.dim[1])
                self._Q.append(copy.deepcopy(Q)) 
                self._oldQ.append(copy.deepcopy(Q))
                Q={}
                #INTERMEDIATE ONES
                # Q[0] = np.random.random(self.dim[1])
                # Q[1] = np.random.random(self.dim[1])
                # Q[3] = np.random.random(self.dim[1])
                # Q[4] = np.random.random(self.dim[1])
                internal_states = stateMap_intermediate.values()
                for k in internal_states:
                    Q[k] = np.random.random(self.dim[1])
                for i in range(1,self._nAgents-1):
                    self._Q.append(copy.deepcopy(Q)) 
                    self._oldQ.append(copy.deepcopy(Q))
                
                #TIP
                Q={}
                tip_states = stateMap_tip.values()
                for k in tip_states:
                    Q[k] = np.random.random(self.dim[1])
                self._Q.append(copy.deepcopy(Q)) 
                self._oldQ.append(copy.deepcopy(Q))
                # Q[self._state_space[6]] = np.random.random(self.dim[1])
                # Q[self._state_space[7]] = np.random.random(self.dim[1])

                self.get_value = self._get_value_noHive
                self.get_av_value = self._get_av_value_noHive
                self._get_diff = self._get_diff_noHive
                self._value.append(self.get_value())
                self._av_value.append(self.get_av_value())
                #plots
                self.plot_value = self._plot_value_noHive
                self.plot_convergence = self._plot_convergence_noHive
                self.plot_av_value = self._plot_av_value_noHive
                

  
        else:
            #TODO populate some learning observables..
            self._multiAgent = False
            self.get_action = self._get_action_singleAgent 
            self.dim = learning_space #+ (nAgents,) not sure
    
        
    # def reset(self):
    #     self.n_episodes = 0
    #     self._Q = np.random.random(self.dim)
    #     self.epsilon = max_epsilon
    #     self.lr = max_lr
    
    def _get_index(self,state,action=0):
        # if self._multiAgent:
            #here action is a scalar and state a 2 elements list. 
            # I need to consider dummy action argument, when I need only state
            #i = interpret_binary(state)
        i = interpret_thernary(state)
        j = action
        # else:
        #     #to check..
        #     # i = interpret_binary(state)
        #     i = interpret_thernary(state)
        #     j = interpret_binary(action)
            
        return i,j
    def _update_Q_parallel(self,newstate,state,action,reward):
        '''
        Update the Q function, return new action.
        Single Q which is updated by all agents.
        Conceptually it's still multiagent for how the states
        have been defined
        '''
        for  k in range(self._nAgents):
            # s_new,_a_new = self._get_index(newstate[k]) 
            # s_old,a_old = self._get_index(oldstate[k],action[k])
            old_state = state[k]
            # print(old_state)
            old_action = action[k]
            # print(old_action)
            new_state = newstate[k]
            # print(new_state)
            self._Q[old_state][old_action] += self.lr* (reward + gamma * np.amax(self._Q[new_state]) - self._Q[old_state][old_action])

        

    def _update_Q_single(self,newstate,state,action,reward):
        #update each agent Q
        for  k in range(self._nAgents):
            # s_new,_a_new = self._get_index(newstate[k]) 
            # s_old,a_old = self._get_index(oldstate[k],action[k])Q.
            old_state = state[k]
            old_action = action[k]
            new_state = newstate[k]
            self._Q[k][old_state][old_action] += self.lr* (reward + gamma * np.amax(self._Q[k][new_state]) - self._Q[k][old_state][old_action])

            #This could be more efficient but applicable only in this case
            # if np.random.random() < (1 - self.epsilon):
            #     newaction.append(np.argmax(self._Q[k][s_new]))
            # else:
            #     newaction.append(np.random.randint(0,  self.action_space))
        #NOT IMPLEMENTED --> (need one for each Q) UPDATE OBSERVABLES
    
    def _get_action_hiveUpdate(self,state):
        new_action = []
        for k in range(self._nAgents):
            # sind,_a = self._get_index(s[k])
            if np.random.random() < (1 - self.epsilon):
                new_action.append(np.argmax(self._Q[state[k]]))
            else:
                new_action.append(np.random.randint(0,self._action_dim))
        return new_action
    def _get_action_not_hiveUpdate(self,state):
        new_action = []
        for k in range(self._nAgents):
            # sind,_a = self._get_index(s[k])
            if np.random.random() < (1 - self.epsilon):
                new_action.append(np.argmax(self._Q[k][state[k]]))
            else:
                new_action.append(np.random.randint(0,self._action_dim))
        return new_action
    def _get_action_singleAgent(self,state):
        #ATTENTION never tested
        #CHECK again and again
            if np.random.random() < (1 - self.epsilon):
                # sind,_a = self._get_index(s)
                new_action=np.argmax(self._Q[state[k]])
            else:
                new_action=np.random.randint(0,self._action_dim)

            return make_binary(new_action) #need to have an instruction to be given to the tentacle (which sucker hangs)
    
    def makeGreedy(self):
        self.lr = self.scheduled_lr[self.n_episodes]
        self.epsilon = self.scheduled_epsilon[self.n_episodes]
        self.n_episodes+=1

        #UPDATE OBSERVABLES (costly)
        self._value.append(self.get_value())
        self._av_value.append(self.get_av_value())
        self._convergence.append(self._get_diff())
        
        
    
    def get_onPolicy_action(self,state):
        if self._multiAgent:
            new_action = []
            if self._parallelUpdate:
                for k in range(self._nAgents):
                    # sind,_a = self._get_index(s[k])
                    # print(s[k],sind)
                    new_action.append(np.argmax(self._Q[state[k]]))
            else:
                #one Q function for each agent
                for k in range(self._nAgents):
                    # sind,_a = self._get_index(s[k])
                    new_action.append(np.argmax(self._Q[k][state[k]]))
            return new_action
        else:
            #CHECK
            new_action=np.argmax(self._Q[state[k]])
        

            return make_binary(new_action) #need to have an instruction to be given to the tentacle (which sucker hangs)
        
    def _get_diff_hive(self):
        diff =[]
        for k in self._Q:
            diff.append(np.abs(self._Q[k] -self._oldQ[k]))
        self._oldQ = copy.deepcopy(self._Q)
        #old implementation with Q not a dictionary
        # diff = np.amax(np.abs(self._Q -self._oldQ))
        # self._oldQ = self._Q.copy()
        return np.amax(np.array(diff))
    
    def _get_diff_noHive(self):
        diff =[]
        for i in range(self._nAgents):
            d = []
            for k in self._Q[i]:
                d.append(np.abs(self._Q[i][k] -self._oldQ[i][k]))
            
            diff.append(np.amax(np.array(d)))
            self._oldQ[i] = copy.deepcopy(self._Q[i])
        return diff

    def _get_value_hive(self):
        value = {}
        for k in self._Q:
            value[k]=(np.amax(self._Q[k]),np.argmax(self._Q[k]))
        return value
        #return np.vstack((np.amax(self._Q,axis=1),np.argmax(self._Q,axis=1))).T

    def _get_av_value_hive(self):
        value = self._get_value_hive()
        return np.mean([value[k][0] for k in value])
        # return np.mean(self._get_value_hive()[:,0])
    
    def _get_value_noHive(self):
        v = []
        # for k in range(self._nAgents):
        #     v.append(np.vstack((np.amax(self._Q[k],axis=1),np.argmax(self._Q[k],axis=1))).T)
        for i in range(self._nAgents):
            value = {}
            Q = self._Q[i]
            for k in Q:
                value[k]=(np.amax(Q[k]),np.argmax(Q[k]))
            v.append(value)
        return v
    def _get_av_value_noHive(self):
        avV =[]
        value = self._get_value_noHive()
        for k in range(self._nAgents):
            vv = value[k]
            avV.append(np.mean([vv[i][0] for i in vv]))
        return avV
    
    def get_conv(self):
        return self._convergence[-1]
    

    def _plot_value_hive(self):
        plt.figure(figsize=(10, 6))
        self._fig_value = plt.subplot(xlabel='episodes', ylabel='value')
        self._fig_value.set_title(label='Value ('+str(self._state_space_dim) + ' states)')
        sub_sampling = 5
        # last = int(self._greedySteps/sub_sampling)
        values = self._value[0:len(self._value):sub_sampling]
        episodes = [e for e in range(0,self.n_episodes+1,sub_sampling)]
        color =['blue','red']
        for i in stateName:
            a = actionState[int(self._value[-1][i][1])]
            # print(values[-1,i,:])
            #print(int(values[values.shape[0]-1,i,1]))
            polVal = [v[i][0] for v in values]
            # c = color[int(polVal[-1][1])]#]*int((self.n_episodes+1)/sub_sampling)
            self._fig_value.plot(episodes,polVal,'-o',label=i+a)

            # self._fig_value.plot(episodes[-last:],values[-last:,i,0],color=c)
        self._fig_value.legend()
        
        for i in stateName:
            plt.figure(figsize=(10, 6))
            self._fig_value_action_all = plt.subplot(xlabel='episodes', ylabel='action')
            self._fig_value_action_all.set_title(label='policy jumps for ' + i)
            polAction = [v[i][1] for v in values]
            self._fig_value_action_all.plot(episodes,polAction,'-x')
            # self._fig_value.plot(episodes[-last:],values[-last:,i,0],color=c)
        
    def _plot_value_noHive(self):
        n = int(input("sucker (agent) number"))
        stateName = self._value[0][n].keys()
        plt.figure(figsize=(10, 6))
        self._fig_value = plt.subplot(xlabel='episodes', ylabel='value')
        self._fig_value .set_title(label='Value ('+str(self._state_space_dim) + ' states)' + 'sucker '+str(n))
        sub_sampling = 10
        episodes = [e for e in range(0,self.n_episodes+1,sub_sampling)]
        # last = int(self._greedySteps/sub_sampling)

        values =  [v[n] for v in self._value[0:len(self._value):sub_sampling]]#np.array(self._value)[0:len(self._value):sub_sampling,n]
        
        

        for i in stateName:
        #     a = actionState[int(np.array(self._value)[:,n][-1,i,1])]
            a = actionState[int(values[-1][i][1])]
            polVal = [v[i][0] for v in values]
            self._fig_value.plot(episodes,polVal,'-o',label=i+a)
            # self._fig_value.plot(episodes[-last:],values[-last:,i,0],color=c)
        self._fig_value.legend()     

        for i in stateName:
            plt.figure(figsize=(10, 6))
            self._fig_value_action_all = plt.subplot(xlabel='episodes', ylabel='action')
            self._fig_value_action_all.set_title(label='Sucker '+ str(n) +'. Policy jumps for ' + i)
            polAction = [v[i][1] for v in values]
            self._fig_value_action_all.plot(episodes,polAction,'-x')


    
    def _plot_av_value_hive(self):
        plt.figure()
        self._fig_av_value = plt.subplot(xlabel='episode', ylabel='average_value')
        self._fig_av_value.set_title(label='Average value')
        episodes = [e for e in range(self.n_episodes+1)]
        self._fig_av_value.plot(episodes,self._av_value)

    def _plot_av_value_noHive(self):
        n = int(input("sucker (agent) number"))
        plt.figure()
        self._fig_av_value = plt.subplot(xlabel='episode', ylabel='average_value')
        self._fig_av_value.set_title(label='Average value sucker '+str(n))
        episodes = [e for e in range(self.n_episodes+1)]
        avValue = np.array(self._av_value)[:,n]
        self._fig_av_value.plot(episodes,avValue)

    
    def _plot_convergence_hive(self):
        plt.figure()
        self._fig_convergence = plt.subplot(xlabel='episode', ylabel='convergence')
        self._fig_convergence.set_title(label='Global convergence Q function')
        episodes = [e for e in range(self.n_episodes)]
        self._fig_convergence.plot(episodes,self._convergence)
    
    def _plot_convergence_noHive(self):
        n = int(input("sucker (agent) number"))
        plt.figure()
        self._fig_convergence = plt.subplot(xlabel='episode', ylabel='convergence')
        self._fig_convergence.set_title(label='Global convergence Q function for sucker  '+str(n))
        episodes = [e for e in range(self.n_episodes)]
        convergence = np.array(self._convergence)[:,n]
        self._fig_convergence.plot(episodes,convergence)

   