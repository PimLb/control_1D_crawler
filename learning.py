#TODO implement tools to monitor learning (value computation, convergence ecc)

from globals import *
import numpy as np

max_lr = 0.1# was 0.3 learning rate
min_lr = 0.0025 # was 0.05 then 0.01
gamma = 0.9#discount 0.9 to reflect upon..
max_epsilon = 0.9
min_epsilon =0.01



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
                self.dim = learning_space
                print("\n** HIVE UPDATE **\n")
                self.update = self._update_Q_parallel
                self.get_action = self._get_action_hiveUpdate
                Q = np.random.random(self.dim)
                self._Q = Q
                #convergence observables:
                self._oldQ = self._Q.copy()  
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
                self.dim = learning_space 
                self.update = self._update_Q_single
                self.get_action = self._get_action_not_hiveUpdate
                self._Q = []
                print("\n** NOT hive update **\n")
                print("A Q matrix per agent")
                self._value =[]
                self._av_value = []
                self._oldQ =[]
                for k in range(nAgents):
                    Q = np.random.random(self.dim)#each agent has Q initialized differently
                    self._Q.append(Q) 
                    self._oldQ.append(Q)
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
        


        
        self._fig_value = None
        self._fig_av_value =None
        self._fig_convergence = None

        
    def reset(self):
        self.n_episodes = 0
        self._Q = np.random.random(self.dim)
        self.epsilon = max_epsilon
        self.lr = max_lr
    
    def _get_index(self,state,action=0):
        if self._multiAgent:
            #here action is a scalar and state a 2 elements list. 
            # I need to consider dummy action argument, when I need only state
            #i = interpret_binary(state)
            i = interpret_thernary(state)
            j = action
        else:
            #to check..
            # i = interpret_binary(state)
            i = interpret_thernary(state)
            j = interpret_binary(action)
            
        return i,j
    def _update_Q_parallel(self,newstate,oldstate,action,reward):
        '''
        Update the Q function, return new action.
        Single Q which is updated by all agents.
        Conceptually it's still multiagent for how the states
        have been defined
        '''
        for  k in range(0,self._nAgents):
            s_new,_a_new = self._get_index(newstate[k]) 
            s_old,a_old = self._get_index(oldstate[k],action[k])
            self._Q[s_old,a_old] += self.lr* (reward + gamma * np.amax(self._Q[s_new]) - self._Q[s_old,a_old])

        

    def _update_Q_single(self,newstate,oldstate,action,reward):
        #update each agent Q
        for  k in range(self._nAgents):
            s_new,_a_new = self._get_index(newstate[k]) 
            s_old,a_old = self._get_index(oldstate[k],action[k])
            self._Q[k][s_old,a_old] += self.lr* (reward + gamma * np.amax(self._Q[k][s_new]) - self._Q[k][s_old,a_old])

            #This could be more efficient but applicable only in this case
            # if np.random.random() < (1 - self.epsilon):
            #     newaction.append(np.argmax(self._Q[k][s_new]))
            # else:
            #     newaction.append(np.random.randint(0,  self.action_space))
        #NOT IMPLEMENTED --> (need one for each Q) UPDATE OBSERVABLES
    
    def _get_action_hiveUpdate(self,s):
        new_action = []
        for k in range(self._nAgents):
            sind,_a = self._get_index(s[k])
            if np.random.random() < (1 - self.epsilon):
                new_action.append(np.argmax(self._Q[sind]))
            else:
                new_action.append(np.random.randint(0,self._action_space))
        return new_action
    def _get_action_not_hiveUpdate(self,s):
        #ATTENTION never tested
        new_action = []
        for k in range(self._nAgents):
            sind,_a = self._get_index(s[k])
            if np.random.random() < (1 - self.epsilon):
                new_action.append(np.argmax(self._Q[k][sind]))
            else:
                new_action.append(np.random.randint(0,self._action_space))
        return new_action
    def _get_action_singleAgent(self,s):
        #ATTENTION never tested
        #CHECK again and again
            if np.random.random() < (1 - self.epsilon):
                sind,_a = self._get_index(s)
                new_action=np.argmax(self._Q[sind])
            else:
                new_action=np.random.randint(0,  self._action_space)

            return make_binary(new_action) #need to have an instruction to be given to the tentacle (which sucker hangs)
    
    def makeGreedy(self):
        self.lr = self.scheduled_lr[self.n_episodes]
        self.epsilon = self.scheduled_epsilon[self.n_episodes]
        self.n_episodes+=1
        #UPDATE OBSERVABLES
        self._value.append(self.get_value())
        self._av_value.append(self.get_av_value())
        self._convergence.append(self._get_diff())
        
        
    
    def get_onPolicy_action(self,s):
        if self._multiAgent:
            new_action = []
            if self._parallelUpdate:
                for k in range(self._nAgents):
                    sind,_a = self._get_index(s[k])
                    # print(s[k],sind)
                    new_action.append(np.argmax(self._Q[sind]))
            else:
                #one Q function for each agent
                for k in range(self._nAgents):
                    sind,_a = self._get_index(s[k])
                    new_action.append(np.argmax(self._Q[k][sind]))
            return new_action
        else:
            #CHECK
            new_action=np.argmax(self._Q[sind])
        

            return make_binary(new_action) #need to have an instruction to be given to the tentacle (which sucker hangs)
        
    def _get_diff_hive(self):
        diff = np.amax(np.abs(self._Q -self._oldQ))
        self._oldQ = self._Q.copy()
        return diff
    
    def _get_diff_noHive(self):
        diff = []
        for k in range(self._nAgents):
            diff.append(np.amax(np.abs(self._Q[k] -self._oldQ[k])))
            self._oldQ[k] = self._Q[k].copy()
        return diff

    def _get_value_hive(self):
        return np.vstack((np.amax(self._Q,axis=1),np.argmax(self._Q,axis=1))).T
    def _get_av_value_hive(self):
        return np.mean(self._get_value_hive()[:,0])
    
    def _get_value_noHive(self):
        v = []
        for k in range(self._nAgents):
            v.append(np.vstack((np.amax(self._Q[k],axis=1),np.argmax(self._Q[k],axis=1))).T)
        return v
    def _get_av_value_noHive(self):
        avV =[]
        for k in range(self._nAgents):
            avV.append(np.mean(np.amax(self._Q[k],axis=1)))
        return avV
    
    def get_conv(self):
        return self._convergence[-1]
    

    def _plot_value_hive(self):
        plt.figure(figsize=(10, 6))
        self._fig_value = plt.subplot(xlabel='episodes', ylabel='value')
        self._fig_value.set_title(label='Value ('+str(self._state_space) + ' states)')
        sub_sampling = 10
        # last = int(self._greedySteps/sub_sampling)
        values = np.array(self._value)[0:len(self._value):sub_sampling]
        episodes = [e for e in range(0,self.n_episodes+1,sub_sampling)]
        color =['blue','red']
        actionState=[' not anchoring',' anchoring']
        stateName = ['->|<- ','->|-> ','->|tip ','<-|<- ','<-|-> ','<-|tip ','base|<- ','base|-> ']
        for i in range(self._state_space):
            c = color[int(np.array(self._value)[-1,i,1])]#]*int((self.n_episodes+1)/sub_sampling)
            a = actionState[int(np.array(self._value)[-1,i,1])]
            # print(values[-1,i,:])
            #print(int(values[values.shape[0]-1,i,1]))
            self._fig_value.plot(episodes,values[:,i,0],'-o',label=stateName[i]+a)
            # self._fig_value.plot(episodes[-last:],values[-last:,i,0],color=c)
        self._fig_value.legend()
        
        for i in range(self._state_space):
            plt.figure(figsize=(10, 6))
            self._fig_value_action_all = plt.subplot(xlabel='episodes', ylabel='action')
            self._fig_value_action_all.set_title(label='policy jumps for ' + stateName[i])
            self._fig_value_action_all.plot(episodes,values[:,i,1],'-x')
            # self._fig_value.plot(episodes[-last:],values[-last:,i,0],color=c)
        
    def _plot_value_noHive(self):
        n = input("sucker (agent) number")
        plt.figure(figsize=(10, 6))
        self._fig_value = plt.subplot(xlabel='episodes', ylabel='value')
        self._fig_value .set_title(label='Value ('+str(self._state_space) + ' states)' + 'sucker '+str(n))
        sub_sampling = 10
        episodes = [e for e in range(0,self.n_episodes+1,sub_sampling)]
        # last = int(self._greedySteps/sub_sampling)

        values = np.array(self._value[n])[0:len(self._value):sub_sampling]
        
        actionState=[' not anchoring',' anchoring']
        stateName = ['->|<- ','->|-> ','->|tip ','<-|<- ','<-|-> ','<-|tip ','base|<- ','base|-> ']
        for i in range(self._state_space):
            a = actionState[int(np.array(self._value[n])[-1,i,1])]
            self._fig_value.plot(episodes,values[:,i,0],'-o',label=stateName[i]+a)
            # self._fig_value.plot(episodes[-last:],values[-last:,i,0],color=c)
        self._fig_value.legend()     

        for i in range(self._state_space):
            plt.figure(figsize=(10, 6))
            self._fig_value_action_all = plt.subplot(xlabel='episodes', ylabel='action')
            self._fig_value_action_all.set_title(label='Sucker '+ str(n) +'. Policy jumps for ' + stateName[i])
            self._fig_value_action_all.plot(episodes,values[:,i,1],'-x')


    
    def _plot_av_value_hive(self):
        plt.figure()
        self._fig_av_value = plt.subplot(xlabel='episode', ylabel='average_value')
        self._fig_av_value.set_title(label='Average value')
        episodes = [e for e in range(self.n_episodes+1)]
        self._fig_av_value.plot(episodes,self._av_value)

    def _plot_av_value_noHive(self):
        n = input("sucker (agent) number")
        plt.figure()
        self._fig_av_value = plt.subplot(xlabel='episode', ylabel='average_value')
        self._fig_av_value.set_title(label='Average value sucker '+str(n))
        episodes = [e for e in range(self.n_episodes+1)]
        self._fig_av_value.plot(episodes,self._av_value[n])

    
    def _plot_convergence_hive(self):
        plt.figure()
        self._fig_convergence = plt.subplot(xlabel='episode', ylabel='convergence')
        self._fig_convergence.set_title(label='Global convergence Q function')
        episodes = [e for e in range(self.n_episodes)]
        self._fig_convergence.plot(episodes,self._convergence)
    
    def _plot_convergence_noHive(self):
        n = input("sucker (agent) number")
        plt.figure()
        self._fig_convergence = plt.subplot(xlabel='episode', ylabel='convergence')
        self._fig_convergence.set_title(label='Global convergence Q function for sucker  '+str(n))
        episodes = [e for e in range(self.n_episodes)]
        self._fig_convergence.plot(episodes,self._convergence)

   