from globals import *
# import numpy as np
import analysis_utilities



max_lr = 0.1# Learning Rate at start of scheduling
# was 0.05 then 0.01, for hive 0.0025
gamma = 0.999 # Discount Factor
max_epsilon = 0.9 # Epsilon Greedy parameter at start of scheduling


stateName =['->|<-','->|->','->|tip','<-|<-','<-|->','<-|tip','base|<-','base|->']
stateMap_base = {('base',0):'base|<-',('base',1):'base|->'}
stateMap_tip = {(0,'tip'):'->|tip',(1,'tip'):'<-|tip'}
stateMap_intermediate = {(0,0):'->|<-',(0,1):'->|->',(1,0):'<-|<-',(1,1):'<-|->'}
actionState=[' not anchoring',' anchoring']


class actionValue(object):
    def __init__(self,info:dict,max_episodes=1500,hiveUpdate = True, singleActionConstraint = False,adaptiveScheduling=False,scheduling_episodes=1000,min_epsilon =0.001,min_lr = 0.001,plateau_conv=0.01) -> None:


        #Learning space a richer container
        learning_space = info["learning space"]
        self._nsuckers = info["n suckers"]
        isGanglia = info["isGanglia"]
        
        self.state_space_dim = learning_space[0]
        self.action_space_dim = learning_space[1]
        
        self.lr = max_lr
        self.discount = gamma
        self.max_episodes = max_episodes
        # scheduling_steps = total_episodes - int(total_episodes/3) #total_episodes
        print("n scheduling episodes =", scheduling_episodes)
        
        self._policyMemory = int((max_episodes - scheduling_episodes))
        self._scheduling_episodes = scheduling_episodes
        print("min epsilon =",min_epsilon)
        print("min lr =",min_lr)
        self.min_lr = min_lr
        self.min_epsilon = min_epsilon
        if adaptiveScheduling:
            print("Adaptive scheduling")
            print("maximum non scheduled episodes =", max_episodes - scheduling_episodes)
        else:
            print("non scheduled episodes =", max_episodes - scheduling_episodes)
        self._update_epsilon = (max_epsilon-min_epsilon)/scheduling_episodes
        self._update_lr = (max_lr-min_lr)/scheduling_episodes

        self.n_episodes = 0
        self._lastPolicies = []
        self._refPolicy = None
        self._av_value = []
        self._convergence = []

        self.nConv = None
        # self.scheduled_epsilon = [max_epsilon-self._upgrade_e*i for i in range(scheduling_steps)] + [min_epsilon] * (total_episodes-scheduling_steps)
        # self.scheduled_lr = [max_lr-self._upgrade_lr*i for i in range(scheduling_steps)] + [min_lr] * (total_episodes-scheduling_steps)
        
        if isGanglia ==False:
            self._singleActionConstraint = False
            self._ganglia = False
            self._nAgents = self._nsuckers 

            self.updateObs = self._updateObsSuckerAgent
            self._value =[] #this observable too messy and unreadable for ganglia. So I have it only for sucker based agents
            
            if hiveUpdate:
                self._parallelUpdate = True
                print("\n** <WARNING>:  HIVE UPDATE **\n")

                self.update = self._update_Q_parallel
                self.get_action = self._get_action_hive
                Q={}
                #   --> Can drop the interpreter but need a function producing all possible states
                
                stateSpace_all = stateMap_intermediate|stateMap_base|stateMap_tip
             
                for k in stateSpace_all.values():
                    Q[k] = np.random.random(self.action_space_dim)
                # Q = np.random.random(self.dim)
                self._Q = Q
                #convergence observables:
                self._oldQ = copy.deepcopy(self._Q)
                
                
               
                self.plot_value = self._plot_value_hive
                
            else:
                self._parallelUpdate = False 
                self.update = self._update_Q_single
                self.get_action = self._get_action_single
                self._Q = []
                
                print("\n** <WARNING>:  NOT HIVE UPDATE **\n")
                print("A Q matrix per agent")
                print("(however epsilon, lr and gamma are universal)")
    
                self._oldQ =[]
                
                #BASE
                Q={}
                base_states = stateMap_base.values()
                for k in base_states:
                    Q[k] = np.random.random(self.action_space_dim)
                self._Q.append(copy.deepcopy(Q)) 
                self._oldQ.append(copy.deepcopy(Q))
             
                Q={}
                internal_states = stateMap_intermediate.values()
                for k in internal_states:
                    Q[k] = np.random.random(self.action_space_dim)
                for i in range(1,self._nAgents-1):
                    self._Q.append(copy.deepcopy(Q)) 
                    self._oldQ.append(copy.deepcopy(Q))
                
                #TIP
                Q={}
                tip_states = stateMap_tip.values()
                for k in tip_states:
                    Q[k] = np.random.random(self.action_space_dim)
                self._Q.append(copy.deepcopy(Q)) 
                self._oldQ.append(copy.deepcopy(Q))

                
                
                #plots
                self.plot_value = self._plot_value_noHive  
                self.plot_av_value = self._plot_av_value_noHive


  
        else:   
            self._ganglia= True
            nGanglia = info["n ganglia"]

            self._nAgents = nGanglia
            self.updateObs = self._updateObsGanglia
 
            print("\n*++++++++++ Control Center (Ganglia) mode ++++++++++++++\n")
            print("Number of Ganglia = ", self._nAgents)
            print("Number of springs per ganglion considered: %d, corresponding to %d states"%((self._nsuckers)/self._nAgents-1,self.state_space_dim))

            suckers_perGanglion = int(self._nsuckers/nGanglia)
            
            if nGanglia>1 and hiveUpdate: #nGanglia>1 checked only to not print useless warning message
                self.epsilon = max_epsilon
                print("\n** <WARNING>: HIVE UPDATE **\n")
                self._parallelUpdate = True
 
                if singleActionConstraint:
                    print("\n** <WARNING>: CONSTRAINING POLICY TO 1 ANCHORING PER GANGLION AT A TIME **\n")
                    self._singleActionConstraint = True
                    self.update = self._update_Q_ganglia_constrained_hive
                    self.get_action = self._get_action_ganglia_constrained_hive
                    self.action_space_dim = suckers_perGanglion +1 #THIS IS THE MAIN FEATURE OF THIS MODE
                else:
                    self._singleActionConstraint = False
                    self.update = self._update_Q_ganglia_hive
                    self.get_action = self._get_action_ganglia_hive
                Q={}
                for k in range(self.state_space_dim):
                    Q[k] = np.random.random(self.action_space_dim)
                
                self._Q = copy.deepcopy(Q)
                self._oldQ = copy.deepcopy(Q)
                
                
                
            else:
                self._parallelUpdate = False
                
                
                if singleActionConstraint:
                    print("\n** <WARNING>: CONSTRAINING POLICY TO 1 ANCHORING PER GANGLION AT A TIME **\n")
                    self._singleActionConstraint = True
                    self.update = self._update_Q_ganglia_constrained
                    self.get_action = self._get_action_ganglia_constrained
                    self.action_space_dim = suckers_perGanglion +1 #THIS IS THE MAIN FEATURE OF THIS MODE
                else:
                    self._singleActionConstraint = False
                    self.update = self._update_Q_ganglia
                    self.get_action = self._get_action_ganglia
                self._Q = []
                self._oldQ = []
                Q={}
                for k in range(self.state_space_dim):
                    Q[k] = np.random.random(self.action_space_dim)
                for i in range(self._nAgents):
                    #1 Q matrix per control center (ganglia)
                    self._Q.append(copy.deepcopy(Q))
                    self._oldQ.append(copy.deepcopy(Q))
                
                self.plot_av_value = self._plot_av_value_ganglia
                
                
            print("possible actions combitations per ganglion = %d, for a total of %d suckers per ganglion"%(self.action_space_dim,suckers_perGanglion))

        if self._parallelUpdate:
            self.epsilon = max_epsilon
            self.get_diff = self._get_diff_hive
            self.get_value = self._get_value_hive
            self.get_av_value = self._get_av_value_hive
            self.plot_convergence = self._plot_convergence_hive
            self.plot_av_value = self._plot_av_value_hive
            if adaptiveScheduling:
                self._moving_window = 100
                self._tollerance = plateau_conv 
                self._decimalDigits = str(self._tollerance)[::-1].find('.')
                print("<WARNING>: Adaptive (hive) scheduling. Tollerance = %.3f\n"%self._tollerance)
                self.makeGreedy = self._makeGreedyAdaptive_parallel
            else:
                self.makeGreedy = self._makeGreedy_parallel
            
        else:
            self.epsilon = np.array([max_epsilon]*self._nAgents) #in principle one epsilon per agent so that I can get greedy policy selectively (not doing for lr since I can just stop updating if needed..)
            self.get_diff = self._get_diff_multiagent
            self.get_value = self._get_value
            self.get_av_value = self._get_av_value
            self.plot_convergence = self._plot_convergence_noHive
            
            # in the multiagent scenario I have in principle one epsilon per Q matrix
            # So far lr identical for all
            self._agentUpdateSet = set([a for a in range(self._nAgents)])
            if adaptiveScheduling:
                self._moving_window = 100
                self._tollerance = plateau_conv 
                self._decimalDigits = str(self._tollerance)[::-1].find('.')
                print("<WARNING>: Adaptive scheduling. Tollerance = %.3f. Convergence checked for each agent (if more than one.)\n"%self._tollerance)
                print("Rounding digits = ", self._decimalDigits)
                self.makeGreedy = self._makeGreedyAdaptive_multi
                self.nConv = np.ones(self._nAgents)*max_episodes
            else:
                self.makeGreedy = self._makeGreedy_multi

        self.updateObs()
    

    def _updateObsSuckerAgent(self):
        self.n_episodes+=1
        #VALUE
        # second index to contain correspondent policy. For 2 action space, 0 or 1 (= argmax) to be associated to color in plot
        self._value.append(self.get_value())
        self._av_value.append(self.get_av_value())
        self._convergence.append(self.get_diff())
        #TRACK LAST N POLICIES
        keep = self._policyMemory
        self._lastPolicies = self._lastPolicies[-keep:] + [self.getPolicy()]
        
        
    def _updateObsGanglia(self):
        self.n_episodes+=1
        self._av_value.append(self.get_av_value())
        self._convergence.append(self.get_diff())
        #TRACK LAST N POLICIES
        keep = self._policyMemory
        self._lastPolicies = self._lastPolicies[-keep:] + [self.getPolicy()]
   
    
    def _update_Q_parallel(self,newstate,state,action,reward):
        '''
        Update the Q function, return new action.
        Single Q which is updated by all agents (HIVE UPDATE).
        Conceptually it's still multiagent for how the states
        have been defined
        '''
        for  k in range(self._nAgents):

            old_state = state[k]
            old_action = action[k]
            new_state = newstate[k]
            self._Q[old_state][old_action] += self.lr* (reward + gamma * np.amax(self._Q[new_state]) - self._Q[old_state][old_action])

        

    def _update_Q_single(self,newstate,state,action,reward):
        '''
        1 Q matrix per agent. Each sucker-agent can develop an original policy.
        NEW: update only Q matrix of non converged agents. Use new container (a set) monitoring agents to be updated
        '''
        #update each agent Q
        for  k in self._agentUpdateSet:

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
    
    def _update_Q_ganglia(self,newstate,state,action,reward):
        '''
        Identical to Q single (with here nAgents = nGanglia) + encoding of state (which are here compression states of all the springs) + encoding of action
        '''
        action_indexes = [interpret_binary(a) for a in action] # need to get back to correct indexing
        encoded_newstate = [interpret_binary(s) for s in newstate]
        encoded_oldstate = [interpret_binary(s) for s in state]

        self._update_Q_single(encoded_newstate,encoded_oldstate,action_indexes,reward)
    
    def _update_Q_ganglia_hive(self,newstate,state,action,reward):
        '''
        Identical to Q single (with here nAgents = nGanglia) + encoding of state (which are here compression states of all the springs) + encoding of action
        '''
        action_indexes = [interpret_binary(a) for a in action] # need to get back to correct indexing
        encoded_newstate = [interpret_binary(s) for s in newstate]
        encoded_oldstate = [interpret_binary(s) for s in state]

        self._update_Q_parallel(encoded_newstate,encoded_oldstate,action_indexes,reward)
        
    def _update_Q_ganglia_constrained(self,newstate,state,action,reward):
        action_indexes = []
        for i in range(self._nAgents):
            try:
                action_indexes.append(self._nsuckers-action[i].index(1))
            except ValueError:
                action_indexes.append(0)

        encoded_newstate = [interpret_binary(s) for s in newstate]
        encoded_oldstate = [interpret_binary(s) for s in state]
        self._update_Q_single(encoded_newstate,encoded_oldstate,action_indexes,reward)

    def _update_Q_ganglia_constrained_hive(self,newstate,state,action,reward):
        action_indexes = []
        for i in range(self._nAgents):
            try:
                action_indexes.append(self._nsuckers-action[i].index(1))
            except ValueError:
                action_indexes.append(0)

        encoded_newstate = [interpret_binary(s) for s in newstate]
        encoded_oldstate = [interpret_binary(s) for s in state]
        self._update_Q_parallel(encoded_newstate,encoded_oldstate,action_indexes,reward)

    
    def _get_action_hive(self,state):
        '''Same Q matrix for each agent.
        For each agent (sucker or ganglion), outputs the action. Random number re-extracted for each one'''
        new_action = []
        for k in range(self._nAgents):
            # sind,_a = self._get_index(s[k])
            if np.random.random() < (1 - self.epsilon):
                new_action.append(np.argmax(self._Q[state[k]]))
            else:
                new_action.append(np.random.randint(0,self.action_space_dim))
        return new_action
    
    def _get_action_single(self,state):
        '''One Q matrix per agent.
        For each agent (sucker or ganglion), outputs the action. Random number re-extracted for each one'''
        new_action = []
        for k in range(self._nAgents):
            if np.random.random() < (1 - self.epsilon[k]):
                new_action.append(np.argmax(self._Q[k][state[k]]))
            else:
                new_action.append(np.random.randint(0,self.action_space_dim))
        return new_action
    
    def _get_action_ganglia(self,state):
        '''
        Identical to action single (with here nAgents = nGanglia) + encoding of state (which are here compression states of all the springs).
        Finally decoding of action, which from an integer are represented as a base 2 array positionally associated to the sucker.
        '''
        encoded_state = [interpret_binary(s) for s in state]
        # print(encoded_state)
        new_action = self._get_action_single(encoded_state)
        decoded_newaction=[]
        for i in range(self._nAgents):
            decoded_newaction.append(make_binary(new_action[i],int(self._nsuckers/self._nAgents)))

        return decoded_newaction
    
    def _get_action_ganglia_hive(self,state):
        '''
        Identical to action hive + encoding of state (which are here compression states of all the springs).
        Finally decoding of action, which from an integer are represented as a base 2 array positionally associated to the sucker.
        '''
        encoded_state = [interpret_binary(s) for s in state]
        # print(encoded_state)
        new_action = self._get_action_hive(encoded_state)
        decoded_newaction=[]
        for i in range(self._nAgents):
            decoded_newaction.append(make_binary(new_action[i],int(self._nsuckers/self._nAgents)))

        return decoded_newaction

    def _get_action_ganglia_constrained(self,state):
        """
        Like get action ganglia, but with the action space constrained to a single action (sucker anchoring) at a time.
        The only change is the decoding of actions
        """
        encoded_state = [interpret_binary(s) for s in state]
        new_action = self._get_action_single(encoded_state)
        # print(new_action)
        decoded_newaction=[]
        for i in range(self._nAgents):
            decoded_newaction.append(make_binary(int(2**(new_action[i]-1.)),int(self._nsuckers/self._nAgents)))
       
        return decoded_newaction

    def _get_action_ganglia_constrained_hive(self,state):
        """
        Like get action ganglia_hive, but with the action space constrained to a single action (sucker anchoring) at a time.
        The only change is the decoding of actions
        """
        encoded_state = [interpret_binary(s) for s in state]
        new_action = self._get_action_hive(encoded_state)
        # print(new_action)
        decoded_newaction=[]
        for i in range(self._nAgents):
            decoded_newaction.append(make_binary(int(2**(new_action[i]-1.)),int(self._nsuckers/self._nAgents)))
       
        return decoded_newaction
    

    def _makeGreedyAdaptive_parallel(self):
        '''
        Here I check convergence and stop updating if Q converged
        '''
        self.updateObs()#updates n episodes as well
        moving_window = self._moving_window 
        # self.n_episodes+=1
        isConv = False
        isMax = False
        if (self.n_episodes >= self._scheduling_episodes):
            self.lr = self.min_lr
            self.epsilon = self.min_epsilon
            # diff = abs(self._av_value[-1]-self._av_value[-2])
            av1 = np.average(np.array(self._av_value)[-moving_window:])
            av2 = np.average(np.array(self._av_value)[-2*moving_window:-moving_window])
            # diff = np.round(np.abs((av1 -av2)/((av1+av2)*0.5)),self._decimalDigits) #moving window average
            diff = np.round(np.abs((av1 -av2)/av1),self._decimalDigits)
            isConv = diff <= self._tollerance
            isMax = self.n_episodes==self.max_episodes
            if  isConv or isMax:
                print(diff)
                self.set_referencePolicy()
                self.nConv = self.n_episodes - moving_window
                
        else:
            self.lr -= self._update_lr
            self.epsilon -= self._update_epsilon
        
        return (isConv,isMax)
    
    def _makeGreedy_parallel(self):
        #UPDATE OBSERVABLES (costly)
        self.updateObs()#updates n episodes as well

        # self.n_episodes+=1
        if self.n_episodes >= self._scheduling_episodes:
            self.lr = self.min_lr
            self.epsilon = self.min_epsilon
            #terminal condition
            if (self.n_episodes==self.max_episodes):
                self.set_referencePolicy()
                return True         
        else:
            # conv = False
            self.lr -= self._update_lr
            self.epsilon -= self._update_epsilon

            return False

    
    def _makeGreedy_multi(self):
        """
        here epsilon is in principle different for each
        """
        # self.n_episodes+=1

        #UPDATE OBSERVABLES (costly)
        self.updateObs()  #updates n episodes as well

        if self.n_episodes >= self._scheduling_episodes:
            self.lr = self.min_lr
            self.epsilon[:] = self.min_epsilon
            if (self.n_episodes==self.max_episodes):
                self.set_referencePolicy()
                return True
            
        else:
            # conv = False
            self.lr -= self._update_lr
            self.epsilon -= self._update_epsilon
            return False


    def _makeGreedyAdaptive_multi(self):
        '''
        Here I check convergence and stop updating selectively converged agents. This is done by making greedy converged Q matrix (epsilon=0) and changing the _agentUpdateSet
        '''

        self.updateObs() #updates n episodes as well
        isConv = False
        isMax = False
        # self.n_episodes+=1
        moving_window = self._moving_window 
        if (self.n_episodes >= self._scheduling_episodes):
            self.lr = self.min_lr
            self.epsilon[:] = self.min_epsilon
            # diff = abs(self._av_value[-1]-self._av_value[-2])
            # diff = np.round(np.abs(np.array(self._av_value)[-1,:] - np.array(self._av_value)[-2,:]),3)
            av1 = np.average(np.array(self._av_value)[-moving_window:,:],axis=0)
            av2 = np.average(np.array(self._av_value)[-2*moving_window:-moving_window,:],axis=0)
            # diff = np.round(np.abs((av1 -av2)/((av1+av2)*0.5)),self._decimalDigits) #moving window average
            diff = np.round(np.abs((av1 -av2)/av1),self._decimalDigits)
            conv_array= diff<=self._tollerance
            isMax = self.n_episodes==self.max_episodes
            isConv = conv_array.all()
            if conv_array.any():
                # print(diff)
                toBeRemoved = set()
                for a in self._agentUpdateSet:
                    if conv_array[a]:
                        toBeRemoved.add(a)
                        self.epsilon[a] = 0
                        self.nConv[a] = self.n_episodes - moving_window
                    # self.n_episodes[a] +=1 #increase episodes of active agent
                self._agentUpdateSet = self._agentUpdateSet - toBeRemoved
                
            if isConv or isMax:
                print("End condition reached")
                print(diff)
                print(conv_array)
                print("resetting update set")
                self._agentUpdateSet = set([a for a in range(self._nAgents)])
                self.set_referencePolicy()
                

        else:
            self.lr -= self._update_lr
            self.epsilon -= self._update_epsilon
        
        return (isConv,isMax)


        
    def _get_diff_hive(self):
        diff =[]
        for k in self._Q:
            diff.append(np.abs(self._Q[k] -self._oldQ[k]))
        self._oldQ = copy.deepcopy(self._Q)
        #old implementation with Q not a dictionary
        # diff = np.amax(np.abs(self._Q -self._oldQ))
        # self._oldQ = self._Q.copy()
        return np.amax(np.array(diff))
    
    def _get_diff_multiagent(self):
        diff =[]
        for i in range(self._nAgents):
            d = []
            for k in self._Q[i]:#loop keys
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
    
    def _get_value(self):
        v = []
        for i in range(self._nAgents):
            value = {}
            Q = self._Q[i]
            for k in Q:
                value[k]=(np.amax(Q[k]),np.argmax(Q[k]))
            v.append(value)
        return v
    def _get_av_value(self):
        avV =[]
        value = self._get_value()
        for k in range(self._nAgents):
            vv = value[k]
            avV.append(np.mean([vv[i][0] for i in vv]))
        return avV
    
    def get_conv(self):
        return self._convergence[-1]
    

    def getPolicy(self):
        """
        Returns an array representing the policy.
        """
        #note that for multiagent I'm looping through dictionary keys, while for ganglia through indices
        
        # policy_vector =[] #policy is a vector of dimension #states

        # if I have more than one Q matrix there is one policy per matrix
        if self._parallelUpdate :
            policy = {}
            for k in self._Q:
                # policy_vector.append(np.argmax(self._Q[k]))
                policy[k]=np.argmax(self._Q[k])
            return policy
        else:
            policies = [] #dimension nAgents
            for i in range(self._nAgents):
                policy = {}
                Q = self._Q[i]
                for k in Q:
                    policy[k] = np.argmax(Q[k])
                    # pv.append(np.argmax(Q[k]))
                # policy_vector.append(pv)
                policies.append(policy)
            return policies
        

    def set_referencePolicy(self,n_previous=1):
        '''
        Sets policy to be followed (when on-policy) and returns correspondent average values (one per agent)
        '''
        self._refPolicy = self._lastPolicies[-n_previous]
        return self._av_value[-n_previous]
        # print("current on policy = last -%d policy"%n_previous)


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


    
    def _plot_av_value_hive(self,labelPolicyChange=False,saveFig = False,outFolder='./'):
        plt.figure()
        plt.ion()
        self._fig_av_value = plt.subplot(xlabel='episode', ylabel='average_value')
        self._fig_av_value.set_title(label='Average value (hive) learning')
        # episodes = [e for e in range(self.n_episodes+1)]
        episodes = [e for e in range(len(self._av_value))]
        self._fig_av_value.plot(episodes,self._av_value,c='black')
        if self.nConv is not None:
            self._fig_av_value.axvline(self.nConv,ls='--',c='red')
        if labelPolicyChange:
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color']
            clrs=itertools.cycle(colors)
            color = next(clrs)
            for t in range(1,len(self._lastPolicies)):
                before = np.array(list(self._lastPolicies[-t-1].values()))
                after = np.array(list(self._lastPolicies[-t].values()))
                difference =np.sum(before-after)
                if difference != 0:
                    color = next(clrs)
                

                self._fig_av_value.scatter(episodes[-t],self._av_value[-t],c=color,s=15)
        plt.pause(1)
        if saveFig:
            if self._ganglia:
                name = "avValue%dSuckers_%dGangliaHIVE.pdf"%(self._nsuckers,self._nAgents)
            else:
                name = "avValue%dSuckers_MultiagentHIVE.pdf"%self._nsuckers
            plt.savefig(outFolder+name)
    def _plot_av_value_noHive(self,labelPolicyChange=False,saveFig = False,outFolder='./'):

        if saveFig:
            for n in range(self._nAgents):
                name = "avValue%dSuckers_Multiagent_agent%d.pdf"%(self._nsuckers,n)
                plt.figure()
                plt.ion()
                self._fig_av_value = plt.subplot(xlabel='episode', ylabel='average_value')
                self._fig_av_value.set_title(label='Average value sucker '+str(n))
                # episodes = [e for e in range(self.n_episodes+1)]
                avValue = np.array(self._av_value)[:,n]
                episodes = [e for e in range(avValue.size)]

                self._fig_av_value.plot(episodes,avValue,c='black')
                if self.nConv[n] is not None:
                    self._fig_av_value.axvline(self.nConv[n],ls='--',c='red')
                if labelPolicyChange:
                    prop_cycle = plt.rcParams['axes.prop_cycle']
                    colors = prop_cycle.by_key()['color']
                    clrs = itertools.cycle(colors)
                    color = next(clrs)
                    for t in range(1,len(self._lastPolicies)):
                        before = np.array(list(self._lastPolicies[-t-1][n].values()))
                        after = np.array(list(self._lastPolicies[-t][n].values()))
                        difference =np.sum(before-after)
                        if difference != 0:
                            color = next(clrs)
                        self._fig_av_value.scatter(episodes[-t],avValue[-t],c=color,s=15)
                plt.savefig(outFolder+name)
        else:
            n = int(input("sucker (agent) number"))
            plt.figure()
            plt.ion()
            self._fig_av_value = plt.subplot(xlabel='episode', ylabel='average_value')
            self._fig_av_value.set_title(label='Average value sucker '+str(n))
            # episodes = [e for e in range(self.n_episodes+1)]
            avValue = np.array(self._av_value)[:,n]
            episodes = [e for e in range(avValue.size)]

            self._fig_av_value.plot(episodes,avValue,c='black')
            if self.nConv[n] is not None:
                self._fig_av_value.axvline(self.nConv[n],ls='--',c='red')
            if labelPolicyChange:
                prop_cycle = plt.rcParams['axes.prop_cycle']
                colors = prop_cycle.by_key()['color']
                clrs = itertools.cycle(colors)
                color = next(clrs)
                for t in range(1,len(self._lastPolicies)):
                    before = np.array(list(self._lastPolicies[-t-1][n].values()))
                    after = np.array(list(self._lastPolicies[-t][n].values()))
                    difference =np.sum(before-after)
                    if difference != 0:
                        color = next(clrs)
                self._fig_av_value.scatter(episodes[-t],avValue[-t],c=color,s=15)
        plt.pause(1)


    def _plot_av_value_ganglia(self,labelPolicyChange=False,saveFig=False,outFolder = './'):
        for n in range(self._nAgents):
            plt.figure()
            plt.ion()
            self._fig_av_value = plt.subplot(xlabel='episode', ylabel='average_value')
            self._fig_av_value.set_title(label='Average value learning for ganglion '+str(n))
            # episodes = [e for e in range(self.n_episodes+1)]
            avValue = np.array(self._av_value)[:,n]
            episodes = [e for e in range(avValue.size)]
            self._fig_av_value.plot(episodes,avValue,c='black')
            if self.nConv[n] is not None:
                self._fig_av_value.axvline(self.nConv[n],ls='--',c='red')
            if labelPolicyChange:
                # polColor = {}
                prop_cycle = plt.rcParams['axes.prop_cycle']
                colors = prop_cycle.by_key()['color']
                clrs = itertools.cycle(colors)
                color = next(clrs)
                # indx = 0
                # polColor[indx] = color
                for t in range(1,len(self._lastPolicies)):
                    before = np.array(list(self._lastPolicies[-t-1][n].values()))
                    after = np.array(list(self._lastPolicies[-t][n].values()))
                    difference =np.sum(before-after)
                    if difference != 0 :
                        color = next(clrs)
                    self._fig_av_value.scatter(episodes[-t],avValue[-t],c = color,s=15)
            plt.pause(1)
            if saveFig:
                name = "avValue%dSuckers_%dGanglia_agent%d.pdf"%(self._nsuckers,self._nAgents,n)
                plt.savefig(outFolder+name)

    def _plot_convergence_hive(self):
        plt.figure()
        self._fig_convergence = plt.subplot(xlabel='episode', ylabel='convergence')
        self._fig_convergence.set_title(label='Global convergence Q function')
        episodes = [e for e in range(self.n_episodes)]
        self._fig_convergence.plot(episodes,self._convergence)
    
    def _plot_convergence_noHive(self):
        if self._nAgents>1:
            n = int(input("Agent (sucker or ganglion) number"))
        else:
            n=0
        plt.figure()
        self._fig_convergence = plt.subplot(xlabel='episode', ylabel='convergence')
        self._fig_convergence.set_title(label='Global convergence Q function for agent  '+str(n))
        episodes = [e for e in range(self.n_episodes)]
        convergence = np.array(self._convergence)[:,n]
        self._fig_convergence.plot(episodes,convergence)


    ####################
    
    def getOnpolicyActionMatrix(self,env,timeFrame = 2000):
        '''
        Returns the time series of all action played during the execution of the active policy (the one in self._refPolicy)
        '''
        actionMatrix = np.zeros((self._nsuckers,0),int)
        env.reset()
        env.deltaT = 0.1
        env.equilibrate(1000)
        state = env.get_state()
        for k in range(timeFrame):
            action = self.getOnPolicyAction(state)
            state,r,_t=env.step(action)
            if self._ganglia:
                action = [a for al in action for a in al] #list of list --> list
            actionMatrix = np.column_stack([actionMatrix,np.array(action)])

        return actionMatrix

    def evaluatePolicy(self,env,returnOrderedStates = False,returnSpringState = False):
        """ 
            Implement some heuristic measures of the policy.
            Can consider last n policies by the argument "which" (default is the last one).
            Does not matter to be hive, since we consider this as a overall tentacle policy and not a per agent policy.
            Returns also (normalized) average CM velocity
        """

        if self._singleActionConstraint:
            exit("Not supported")

        # actionMatrix = np.zeros((self._nsuckers,0),int)
        evaluation_steps = 20000

        # self.set_referencePolicy(which)
        if returnOrderedStates == True:
            orderedVisits = []
            orderedVisitsAll =[]

        visitedStates = set()
        visitedStatesTentacle = set()
        cumulativeReward = 0
        n_activeSuckers = 0

        
        state_frequency = {}
        
        if self._ganglia==False:
            # print("MULTIAGENT")
            # if self._parallelUpdate:
            for k in stateName:
                state_frequency[k]=0

        else:
            for k in range(self.state_space_dim):
                state_frequency[k]=0

 
        actionPerState = analysis_utilities.actionMapState_dict(self._refPolicy,self._ganglia,self._parallelUpdate,self._nsuckers,self._nAgents)


        # ******** LOOP TO GATHER STATS **********
        env.reset()
        env.deltaT = 0.1
        env.equilibrate(1000)
        state = env.get_state()

        if returnSpringState:
            springStates = [] #this is ordered
            springStatesALL = [] #this is ordered
            observedSpringState = set() #unordered visits
        for k in range(evaluation_steps):
            action,encoded_state_perTentacle,encoded_state_perAgent = self.getOnPolicyAction(state,returnEncoding=True)
            state,r,_t=env.step(action)
            if returnSpringState:
                springState = env._getSpringStates() #AFTER INTEGRATION STEP
                # print(springState)
                if tuple(springState) not in observedSpringState:
                    springStates.append(springState) #independent from agent definition
                observedSpringState.add(tuple(springState))
                # print(observedSpringState)
                springStatesALL.append(springState)
             
            cumulativeReward += r
            if self._ganglia:
                action = [a for al in action for a in al] #list of list --> list
          
            n_activeSuckers += sum(action)
            for sid in encoded_state_perTentacle:
                state_frequency[sid] +=1 #not making distinction between different ganglia..
            if returnOrderedStates:

                orderedVisitsAll.append(encoded_state_perTentacle)
                if tuple(encoded_state_perTentacle) not in visitedStatesTentacle:
                    orderedVisits.append(encoded_state_perTentacle) 
            visitedStatesTentacle.add(tuple(encoded_state_perTentacle))
            for sid in encoded_state_perAgent:
                visitedStates.add(sid)

        norm_vel = env.get_averageVel()/env.x0
        

    # ************

        averageActiveSuckers = n_activeSuckers/(k+1)
        # state_frequency = state_frequency/((k+1))
        #NORMALIZATION
        state_frequency.update((key, val/(k+1)) for key, val in state_frequency.items())

        if self._ganglia==False:
            #need to correct for counting several time state for each agent (state here is still a property of each sucker)
            for s in stateMap_intermediate.values():
                state_frequency[s] = state_frequency[s]/(self._nAgents-2) #base and tip have no multiplicity
        else:
            #here state is a ganglion state with the multiplicity of the number of ganglion (one state per ganglion)
            state_frequency.update((key, val/self._nAgents) for key, val in state_frequency.items())
            # state_frequency = state_frequency/self._nAgents


        weighted_actionPerState = {key:actionPerState[key] * state_frequency[key] for key in state_frequency }
        weighted_averageActivity = sum(weighted_actionPerState.values())
        
        if returnOrderedStates:
            return norm_vel,state_frequency,averageActiveSuckers/self._nsuckers,visitedStates,orderedVisits,orderedVisitsAll
        if returnSpringState:
            return norm_vel,observedSpringState,springStates,springStatesALL
        return norm_vel,state_frequency,averageActiveSuckers/self._nsuckers,visitedStates
        


    def evaluateTrivialPolicy(self,env,isRandom=True,returnOrderedStates = False,returnSpringState = False):
        """ 
            State frequency under random policy
        """

        if isRandom:
            print("Evaluating Random policy")
            self._getTrivialAction = self._getRandomAction
        else:
            print("Evaluating Null policy")
            self._getTrivialAction = self._getNullAction

        if returnOrderedStates == True:
            orderedVisits = []
            orderedVisitsAll =[]
            visitedStatesTentacle = set()

        if returnSpringState:
            springStates = [] #this is ordered
            springStatesALL = [] #this is ordered
            observedSpringState = set() #unordered visits
        # print(env.deltaT)
        evaluation_steps = 20000

        # self.set_referencePolicy(which)
        visitedStates = set()
        cumulativeReward = 0
        n_activeSuckers = 0

        
        state_frequency = {}
        if self._ganglia==False:
            # print("MULTIAGENT")
            for k in stateName:
                state_frequency[k]=0
        else:
            # print("CONTROL CENTER")
            for k in range(self.state_space_dim):
                state_frequency[k]=0
        
        # print(state_frequency)
        
        # ******** LOOP TO GATHER STATS **********
        env.reset()
        env.equilibrate(1000)
        state = env.get_state()
        for k in range(evaluation_steps):
            action,encoded_state_perTentacle,encoded_state_perAgent = self._getTrivialAction(state,returnEncoding=True)
            state,r,_t=env.step(action)
            cumulativeReward += r
            if returnSpringState:
                springState = env._getSpringStates() #AFTER INTEGRATION STEP
                if tuple(springState) not in observedSpringState:
                    springStates.append(springState) #independent from agent definition
                observedSpringState.add(tuple(springState))
                springStatesALL.append(springState)
            if self._ganglia:
                action = [a for al in action for a in al] #list of list --> list
            n_activeSuckers += sum(action)
            for sid in encoded_state_perTentacle:
                state_frequency[sid] +=1 
            if returnOrderedStates:
                orderedVisitsAll.append(encoded_state_perTentacle)
                if tuple(encoded_state_perTentacle) not in visitedStatesTentacle:
                    orderedVisits.append(encoded_state_perTentacle) 
            for sid in encoded_state_perAgent:
                visitedStates.add(sid)
  
        norm_vel = env.get_averageVel()/env.x0
    # ************

        averageActiveSuckers = n_activeSuckers/(k+1)

        state_frequency.update((key, val/(k+1)) for key, val in state_frequency.items())
       
        if self._ganglia==False:
            for s in stateMap_intermediate.values():
                state_frequency[s] = state_frequency[s]/(self._nAgents-2) #base and tip have no multiplicity
        else:
            state_frequency.update((key, val/self._nAgents) for key, val in state_frequency.items())
           
        # print("frequency state visits:")
        # print(state_frequency)

        print("number of visited states out of all possible states:")
        print(len(visitedStates),self.state_space_dim)

        if returnOrderedStates:
            return norm_vel,state_frequency,averageActiveSuckers/self._nsuckers,visitedStates,orderedVisits,orderedVisitsAll
        if returnSpringState:
            return norm_vel,observedSpringState,springStates,springStatesALL
        
        return norm_vel,state_frequency,visitedStates
    ###############
    ##########

    def evaluateRandomDeterministic(self,env,returnOrderedStates = False,returnSpringState = False):
        """ 
            State frequency under random deterministic policy
        """

        #Build policy
        if self._parallelUpdate:
            policy = {}
            for s in self._Q:
                policy[s] = np.random.randint(0,self.action_space_dim)
        else:
            policy = []
            for a in range(self._nAgents):
                pol = {}
                for s in self._Q[a]:
                    pol[s] = np.random.randint(0,self.action_space_dim)
                policy.append(pol)
        self._refPolicy = policy
        
        # print("random deterministic policy:",policy)
        out = self.evaluatePolicy(env,returnOrderedStates=returnOrderedStates,returnSpringState = returnSpringState)
        #ripristina policy da Q matrix nel caso servisse per altre analisi
        self._refPolicy = self.getPolicy()
        return out
       
    def getOnPolicyAction(self,state,returnEncoding = False):
        out_action=[]
        if self._ganglia==False:
            encoded_state = state
            
            if self._parallelUpdate:
                out_state = encoded_state
                for k in range(self._nsuckers):
                    out_action.append(self._refPolicy[encoded_state[k]])
            else:
                out_state = [(agent,state) for agent,state in enumerate(state)]
                for k in range(self._nAgents):
                    out_action.append(self._refPolicy[k][encoded_state[k]])
            
        else:
            #GANGLIA (CONTROL CENTER) SCENARIO
            #here we need a specific binary decoding for the actions and encoding for states
            encoded_state = [interpret_binary(s) for s in state] #here state index run trhough ganglia
            padding= int(self._nsuckers/self._nAgents)
            if self._parallelUpdate:
                out_state = encoded_state
                for k in range(self._nAgents):
                    out_action.append(make_binary(self._refPolicy[encoded_state[k]],padding))
            else:
                out_state = [(agent,state) for agent,state in enumerate(encoded_state)]
                for k in range(self._nAgents):
                    # print(k,self._refPolicy[k])
                    out_action.append(make_binary(self._refPolicy[k][encoded_state[k]],padding))
        if returnEncoding:
            return out_action,encoded_state,out_state # out state distinguishes per agent state #encoded_state is an tentacle based definition
        else:
            return out_action

    def _getRandomAction(self,state,returnEncoding = False):
        
        action=[]
        for k in range(self._nAgents):
                action.append(np.random.randint(0,self.action_space_dim))
        if self._ganglia==False:
            encoded_state = state
            out_action = action
        else:
            #Control center scenarion
            encoded_state = [interpret_binary(s) for s in state] #here state index run through ganglia
            padding= int(self._nsuckers/self._nAgents)
            out_action = [make_binary(a,padding) for a in action]
        if self._parallelUpdate:
            out_state = encoded_state
        else:
            out_state = [(agent,state) for agent,state in enumerate(encoded_state)]
        if returnEncoding:
            return out_action,encoded_state,out_state
        else:
            return out_action
        
    def _getNullAction(self,state,returnEncoding = False):
        action=[]
        for k in range(self._nAgents):
                action.append(0)
        if self._ganglia==False:
            encoded_state = state
            out_action = action
        else:
            encoded_state = [interpret_binary(s) for s in state] #here state index run trhough ganglia
            padding= int(self._nsuckers/self._nAgents)
            out_action = [make_binary(a,padding) for a in action]
        if self._parallelUpdate:
            out_state = encoded_state
        else:
            out_state = [(agent,state) for agent,state in enumerate(encoded_state)]
        if returnEncoding:
            return out_action,encoded_state,out_state
        else:
            return out_action
        
    
    def loadPolicy(self,policy):
        '''
        Set by hand the default policy
        '''
        self._refPolicy = policy
        

    def test(self,env,policy,steps = 10000,doMovie=True):
        from tqdm import trange
        #consistency checks
        learning_space = env.info["learning space"]
        state_space_dim = learning_space[0]
        action_space_dim = learning_space[1]
        nsuckers = env.info["n suckers"]
        isGanglia = env.info["isGanglia"]
        self.loadPolicy(policy)

        if self.state_space_dim != state_space_dim:
            print("Not expected environment. Exit")
            return
        elif self.action_space_dim != action_space_dim:
            print("Not expected environment. Exit")
            return
        elif self._nsuckers != nsuckers:
            print("Not expected environment. Exit")
            return
        elif self._ganglia != isGanglia:
            print("Not expected environment. Exit")
            return
        try:
            nPolicies = len(policy)
            isHive = False
            print("number of independent agents (not Hive) = ",nPolicies)
        except TypeError:
            isHive = True
        if self._parallelUpdate != isHive:
            print("loaded policy not consistent with initialization (hive/not hive)")
            return
        else:
            self._refPolicy=self._refPolicy[()]

        
        print("Ganglia = ",isGanglia)
        print("number of suckers = ", nsuckers)
        if isGanglia==False:
            print(self._refPolicy)
        
        env.reset(equilibrate = True)
        state = env.get_state()
        cumulativeR = 0
        
        for s in trange(steps):
            action = self.getOnPolicyAction(state)
            state,r,_t = env.step(action)
            cumulativeR +=r
            if doMovie:
                if s % 10 ==0:
                    env.render()
        normVel = env.get_averageVel()/env.x0
        print("total Reward = ", cumulativeR)
        return normVel

    def getCurrentPolicy(self):
        return self._refPolicy