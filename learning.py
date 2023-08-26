import numpy as np

lr = 0.1#learning rate
gamma = 0.9#discount 0.9 to reflect upon..
epsilon = 0.9
min_epsilon =0.


def interpret_binary(s:tuple):
    return int("".join(str(ele) for ele in s), 2)

def interpret_thernary(s:tuple):
    return int("".join(str(ele) for ele in s), 3)


def make_binary(index:int):
    return [int(i) for i in bin(index)[2:]]

class actionValue(object):
    def __init__(self,learning_space:tuple,nAgents,n_episodes,collapse = True) -> None:
        self._state_space = learning_space[0]
        self._action_space = learning_space[1]
        self._nAgents = nAgents
        self.epsilon = epsilon
        self.discount = gamma
        self._upgrade_e = (epsilon-min_epsilon)/n_episodes
        if collapse:
            self._parallelUpdate = True
            self.dim = learning_space
            print("\n** HIVE UPDATE **\n")
        else:
            self._parallelUpdate = False
            self.dim = learning_space + (nAgents,)
        if nAgents>1:
            self._multiAgent = True    
        else:
            self._multiAgent = False

        


        Q = np.zeros(learning_space)
        
        if not collapse and self._multiAgent:
            self._Q = []
            for k in range(nAgents):
                self._Q.append(Q) 
        else:
            self._Q = Q

        
    def reset(self):
        self._Q = np.zeros(self.dim)
        self.epsilon = epsilon
    
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
    def _update_Q(self,newstate,oldstate,action,reward):
        '''
        Update the Q function, return new action.
        Both encompass a distinct Q function for each agent, and a single one which is updated by all agents.
        The latter is for the nature of the problem. Conceptually it's still multiagent for how the states
        have been defined
        '''

        #update each agent Q
        if self._parallelUpdate:
            for  k in range(0,self._nAgents):
                s_new,_a_new = self._get_index(newstate[k]) #CAREFUL HERE ACTION DOES NOT MATTER, IS A DUMMY NUMBER
                s_old,a_old = self._get_index(oldstate[k],action[k])
                self._Q[s_old,a_old] += lr* (reward + gamma * np.amax(self._Q[s_new]) - self._Q[s_old,a_old])
        else:
            for  k in range(self._nAgents):
                s_new,_a_new = self._get_index(newstate[k],action[k]) #CAREFUL HERE ACTION DOES NOT MATTER, IS A DUMMY NUMBER
                s_old,a_old = self._get_index(oldstate[k],action[k])
                self._Q[k][s_old,a_old] += lr* (reward + gamma * np.amax(self._Q[k][s_new]) - self._Q[k][s_old,a_old])

                #This could be more efficient but applicable only in this case
                # if np.random.random() < (1 - self.epsilon):
                #     newaction.append(np.argmax(self._Q[k][s_new]))
                # else:
                #     newaction.append(np.random.randint(0,  self.action_space))

        
    
    def update(self,newState,oldState,oldAction,reward):
        # newState = env._newState
        # oldState = env._oldState
        # oldAction = env._oldAction
        # reward = env._current_reward
        self._update_Q(newState,oldState,oldAction,reward)

        
    def get_action(self,s):
        
        if self._multiAgent:
            new_action = []
            if self._parallelUpdate:
                #single Q funcion updated by each agent. 
                #Fetching action for each agent looping through each agent state
                for k in range(self._nAgents):
                    # print(s[k])
                    #SKIPPING THOSE
                    # if(k==0 or k==self._nAgents-1):
                    #     new_action.append(0)
                    #     continue
                    sind,_a = self._get_index(s[k])
                    # print(s[k],sind)
                    if np.random.random() < (1 - self.epsilon):
                        # print("greedy")
                        new_action.append(np.argmax(self._Q[sind]))
                    else:
                        new_action.append(np.random.randint(0,  self._action_space))
            else:
                #one Q function for each agent
                for k in range(self._nAgents):
                    sind,_a = self._get_index(s[k])
                    if np.random.random() < (1 - self.epsilon):
                        new_action.append(np.argmax(self._Q[k][sind]))
                    else:
                        new_action.append(np.random.randint(0,  self._action_space))
            return new_action
        else:
            #CHECK
            if np.random.random() < (1 - self.epsilon):
                sind,_a = self._get_index(s)
                new_action=np.argmax(self._Q[sind])
            else:
                new_action=np.random.randint(0,  self._action_space)

            return make_binary(new_action) #need to have an instruction to be given to the tentacle (which sucker hangs)

    def makeGreedy(self):
        self.epsilon -= self._upgrade_e
        return self.epsilon
    
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
    
    
    def get_value(self):
        return self._Q
    
   