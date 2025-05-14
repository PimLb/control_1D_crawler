import numpy as np
import matplotlib.pyplot as plt
import sys

#REMEMBER: Spring tension based on rightHand tension

nsuckers  = 12 
isGanglia = True
nGanglia = 2
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
inputStates = np.load(filename,allow_pickle=True)
# print(inputStates)
# input()
# if nAgents >1:
#     exit("More than 1 agent not admittable yet..")

#preprocessing
# if not isHive:
# states = [state[1] for state in inputStates.tolist()]
states = inputStates.tolist()
# if oneAgent: 
#     agents = [1]
# else:
#     agents = []
#     for agentState in states:
#         agents = agentState
#     agents = [state[0] for state in states.tolist()]
    # states = [state[1] for state in states.tolist()]
# else:
#     pass
nAgents = 12
padding= int(nsuckers/nGanglia)-1 #padding on number of springs not suckers
# print(states)
# print(padding)

###3

n_states = len(states)
print("Number of states visited (by considered policy) = ", n_states)
#hard fix
plt.figure()
plt.ion()
plt.xlabel("spring")
if n_states<50:
    plt.ylabel("unique traversed states")
    plt.title("Unique spring states")
    print("UNIQUE STATES")
    input("continue?")
else:
    plt.ylabel("time")
    plt.title("All spring states")
    print("ALL STATES (time plot)")
    input("continue?")
if isGanglia:
    Z=np.empty(int((nsuckers/nGanglia-1)*nGanglia)) #number of springs
else:
    Z = np.empty(nsuckers -1 )
# print(Z.shape)
# print(Z) #is not actually empty
for n in range(n_states):
    if isGanglia:
        decoding = []
        for a in range(nGanglia):
            decoding.extend(make_binary(states[n][a],padding))
        #     print(states[n][a])
        #     print([int(i) for i in bin(states[n][a])[2:]])
        # input()
        #check
    else:
        #convert to spring states
        decoding = readSpring(states[n],nsuckers)
    # print(decoding)
    decoding = np.array(decoding)
    Z=np.vstack((Z,decoding))
    # print(decoding)
Z = Z[1:] #discard first random entry
print(Z.shape)
if nGanglia==2 and isGanglia:
    print("insert idle spring")
    Z = np.insert(Z, 5, 10, axis=1)
    # print(Z)
    Z=np.ma.masked_where(Z==10,Z)
# print(Z)
# input()
    # Think about plotting strategy.. -> color for elongation/compression of spring
aspect = 11/n_states*1.2
plt.imshow(Z,aspect="auto")
plt.show()
input()


