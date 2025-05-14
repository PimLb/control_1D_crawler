import numpy as np
import matplotlib.pyplot as plt

dataHive = np.loadtxt("12_distributed_hive_uniquePolDistanceFromMax_agentBased.txt")
vel_hive = dataHive[:,0]
distance_hive = dataHive[:,1]
distanceNorm_hive = dataHive[:,2] #absoluteNorm (normalizing by all possible control states - sum of Q matrixes states)
visitedStates_hive = dataHive[:,3] #can be all tentacle based: [(statesAgent1,statesAgent2),(statesAgent1,statesAgent2)]
                                   #can be agent based:[(agent1,state1),(agent2,state2) ecc] --> here I count less since don't recount if other agent in different state..

dataMulti = np.loadtxt("12_distributed_standard_uniquePolDistanceFromMax_agentBased.txt")
vel_multi = dataMulti[:,0]
distance_multi = dataMulti[:,1]
distanceNorm_multi = dataMulti[:,2] 
visitedStates_multi = dataMulti[:,3]

data1G = np.loadtxt("12_1G_uniquePolDistanceFromMax.txt")
vel_1G = data1G[:,0]
distance_1G = data1G[:,1]
distanceNorm_1G = data1G[:,2] 
visitedStates_1G = data1G[:,3]

data2G = np.loadtxt("12_2G_standard_uniquePolDistanceFromMax_agentBased.txt")
vel_2G = data2G[:,0]
distance_2G = data2G[:,1]
distanceNorm_2G = data2G[:,2] 
visitedStates_2G = data2G[:,3]

data2G_hive = np.loadtxt("12_2G_hive_uniquePolDistanceFromMax_agentBased.txt")
vel_2G_hive = data2G_hive[:,0]
distance_2G_hive = data2G_hive[:,1]
distanceNorm_2G_hive = data2G_hive[:,2] 
visitedStates_2G_hive = data2G_hive[:,3]

# Counting tentacle -state TODO
# visitedStates_RDP_hive = [8] * len(vel_hive)
# visitedStates_RDP_hive_std = [0.24] * len(vel_hive)
# visitedStates_RDP_multi = [39.91] * len(vel_multi) 
# visitedStates_RDP_multi_std = [4.31] * len(vel_multi) 
# visitedStates_RDP_2G = [25.98] * len(vel_2G) 
# visitedStates_RDP_2G_std = [5.46] * len(vel_2G) 
# visitedStates_RDP_2G_hive= [16.87] * len(vel_2G_hive) 
# visitedStates_RDP_2G_hive_std = [3.62] * len(vel_2G_hive) 
# visitedStates_RDP_1G= [40.57] * len(vel_1G)
# visitedStates_RDP_1G_std = [10.43] * len(vel_1G)

# Counting agent -state
visitedStates_RDP_hive = np.array([7.95] * len(vel_hive))
visitedStates_RDP_hive_std = np.array([0.23] * len(vel_hive))
visitedStates_RDP_multi = np.array([38.74] * len(vel_multi)) 
visitedStates_RDP_multi_std = np.array([4.44] * len(vel_multi) )
visitedStates_RDP_2G = np.array([26.13] * len(vel_2G) ) 
visitedStates_RDP_2G_std = np.array([4.97] * len(vel_2G) )
visitedStates_RDP_2G_hive= np.array([16.83] * len(vel_2G_hive) )
visitedStates_RDP_2G_hive_std = np.array([3.75] * len(vel_2G_hive)) 
visitedStates_RDP_1G= np.array([40.57] * len(vel_1G))
visitedStates_RDP_1G_std = np.array([10.43] * len(vel_1G))

#Absolute norms
norm_hive = 8
norm_multi = 44
norm_2G_hive = 32
norm_2G_standard = 64
norm_1G = 2048



#non normalized DISTANCE PLOT

plt.figure()
plt.ion()
plt.ylabel("distance")
plt.title("Policy distance from best normalized")

plt.plot(vel_hive,distance_hive,color="tab:blue",label="distributed hive")
plt.plot(vel_multi,distance_multi,color="tab:purple",label="distributed standard")
plt.plot(vel_2G_hive,distance_2G_hive,color="tab:green",label="2G hive")
plt.plot(vel_2G,distance_2G,color="tab:orange",label="2G standard")
plt.plot(vel_1G,distance_1G,color="tab:red",label="1G")

plt.legend()


#Normalized DISTANCE PLOT

plt.figure()
plt.ion()
plt.ylabel("distance [%%]")
plt.title("Policy distance from best normalized")

plt.plot(vel_hive,distanceNorm_hive,color="tab:blue",label="distributed hive")
plt.plot(vel_multi,distanceNorm_multi,color="tab:purple",label="distributed standard")
plt.plot(vel_2G_hive,distanceNorm_2G_hive,color="tab:green",label="2G hive")
plt.plot(vel_2G,distanceNorm_2G,color="tab:orange",label="2G standard")
plt.plot(vel_1G,distanceNorm_1G,color="tab:red",label="1G")

plt.legend()


# Visited States

plt.figure()
plt.ion()
plt.ylabel("# states")
plt.title("Number of states with Random Deterministic")

plt.plot(vel_hive,visitedStates_hive,color="tab:blue",label="distributed hive")
plt.plot(vel_multi,visitedStates_multi,color="tab:purple",label="distributed standard")
plt.plot(vel_2G_hive,visitedStates_2G_hive,color="tab:green",label="2G hive")
plt.plot(vel_2G,visitedStates_2G,color="tab:orange",label="2G standard")
plt.plot(vel_1G,visitedStates_1G,color="tab:red",label="1G")

#shaded areas std # states Random Deterministic, dashed line: average
plt.plot(vel_hive,visitedStates_RDP_hive,'--',color = "tab:blue")
plt.fill_between(vel_hive,visitedStates_RDP_hive+visitedStates_RDP_hive_std,visitedStates_RDP_hive-visitedStates_RDP_hive_std,color="tab:blue",alpha=0.5)
plt.plot(vel_multi,visitedStates_RDP_multi,'--',color = "tab:purple")
plt.fill_between(vel_multi,visitedStates_RDP_multi+visitedStates_RDP_multi_std,visitedStates_RDP_multi-visitedStates_RDP_multi_std,color="tab:purple",alpha=0.5)
plt.plot(vel_2G_hive,visitedStates_RDP_2G_hive,'--',color = "tab:green")
plt.fill_between(vel_2G_hive,visitedStates_RDP_2G_hive+visitedStates_RDP_2G_hive_std,visitedStates_RDP_2G_hive-visitedStates_RDP_2G_hive_std,color="tab:green",alpha=0.5)
plt.plot(vel_2G,visitedStates_RDP_2G,'--',color = "tab:orange")
plt.fill_between(vel_2G,visitedStates_RDP_2G+visitedStates_RDP_2G_std,visitedStates_RDP_2G-visitedStates_RDP_2G_std,color="tab:orange",alpha=0.5)
plt.plot(vel_1G,visitedStates_RDP_1G,'--',color = "tab:red")
plt.fill_between(vel_1G,visitedStates_RDP_1G+visitedStates_RDP_1G_std,visitedStates_RDP_1G-visitedStates_RDP_1G_std,color="tab:red",alpha=0.5)

plt.legend()


# Visited States Norm

plt.figure()
plt.ion()
plt.ylabel("# states")
plt.title("Number of states with Random Deterministic Normalized")

plt.plot(vel_hive,visitedStates_hive/norm_hive,color="tab:blue",label="distributed hive")
plt.plot(vel_multi,visitedStates_multi/norm_multi,color="tab:purple",label="distributed standard")
plt.plot(vel_2G_hive,visitedStates_2G_hive/norm_2G_hive,color="tab:green",label="2G hive")
plt.plot(vel_2G,visitedStates_2G/norm_2G_standard,color="tab:orange",label="2G standard")
plt.plot(vel_1G,visitedStates_1G/norm_1G,color="tab:red",label="1G")

#shaded areas std # states Random Deterministic, dashed line: average
plt.plot(vel_hive,visitedStates_RDP_hive/norm_hive,'--',color = "tab:blue")
plt.fill_between(vel_hive,(visitedStates_RDP_hive+visitedStates_RDP_hive_std)/norm_hive,(visitedStates_RDP_hive-visitedStates_RDP_hive_std)/norm_hive,color="tab:blue",alpha=0.5)
plt.plot(vel_multi,visitedStates_RDP_multi/norm_multi,'--',color = "tab:purple")
plt.fill_between(vel_multi,(visitedStates_RDP_multi+visitedStates_RDP_multi_std)/norm_multi,(visitedStates_RDP_multi-visitedStates_RDP_multi_std)/norm_multi,color="tab:purple",alpha=0.5)
plt.plot(vel_2G_hive,visitedStates_RDP_2G_hive/norm_2G_hive,'--',color = "tab:green")
plt.fill_between(vel_2G_hive,(visitedStates_RDP_2G_hive+visitedStates_RDP_2G_hive_std)/norm_2G_hive,(visitedStates_RDP_2G_hive-visitedStates_RDP_2G_hive_std)/norm_2G_hive,color="tab:green",alpha=0.5)
plt.plot(vel_2G,visitedStates_RDP_2G/norm_2G_standard,'--',color = "tab:orange")
plt.fill_between(vel_2G,(visitedStates_RDP_2G+visitedStates_RDP_2G_std)/norm_2G_standard,(visitedStates_RDP_2G-visitedStates_RDP_2G_std)/norm_2G_standard,color="tab:orange",alpha=0.5)
plt.plot(vel_1G,visitedStates_RDP_1G/norm_1G,'--',color = "tab:red")
plt.fill_between(vel_1G,(visitedStates_RDP_1G+visitedStates_RDP_1G_std)/norm_1G,(visitedStates_RDP_1G-visitedStates_RDP_1G_std)/norm_1G,color="tab:red",alpha=0.5)

plt.legend()

plt.show()
input()