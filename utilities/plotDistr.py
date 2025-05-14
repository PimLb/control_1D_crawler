import matplotlib.pyplot as plt
import numpy as np
import sys


#red = velocity
#blue = policy
totalNumberPolicies = 501

filename = sys.argv[1]
archType = filename.split(".")[0].split("_")[1:]
title = ("").join(archType)

data = np.loadtxt(filename)

multiplicity = data[:,0] #=distr * 501..
distr = data[:,1]
vel = data[:,2]

n = np.arange(0,len(distr))

#re-normalized data
norm_vel = vel/np.amax(vel)
maxNorm_distr = distr/np.amax(distr)
# print(n)

plt.figure()
plt.ion()

plt.hist(vel,bins=100,weights=distr,label = "velocity frequency",color = "tab:blue")
plt.plot(vel,distr,'*--',color = "tab:red",lw=2,label = "associated policy frequency")
plt.xlabel("velocity")
plt.ylabel("frequency")
# plt.xscale("log")
# plt.ylim(-0.025,np.amax(distr)+0.025)
plt.title(title)
plt.legend()


plt.figure()
plt.ion()

plt.hist(norm_vel,bins=100,weights=distr,label = "normalized velocity frequency",color = "tab:blue")
plt.plot(norm_vel,distr,'*--',lw=2,label = "associated policy frequency",color = "tab:red")
plt.xlabel("norm velocity")
plt.ylabel("frequency")
# plt.ylim(-0.025,np.amax(distr)+0.025)
plt.title(title)
plt.legend()




plt.figure()
plt.ion()


plt.bar(n,maxNorm_distr,label = "renormalized policy frequency",color = "tab:blue")
# plt.plot(n,maxNorm_distr,'*',label = "renormalized policy frequency",color = "tab:blue")
plt.plot(n,norm_vel,'o',color = "tab:red",lw=2,label = "normalized velocity")
plt.xlabel("policy id (sorted)")
plt.ylabel("distribution")
# plt.xscale("log")
# plt.ylim(np.amin(norm_vel)-0.025,1.025)
plt.title(title)
plt.legend()

plt.show()


#NEW policy frequency in same binning

plt.figure(figsize=(10,8))
plt.ion()
n_bins = 50

plt.hist(vel,bins=n_bins,color = "tab:blue",histtype = "step",alpha=1,linewidth=2,label = "# unique policies",log="true",align="mid")
counts, edges, bars = plt.hist(vel,bins=n_bins,color = "tab:orange",weights=multiplicity,histtype = "bar",align="mid",label = "# times velocity realized",log="true")
plt.axhline(1,ls="--",c="black",lw=1)

# plt.hist(vel,bins=100,color = "tab:red",weights=distr,label = "velocity frequency",log="true")
# print(bars)
# plt.bar_label(bars)
plt.xlabel("velocity")
plt.ylabel("counter")
plt.xlim(-0.005,0.025)
plt.ylim(top=370)
plt.title(title)
plt.legend()

plt.show()


# plt.figure()
# plt.ion()

# # plt.hist(vel,bins=100,color = "tab:red",weights=distr,label = "velocity frequency")
# plt.hist(vel,bins=100,color = "tab:blue",label = "policy frequency")
# plt.xlabel("velocity")
# plt.ylabel("frequency")
# # plt.xscale("log")
# # plt.ylim(-0.025,np.amax(distr)+0.025)
# plt.title(title)
# plt.legend()

# plt.show()


input()