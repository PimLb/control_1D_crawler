import matplotlib.pyplot as plt
import numpy as np
import glob
import sys


totalNumberPolicies = 501
folder = sys.argv[1]
filenames = glob.glob(folder+"*.txt")
if len(filenames) == 0:
    folder = ""
    filenames = glob.glob(folder+"*.txt")
# print(filenames)
plt.figure()
plt.ion()
for filename in filenames:
    data = np.loadtxt(filename)
    archType = filename.split(".")[0].split("_")[1:]
    title = ("").join(archType)
    distr = data[:,1]
    vel = data[:,2]
    # print(title)

    n = np.arange(0,len(distr))

    #re-normalized data
    norm_vel = vel/np.amax(vel)
    maxNorm_distr = distr/np.amax(distr)
   
    # p = plt.plot(vel,maxNorm_distr,"--o",ms=4,label = "re-normalized pol freq "+title)
    # currentColor = p[-1].get_color()
    # plt.hist(vel,bins=100,weights=distr,color = currentColor,label = "normalized vel freq "+title)
    plt.hist(vel,bins=100,weights=distr,label = "vel freq "+title)
    plt.xlabel("velocity")
    plt.ylabel("frequency")
    # plt.ylim(-0.025,np.amax(distr)+0.025)
    # plt.title(title)
    plt.legend()



plt.show()

input()
