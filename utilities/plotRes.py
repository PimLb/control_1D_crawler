import matplotlib.pyplot as plt
import numpy as np
from analysis_utilities import anal_vel_l0norm
#Converter functions to customize value parsing. 
#If converters is callable, the function is applied to all columns, 
#else it must be a dict that maps column number to a parser function.
DIRECTORY = "cluster_fetchedData/"

omega = 0.1

def runTime2float(t):
    s = t.decode("utf-8")
    # print(s)
    ts = [float(x) for x in s.split(":")]
    hours = ts[0]+(t[1]+t[2]/60)/60
    return hours
converter = {6: runTime2float} #specify how to parse column 6 containing time info in the form h:m:s
#import data
# velHiveRef = [0.010602357406157392,
#  0.013202705697254995,
#  0.01481555291026699,
#  0.015953827010329682,
#  0.016270444776155107,
#  0.01510136549980456,
#  0.013569747807210188,
#  0.011986004466139594,
#  0.010602984181803643]
#UPDATED AFTER CORRECTING SMALL MISTAKE ON INTEGRATION..
velHiveRef= [0.010691365077392847,
 0.013418984458701443,
 0.014880122682468761,
 0.015790674397143427,
 0.015753154897242726,
 0.014311517086968987,
 0.012644010828694891,
 0.011044707327390482]
#  ,0.009715218837408168]

velHive_andInternalAnalytical =  [0.015075165504025566, 
                                  0.017997824376039404, 
                                  0.018964804390371397,
                                  0.019145151360623423, 
                                  0.01820729949733445, 
                                  0.015571697124311015, 
                                  0.01330342100296222, 
                                  0.01152438732777765]
                                #   ,0.010100407476570655]
                                 #[0.01504604189650354, 
                                 #0.01801948632355675, 
                                 #0.018987071999865122, 
                                 #0.019143955394042313, 
                                 #0.018172997469969917, 
                                 #0.015486257868610891, 
                                 #0.01318619794864702, 
                                 #0.011392444202643674, 
                                 #0.0099640551181798]
velRandomPolicy_internal = [0.007381587321333672, 
                            0.009382863933990897, 
                            0.009861056606322393, 
                            0.00974469210517905, 
                            0.009159946493168387, 
                            0.007901160413333998, 
                            0.0067430043981731936, 
                            0.005815911740881602]
                            # ,0.005102249313776164]

velRandomPolicy_all = [1.1583810086719382e-05, 
                       2.7872820799369438e-05, 
                       -2.7500561831688585e-05, 
                       3.483122997752872e-05, 
                       1.2055079057978536e-05, 
                       0.00012575697627756276, 
                       -1.2998087279170178e-06, 
                       5.237910363703691e-06]
                    #    ,-1.944032698717857e-05]


phaseVel = [0.07957747154594767, #w/k = w*N/2pi
0.12732395447351627,
0.15915494309189535,
0.19098593171027445,
0.23873241463784303,
0.3183098861837907,
0.3978873577297383,
0.47746482927568606]

analVel = [0.01985459776677514,
0.03142078283072522,
0.03857058159880506,
0.044855675246500755,
0.05185284839162747,
0.0558992919552935,
0.053121818061563714,
0.04794963081538428]

nsAll = [5, 8, 10, 12, 15, 20, 25, 30]


velMulti = np.loadtxt(DIRECTORY+'velSummaryMULTI.txt',converters=converter)
velHive = np.loadtxt(DIRECTORY+'velSummaryHIVE.txt',converters=converter)
vel1G = np.loadtxt(DIRECTORY+'velSummary1Ganglion.txt',converters=converter)
vel2GHive = np.loadtxt(DIRECTORY+'velSummary2GangliaHIVE.txt',converters=converter)
vel2G = np.loadtxt(DIRECTORY+'velSummary2Ganglia.txt',converters=converter)
vel3GHive = np.loadtxt(DIRECTORY+'velSummary3GangliaHIVE.txt',converters=converter)
vel3G = np.loadtxt(DIRECTORY+'velSummary3Ganglia.txt',converters=converter)
vel4GHive = np.loadtxt(DIRECTORY+'velSummary4GangliaHIVE.txt',converters=converter)
vel4G = np.loadtxt(DIRECTORY+'velSummary4Ganglia.txt',converters=converter)
vel5GHive = np.loadtxt(DIRECTORY+'velSummary5GangliaHIVE.txt',converters=converter)


nsMulti = velMulti[:,0]
avVelMulti = velMulti[:,1]
avVelMulti_err = velMulti[:,2]
avVelMultiMax = velMulti[:,3]
avVelMultiMax_err = velMulti[:,4]
velMultiBest = velMulti[:,5]

nsHive = velHive[:,0]
avVelHive = velHive[:,1]
avVelHive_err = velHive[:,2]
avVelHiveMax = velHive[:,3]
avVelHiveMax_err = velHive[:,4]
velHiveBest = velHive[:,5]


ns1g = vel1G[:,0]
avVel1g = vel1G[:,1]
avVel1g_err = vel1G[:,2]
avVel1gMax = vel1G[:,3]
avVel1gMax_err = vel1G[:,4]
vel1GBest = vel1G[:,5]

ns2gHive = vel2GHive[:,0]
avVel2gHive = vel2GHive[:,1]
avVel2gHive_err = vel2GHive[:,2]
avVel2gHiveMax = vel2GHive[:,3]
avVel2gHiveMax_err = vel2GHive[:,4]
vel2gHiveBest = vel2GHive[:,5]

ns2g = vel2G[:,0]
avVel2g = vel2G[:,1]
avVel2g_err = vel2G[:,2]
avVel2gMax = vel2G[:,3]
avVel2gMax_err = vel2G[:,4]
vel2gBest= vel2G[:,5]

# print(vel3GHive)
ns3gHive = vel3GHive[:,0]
avVel3gHive = vel3GHive[:,1]
avVel3gHive_err = vel3GHive[:,2]
avVel3gHiveMax = vel3GHive[:,3]
avVel3gHiveMax_err = vel3GHive[:,4]
vel3gHiveBest = vel3GHive[:,5]

ns3g = vel3G[:,0]
avVel3g = vel3G[:,1]
avVel3g_err = vel3G[:,2]
avVel3gMax = vel3G[:,3]
avVel3gMax_err = vel3G[:,4]
vel3gBest = vel3G[:,5]

ns4gHive = vel4GHive[0]
avVel4gHive = vel4GHive[1]
avVel4gHive_err = vel4GHive[2]
avVel4gHiveMax = vel4GHive[3]
avVel4gHiveMax_err = vel4GHive[4]
vel4gHiveBest = vel4GHive[5]

ns4g = vel4G[0]
avVel4g = vel4G[1]
avVel4g_err = vel4G[2]
avVel4gMax = vel4G[3]
avVel4gMax_err = vel4G[4]
vel4gBest = vel4G[5]

# ns5gHive = vel5GHive[0]
# avVel5gHive = vel5GHive[1]
# avVel5gHive_err = vel5GHive[2]
# avVel5gHiveMax = vel5GHive[3]
# avVel5gHiveMax_err = vel5GHive[4]
# intTime_5gHive = vel5GHive[6]


plt.figure()
plt.ion()
# n = np.arange(5,40)
plt.xticks([5,8,10,12,15,20,25,30])

plt.plot(nsAll,velHive_andInternalAnalytical,'--',label="BENCHMARK: internal suckers analytical",lw=3, color='gray')
plt.plot(nsAll,velHiveRef,'p',label="reference sucker hive policy",ms=10)

# plt.errorbar(nsHive,avVelHive,yerr=avVelHive_err,fmt='p',label="Multi Hive Average",ms=6,elinewidth=2,capsize=7,capthick=1)
plt.errorbar(nsMulti,avVelMulti,yerr=avVelMulti_err,fmt='^',label="Multi Average",ms=6,elinewidth=2,capsize=7,capthick=1)
plt.errorbar(ns1g,avVel1g,yerr=avVel1g_err,fmt='*',label="1 Ganglion Average",ms=9,elinewidth=2,capsize=7,capthick=1)

plt.errorbar(ns2gHive,avVel2gHive,yerr=avVel2gHive_err,fmt='8',label="2 Ganglia HIVE Average",ms=6,elinewidth=2,capsize=7,capthick=1)
plt.errorbar(ns2g,avVel2g,yerr=avVel2g_err,fmt='o',label="2 Ganglia Average",ms=6,elinewidth=2,capsize=7,capthick=1)

plt.errorbar(ns3g,avVel3gHive,yerr=avVel3gHive_err,fmt='8',label="3 Ganglia HIVE Average",ms=6,elinewidth=2,capsize=7,capthick=1)
plt.errorbar(ns3g,avVel3g,yerr=avVel3g_err,fmt='o',label="3 Ganglia Average",ms=6,elinewidth=2,capsize=7,capthick=1)

plt.errorbar(ns4g,avVel4gHive,yerr=avVel4gHive_err,fmt='s',label="4 Ganglia HIVE Average",ms=6,elinewidth=2,capsize=7,capthick=1)
plt.errorbar(ns4g,avVel4g,yerr=avVel4g_err,fmt='D',label="4 Ganglia Average",ms=6,elinewidth=2,capsize=7,capthick=1)

# plt.errorbar(ns5gHive,avVel5gHive,yerr=avVel5gHive_err,fmt='s',label="5 Ganglia HIVE Average",ms=6,elinewidth=2,capsize=7,capthick=1)


plt.legend()
plt.show()

plt.figure()
plt.ion()
plt.xticks([5,8,10,12,15,20,25,30])
plt.title("ALL RESULTS")
# plt.plot(n,anal_vel_l0norm(n,omega),label="analytical")
plt.plot(nsAll,velHive_andInternalAnalytical,'--',label="BENCHMARK",lw=3, color='gray')
#plt.plot(nsAll,velHiveRef,'p',label="Multiagent Hive",ms=10)
plt.errorbar(nsHive,avVelHiveMax,yerr=avVelHiveMax_err,fmt='p',label="Suckers Hive",ms=6,elinewidth=2,capsize=7,capthick=1)
plt.errorbar(ns1g,avVel1gMax,yerr=avVel1gMax_err,fmt='*',label="1 Ganglion standard",ms=9,elinewidth=2,capsize=7,capthick=1)

plt.errorbar(ns2gHive,avVel2gHiveMax,yerr=avVel2gHiveMax_err,fmt='8',label="2 Ganglia Hive",ms=6,elinewidth=2,capsize=7,capthick=1)
plt.errorbar(nsMulti,avVelMultiMax,yerr=avVelMultiMax_err,fmt='^',label="Multiagent standard",ms=6,elinewidth=2,capsize=7,capthick=1)

plt.errorbar(ns2g,avVel2gMax,yerr=avVel2gMax_err,fmt='o',label="2 Ganglia standard",ms=6,elinewidth=2,capsize=7,capthick=1)

plt.errorbar(ns3g,avVel3gHiveMax,yerr=avVel3gHiveMax_err,fmt='8',label="3 Ganglia Hive",ms=6,elinewidth=2,capsize=7,capthick=1)
plt.errorbar(ns3g,avVel3gMax,yerr=avVel3gMax_err,fmt='o',label="3 Ganglia standard",ms=6,elinewidth=2,capsize=7,capthick=1)

plt.errorbar(ns4g,avVel4gHiveMax,yerr=avVel4gHiveMax_err,fmt='s',label="4 Ganglia Hive",ms=6,elinewidth=2,capsize=7,capthick=1)
plt.errorbar(ns4g,avVel4gMax,yerr=avVel4gMax_err,fmt='D',label="4 Ganglia standard",ms=6,elinewidth=2,capsize=7,capthick=1)

# plt.errorbar(ns5gHive,avVel5gHiveMax,yerr=avVel5gHiveMax_err,fmt='s',label="5 Ganglia Hive Max",ms=6,elinewidth=2,capsize=7,capthick=1)
plt.legend()
plt.show()

################# ONLY DISTRIBUTED ###############
plt.figure()
plt.ion()
plt.title("Distributed architerctures")
plt.xticks([5,8,10,12,15,20,25,30])
plt.ylim(-0.001,0.022)

plt.plot(nsAll,velHive_andInternalAnalytical,'--',label="Semi analytical benchmark",lw=3, color='gray')
plt.errorbar(nsHive,avVelHiveMax,yerr=avVelHiveMax_err,fmt='o',label="Suckers Hive",ms=6,elinewidth=2,capsize=7,capthick=1)
plt.plot(nsAll,velRandomPolicy_internal,'--',label="Random policy inner suckers",lw =3 )
plt.plot(nsAll,velRandomPolicy_all,'--',label="Random policy (all)",lw=3)

plt.errorbar(nsMulti,avVelMultiMax,yerr=avVelMultiMax_err,fmt='^',label="Suckers standard",ms=6,elinewidth=2,capsize=7,capthick=1)


plt.legend()
plt.show()


plt.figure()
plt.ion()
plt.title("Analytical")
plt.xticks([5,8,10,12,15,20,25,30])
# plt.xlim(5,20)
# plt.ylim(-0.001,0.022)
# plt.plot(n,anal_vel_l0norm(n,omega),label="analytical")

plt.plot(nsAll,velHive_andInternalAnalytical,'--',label="Semi analytical benchmark",lw=2, color='gray')
# plt.errorbar(nsHive,avVelHiveMax,yerr=avVelHiveMax_err,fmt='o',label="Suckers Hive",ms=6,elinewidth=2,capsize=7,capthick=1)
# plt.plot(nsAll,velRandomPolicy_internal,'--',label="Random policy inner suckers",lw =3 )
# plt.plot(nsAll,velRandomPolicy_all,'--',label="Random policy (all)",lw=3)

# plt.errorbar(nsMulti,avVelMultiMax,yerr=avVelMultiMax_err,fmt='^',label="Suckers standard",ms=6,elinewidth=2,capsize=7,capthick=1)

plt.plot(nsAll,analVel,'-',label="Analytical benchmark",lw=2, color='black')
plt.plot(nsAll,phaseVel,'--',label="Phase velocity",lw=2, color='red')


plt.legend()
plt.show()



plt.figure()
plt.ion()
plt.title("ALL RESULTS REDUCED")
plt.xticks([5,8,10,12,15,20,25,30])
plt.ylim(-0.001,0.022)
# plt.plot(n,anal_vel_l0norm(n,omega),label="analytical")
plt.plot(nsAll,velHive_andInternalAnalytical,'--',label="BENCHMARK",lw=3, color='gray')
# plt.plot(nsAll,velHiveRef,'p',label="Multiagent Hive",ms=10)
# plt.errorbar(nsHive,avVelHiveMax,yerr=avVelHiveMax_err,fmt='p',label="Multi Hive Max",ms=6,elinewidth=2,capsize=7,capthick=1)

plt.errorbar(ns1g,avVel1gMax,yerr=avVel1gMax_err,fmt='o',color ="tab:red", label="1 CC standard",ms=6,elinewidth=2,capsize=7,capthick=1)
ns1gBest = [5.5,8.5,10.5,12.5,15]
plt.plot(ns1gBest,vel1GBest,'*',color ="tab:red",ms=12)

ns2gHive = [10.5,12.5,20,30]
plt.errorbar(ns2gHive,avVel2gHiveMax,yerr=avVel2gHiveMax_err,fmt='o',color="blue",label="2 CC Hive",ms=6,elinewidth=2,capsize=7,capthick=1)
ns2gHiveBest = [10.5,12.5,20,31]
plt.plot(ns2gHiveBest,vel2gHiveBest,'*',color ="blue",ms=12)
# plt.errorbar(nsMulti,avVelMultiMax,yerr=avVelMultiMax_err,fmt='^',label="Multiagent standard",ms=6,elinewidth=2,capsize=7,capthick=1)

ns2g = [9,11.5,19.5,29.5]
plt.errorbar(ns2g,avVel2gMax,yerr=avVel2gMax_err,fmt='o',color="tab:orange",label="2, CC standard",ms=6,elinewidth=2,capsize=7,capthick=1)
ns2gBest = [9.5,11.5,20,30]
plt.plot(ns2gBest,vel2gBest,'*',color ="tab:orange",ms=12)

ns3gHive = [15.5,30]
plt.errorbar(ns3gHive,avVel3gHiveMax,yerr=avVel3gHiveMax_err,fmt='o',color="green", label="3 CC Hive",ms=6,elinewidth=2,capsize=7,capthick=1)
ns3gHiveBest = [15.5,30.5]
plt.plot(ns3gHiveBest,vel3gHiveBest,'*',color ="green",ms=12)

ns3g = [14.5,29.5]
plt.errorbar(ns3g,avVel3gMax,yerr=avVel3gMax_err,fmt='o',color="tab:purple",label="3 CC standard", ms=6,elinewidth=2,capsize=7,capthick=1)
ns3gBest = [14.5,29.5]
plt.plot(ns3gBest,vel3gBest,'*',color ="tab:purple",ms=12)

plt.errorbar(ns4g,avVel4gHiveMax,yerr=avVel4gHiveMax_err,fmt='o',color="tab:brown", label="4 CC Hive",ms=6,elinewidth=2,capsize=7,capthick=1)
ns4gHiveBest = [20.5]
plt.plot(ns4gHiveBest,vel4gHiveBest,'*',color ="tab:brown",ms=12)

plt.errorbar(ns4g,avVel4gMax,yerr=avVel4gMax_err,fmt='o',color="tab:pink", label="4 CC standard",ms=6,elinewidth=2,capsize=7,capthick=1)
ns4gBest = [19.5]
plt.plot(ns4gBest,vel4gBest,'*',color ="tab:pink",ms=12)

# plt.errorbar(ns5gHive,avVel5gHiveMax,yerr=avVel5gHiveMax_err,fmt='s',label="5 Ganglia Hive Max",ms=6,elinewidth=2,capsize=7,capthick=1)
plt.legend()
plt.show()


input()