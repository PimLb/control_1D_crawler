

import numpy as np
from env import Environment
from env import x0Fraction
from learning import actionValue
from analysis_utilities import *
import matplotlib.pyplot as plt
from tqdm import trange

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})


# sim_shape =(110,)
# t_position = 3000

# print ("Epsilon = ",epsilon)
# print("Period = ",Environment.N)

# print("** Checking if control parameter is just N vs Nsuckers (x0 does not matter..)**")
# # #Fix N

# Ltmax = 100
# plt.figure()
# plt.ion()

# fig = plt.subplot(xlabel='N suckers', ylabel ='$v_{CM}/l_0$')
# fig.title.set_text('Checking continuous limit, N=%d. -alpha + beta'%Environment.N)
# fig.axhline(anal_vel_l0norm(Environment.N),xmin=0,xmax=Ltmax,c='black',ls ='dashed',label='analytical prediction')

# fig.set_ylim([0, anal_vel_l0norm(Environment.N)+anal_vel_l0norm(Environment.N)/2])
# fig.set_xlim([0, Ltmax])

# plt.legend()
# plt.show()

# print('normalized prediction',anal_vel_l0norm(Environment.N))
# input()
# # while(1):
# #     x0 = float(input('insert x0\n'))
# for x0 in [0.1,0.5,1.]:
#     print("\n*****")
#     print('x0=',x0)
#     fname = 'results/analyticalCompareson_periodic/infAnchoring_longT/' + str(x0)+'.txt'
#     # nmax = int(Ltmax/x0)
#     nmax = 100
#     ns = np.arange(10,nmax+1,10,dtype=int)
#     print(ns)
#     # ns = [5,10,15,20,30,40,50,70]
#     sim_vel =[]
#     for n_suckers in ns:
#         print('\n+++')
#         print(n_suckers)
#         print()
#         env = Environment(n_suckers,x0,sim_shape,t_position, carrierMode = 1,omega =0.1,isOverdamped=True)
#         env.equilibrate(1000)
#         for k in range(10000):
#             action = [0]*n_suckers
#             s_id,plot,ind=plot_Optimalpulse_tlength(env._t,env.x0,env._nsuckers,env.wavelength,env.omega)
#             #print(s_id) 
#             for s in s_id:
#                 action[s]=1
#             env.step(action)
#         print("average velocity (normalized) =",env.get_averageVel()/env.x0)
#         sim_vel.append(env.get_averageVel()/env.x0)
#     # print(ns*env.x0)
#     print(ns)
#     print(sim_vel)

#     # fig.plot(ns*env.x0,sim_vel,'o',label ='x0 = %.3f'%x0)
#     fig.plot(ns,sim_vel,'o',label ='x0 = %.3f'%x0)
#     plt.legend()
#     plt.pause(0.1)
#     # np.savetxt(fname,np.column_stack((ns*env.x0,np.round(sim_vel,6))),fmt='%.2f\t%.6f',header = 'Lt\tsim_vel\t\tperiod = %d'%Environment.N)
# input()

#Check influence of periodicity
#I expect there isn't.. why ?
def main(figure=None):

    
    n = int(input("choose: 1 =  periodicity study inf anchoring. 2 = continuity study epsilon dependency\n"))
    if n==1:
        x0=1
        omega = 0.1
        deltaT = 0.1
        if figure is not None:
            fig = figure
        else:
            print("new figure object created")
            plt.figure()
            plt.ion()
            fig = plt.subplot(xlabel='periodicity', ylabel ='$v_{CM}/l_0$')
            n = np.arange(5,110)
            fig.plot(n,anal_vel_l0norm(n,omega),label="analytical prediction")
        
        # print("******Trying with -alpha and +beta ******\n")
        print("Checking purely influence of N vs Nsuckers")
        
        # fig = plt.subplot(xlabel='N suckers', ylabel ='$v_{CM}/x_0$')
        
        
        sim_shape = (510,)
        t_position=10000

        # fname = 'results/N_dependency_Lt100_x01.txt'
        periodicity = [5, 8, 10, 12, 15, 20, 25, 30, 35,50,100]
        # periodicity = [5,10,20,30,40,50,60,70,80,100,200,400]
        # periodicity = [5,10,25,50,100,150,180,200,250,400,600]
        
        # ns=[100]
        
        # amplitude = x0/x0Fraction
        print("x0Fraction = ",x0Fraction)
        
        # fig.title.set_text('Effect of periodicity, delta t= %.2f, omega=%.2f'%(deltaT,omega))
        plt.show()
        N_max = 2*np.pi/np.sqrt(omega)
        print("Nmax=",N_max)
        input()
        # 
        # ns = [20,200,400,600]

        # nsmax=ns[-1]
        
    
        # print ("Epsilon = ",epsilon)
        colors = ['blue','red','orange','green','black']
        cc=0
        # for n_suckers in ns:
        #     print("\n ******** NSuck=%d ********\n"%n_suckers)
        sim_vel=[]
        for N in periodicity:
            ns_plot=[]
            print("+++++++periodicity = ", N)
            Environment.N = N
            # sim_vel=[]
            # fig.axhline(anal_vel_l0norm(Environment.N),xmin=0,xmax=nsmax,ls ='dashed',color=colors[cc],label='analytical prediction N=%d'%N)
            
            # # # fig.plot(nsmax/N,anal_vel_l0norm(Environment.N),'*',lw=4, label='analytical prediction N=%d'%N)
            # # last_color = fig.get_lines()[0].get_color()
            # # print(last_color)
            # print(ns)
            
            # for n_suckers in ns:
            #     if (n_suckers<N):
            #         print("skipping")
            #         continue
            #     ns_plot.append(n_suckers)
            #     print('----------')
            #     print(n_suckers)
            n_suckers = N
            #__init__(self,n_suckers,sim_shape,t_position,tentacle_length = 10,carrierMode = 1,omega=0.1,is_Ganglia = False,isOverdamped = True, nGanglia =1,fixN=False)
            env = Environment(n_suckers,sim_shape,t_position, omega =omega)
            if env.info["isPeriodic"]:
                periodic = True
                type_of_T = "periodic tentacle"
            else:
                periodic =False
                type_of_T = "finite tentacle"
                Q = actionValue(env.info,hiveUpdate=True)
            # print(Q._refPolicy)
                hivePol = {'->|<-': 0, '->|->': 0, '<-|<-': 1, '<-|->': 0, 'base|<-': 1, 'base|->': 0, '->|tip': 0, '<-|tip': 1}
                Q.loadPolicy(hivePol)
            # print(Q._refPolicy)
            env.deltaT = deltaT
            print("delta T = ",env.deltaT)
            print('normalized prediction',anal_vel_l0norm(env.N,env.omega))
            env.equilibrate(1000)
            state = env.get_state()
            for k in trange(10000):
                action = [0]*n_suckers
                if periodic ==False:
                    onPolAction = Q.getOnPolicyAction(state)
                # print(state)
                # print(onPolAction)
                
                # s_id,plot,ind=plot_Optimalpulse_tlength(env._t,env.x0,env._nsuckers,env.wavelength,env.omega)
                ns,ids = optimum_impulse(env._t,env.omega,env.N,env._nsuckers)
                #print(s_id) 
                for s in ids:
                    action[s]=1
                # print("--")
                # print(action)
                if periodic == False:
                    action[0] = onPolAction[0]#base
                    action[-1] = onPolAction[-1]#tip
                # print(action)
                # input()
                state,r,t=env.step(action)
            print("average velocity (normalized) =",env.get_averageVel()/env.x0)
            sim_vel.append(env.get_averageVel()/env.x0)
            # fig.plot(np.array(ns_plot),sim_vel,'o',c=colors[cc],label="periodicity=%d"%N)
            # cc+=1
            # plt.legend()
            # plt.pause(0.01)
            # fname = "results/periodicity_effect_N"+str(N)+".txt"
            # np.savetxt(fname,np.column_stack((np.array(ns),np.round(sim_vel,6))),fmt='%d\t%.6f',header = 'Nsucker\tsim_vel\t\tperiodicity= %d'%(env._nsuckers))
        fig.plot(periodicity,sim_vel,'o',label="implementation "+type_of_T)
        print(periodicity,sim_vel)
        plt.legend()
        input()



    #-------------------------


    elif n==2:

        #TODO 
        #farlo con alpha e beta nelle varie combinazioni
        #con e senza 2pi correction

        #save_results
        FOLDER = "results/epsilon_x0/"


        sim_shape = (100,)
        t_position = 1000

        omega = 0.1
        deltaT = 0.1

        x0min = 0.01
        x0max = 0.5
        x0s = np.arange(x0min,x0max,0.01)
        x0s = np.insert(x0s,[0],[0.005])
        # x0s = np.insert(x0s,[0,0],[0.001,0.005])
        # x0min = 0.05
        # x0max = 0.5
        # x0s = np.arange(x0min,x0max,0.025)

        print(x0s)
        for wavelength in [2]:
        # wavelength =2
        
            plt.figure()
            plt.ion()
            name = "wavelength "+str(wavelength)+"_longT"
            fig = plt.subplot(xlabel='1/x0', ylabel ='$v_{CM}/x0$')
            fig.title.set_text("wavelenght "+str(wavelength)+"_small omega=%.3f, deltaT=%.2f"%(omega,deltaT))
            print("\n** WAVELENGTH = %f **\n"%wavelength)
        # Lt = [wavelength,100*wavelength]
            Lt = [wavelength]
            
            n = np.arange(2,wavelength/0.001)
            x1 = n/wavelength #1/x0 N = wavelength/x0
            fig.plot(x1,anal_vel_l0norm(n,omega = omega),label="analytical epsilon = 1, wavelength = %d"%wavelength)
            plt.show()
            input()
            for l_tentacle in Lt:
                print("\nTentacle length = ",l_tentacle)
                sim_vel=[]
                Np=[]
                for x0 in x0s:
                    print("x0 = ",x0)
                    N = int(wavelength/x0)
                    print("periodicity (wavelength/x0) = ",N)
                    #must be the first integer multiple
                    print(int(np.ceil(l_tentacle/x0)))
                    n_suckers = int(np.ceil(l_tentacle/x0)/N)*N
                    print("n suckers = ",n_suckers)
                    env = Environment(n_suckers,x0,sim_shape,t_position, carrierMode = 1,wavelength=wavelength,omega =omega,isOverdamped=True,setEpsilon=True)
                    print("delta T = ",env.deltaT)
                    env.deltaT = deltaT
                    env.equilibrate(1000)
                    Np.append(env.N)
                    for k in trange(10000):
                        action = [0]*n_suckers
                        # s_id,plot,ind=plot_Optimalpulse_tlength(env._t,env.x0,env._nsuckers,env.wavelength,env.omega)
                        ns,ids = optimum_impulse(env._t,env.omega,env.N,env._nsuckers)
                        for s in ids:
                            action[s]=1
                        env.step(action)
                    norm_vel = env.get_averageVel()/x0
                    print("average velocity (normalized) =",norm_vel)
                    sim_vel.append(norm_vel)
                
                fig.plot(1/x0s,sim_vel,'o',label="Lt = %d , wavelength = %d"%(l_tentacle,wavelength))
                fname =  FOLDER + name+"_Lt"+str(l_tentacle)+".txt"
                np.savetxt(fname,np.column_stack((np.array(1./x0s),np.round(sim_vel,6))),fmt='%.2f\t%.6f',header = '1/x0(~Ns)\tsim_vel\t\wavelength= %.1f'%(env.wavelength))
                plt.legend()
                plt.pause(0.1)
        input()

    else:
        exit()

    return fig

if __name__ == "__main__":
    main()












    #Fix Lt
    # plt.figure()
    # plt.ion()

    # x0_min = 0.01
    # x0_max = 1
    # fig = plt.subplot(xlabel='$N_{suckers}$', ylabel ='$v_{CM}/L_t$')

    # # l0_inv = np.linspace(0.1,10,nmax)
    # n= np.arange(3,nmax+1,1)
    # fig.plot(n,anal_vel_Ltnorm(n),c='black',ls ='dashed',label='analytical prediction')

    # # fig.set_ylim([0, anal_vel_l0norm()+anal_vel_l0norm()/2])

    # plt.legend()
    # plt.show()
    # input()

    # # print('normalized prediction',anal_vel_l0norm())
    # # while(1):
    # #     Lt = int(input('insert tentacle length\n'))
    # for Lt in [10,15,20,25,30]:
    #     print('\n******')
    #     print('Lt=',Lt)
    #     nmax = int(Lt/x0_min)
    #     # ns =np.append([np.rint(Lt/x0_max)],np.arange(5,nmax+1,5,dtype=int))
    #     ns = np.arange(5,nmax+1,5,dtype=int)
    #     print(ns)
    #     # ns = [5,10,15,20,30,40,50,70]
    #     sim_vel =[]
    #     for n_suckers in ns:
    #         print(int(n_suckers))
    #         env = Environment(int(n_suckers),sim_shape,t_position,tentacle_length=Lt, carrierMode = 1,omega =0.1,isOverdamped=True) #this embeds rescaling of x0 according to Lt
    #         env.equilibrate(1000)
    #         for k in range(10000):
    #             action = [0]*int(n_suckers)
    #             s_id,plot,ind=plot_Optimalpulse(env._t,env._nsuckers,env.omega)
                
    #             #print(s_id) 
                
    #             action[s_id]=1
    #             env.step(action)
    #         print("average velocity (NOT normalized) =",env.get_averageVel())
    #         sim_vel.append(env.get_averageVel()/env.tentacle_length)
    #     print(sim_vel)
    #     fig.plot(np.array(ns),np.array(sim_vel)/Lt,label ='Lt = %i'%Lt)
    #     plt.legend()
    # input()