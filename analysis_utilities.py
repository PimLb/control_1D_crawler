#MISTERY: I need a 2pi correction to make things work better
import numpy as np
from env import x0Fraction
# def u0_cont(t:float,s:float,N:int,omega,optimalShift,carrierMode=1) -> float:
#         '''
#         N = number of suckers
#         '''
#         # amplitude = tentacle_length/(N*x0Fraction)
#         # the k dependent term mimics some time delay in the propagation 
#         wavelengthFraction = carrierMode
#         # print (wavelengthFraction)
#         k = 2*np.pi/N #N*x0 but aslo is s*x0 so they simplify out
#         # diffusion = elastic_constant/zeta
#         # alpha = math.atan(omega*x0*x0/(diffusion*k*k))
#         # alpha=0 #appears also in equation for  pulse
#         alpha = np.arctan(omega/(k*k))#with a 2pi correction it works much better
#         # alpha = np.arctan(omega*N*N/(4*np.pi*np.pi)) Actual paper formula
#         # print(alpha)
#         # A = amplitude/k * np.cos(alpha)#USELESS SINCE IS JUST A SCALING INDEPENDENT FROM S
#         A=1
#         return A*np.cos(omega*t - k*(s) +alpha - optimalShift),A

# def l0_cont(t:float,s:float,N:int,tentacle_length,omega=0.1,carrierMode=1) -> float:
#         '''
#         N = number of suckers
#         '''
#         amplitude = tentacle_length/(N*x0Fraction)
#         wavelengthFraction = carrierMode
#         k = 2*np.pi*wavelengthFraction/(N) #N*x0 but aslo is s*x0 so they simplify out
#         A=amplitude
#         A=1
#         return A*np.cos(omega*t - k*s ),A

# def plot_l0(t,N_suckers,omega,followSuck=5):
#     tlength = N_suckers
#     s = np.arange(1,tlength,discretization)
#     l,amplitude = l0_cont(t,s,N_suckers,omega)
#     peak_ind = np.where(abs(l-amplitude)<=0.00001)
#     ind = np.where(abs(s-followSuck)<=0.00001)
#     return s,l,ind
# def plot_Optimalpulse(t,N_suckers,omega):
#     # fig.clear()
#     # tlength = N_suckers #[0-->N_sucker includes also virtual sucker]
#     s = np.arange(0,N_suckers-1,0.0001)
#     l,max = u0_cont(t,s,N_suckers,omega,optimalShift=2*np.pi/2)
#     target = max
#     # fig.plot(s,l)
#     plot = s,l
#     # pulse = np.where(abs(l-(amplitude+x0))<=0.001)
#     pulse = np.where(abs(l-target)<=0.000005)
#     # print(pulse[0])
#     try:
#         p = pulse[0][0]
#     except:
#         p=0
#         pulse=-1
#     # print(p)
#     # if pulse[0].size>0:
#     # sucker = np.rint(s[pulse][0])#closer one
    
#     sucker = np.rint(s[p])#closer one
#     # print(s[p],sucker)
#     # print(sucker)
#     # if sucker==N_suckers:
#     #      sucker =0
#          # Virtual sucker N-->base
    
#     # fig.plot(s[pulse],l[pulse],'o')
#     plot_peaks = s[pulse],l[pulse]
#     plot_peak2= s[p],l[p]
#     return int(sucker),plot,p
#     # else:
#     #     plot_peak=s[-1],l[-1]
#     #     print(l)
#         # return -1,plot,plot_peak


# def plot_Optimalpulse2(t,N_suckers,omega):
#     #true reference function the above is to spot precisely the shift 3/2pi
#     # fig.clear()
#     # tlength = N_suckers #[0-->8]

#     s = np.arange(0,N_suckers-1,0.0001)
#     l,max = u0_cont(t,s,N_suckers,omega,optimalShift=3*np.pi/2)
#     target = 0
#     # fig.plot(s,l)
#     plot = s,l
#     # pulse = np.where(abs(l-(amplitude+x0))<=0.001)
#     # pulse = np.where(abs(l-target)<=0.001)
#     # # print(pulse[0])
#     # try:
#     #     p = pulse[0][-1]
#     # except:
#     #     p=-1
#     #     pulse=-1

#     # # if pulse[0].size>0:
#     # # sucker = np.rint(s[pulse][0])#closer one
#     # sucker = np.rint(s[p])#closer one
#     # # print(sucker)
#     # # fig.plot(s[pulse],l[pulse],'o')
#     # plot_peak = s[pulse],l[pulse]
#     # plot_peak2= s[p],l[p]
#     return plot


# def anal_vel(n,tentacle_length,omega=0.1):
#     phase_vel = omega/(2*np.pi) *tentacle_length

#     amplitude_fraction =1./x0Fraction
#     k = 2*np.pi/(n)
#     alpha = np.arctan(omega/(k*k)) #with 2*np.pi correction seems to work better..
#     reducedOmega = omega/(k*k)
#     cos_alpha = 1/(np.sqrt(1+reducedOmega*reducedOmega))
#     # print(alpha,np.cos(alpha))
#     return phase_vel * amplitude_fraction * cos_alpha#omega*n*carrierMode/(2*np.pi) *amplitude * np.cos(alpha)


# for n_suckers in ns:
#     print("\n\nNSUCKERS= ",n_suckers)
#     print("\n\nNSUCKERS= ",n_suckers)
#     env = Environment(n_suckers,sim_shape,t_position, carrierMode = 1,omega=0.1,isOverdamped=True)
#     env.equilibrate(1000)
#     for k in range(20000):
#         action = [0]*n_suckers
#         s_id,plot,ind=plot_Optimalpulse(env._t,env._nsuckers,env.omega)
#         action[s_id]=1
#         env.step(action)
#     print("\nAverage vel= ",env.get_averageVel())
#     semi_anal_vel_finite100.append(env.get_averageVel())
# for k in range(1000):
#     ...:      ...:     action = [0]*n_suckers
#     ...:      ...:     s_id,plot,plot_peak,ind=plot_Optimalpulse(env._t,env._nsu
#     ...: ckers,env.omega)
#     ...:      ...:     s_id_2,plot_2,plot_peak_2,plot_peak2_2=plot_Optimalpulse2
#     ...: (env._t,env._nsuckers,env.omega)
#     ...:      ...:     action[s_id]=1
#     ...:      ...:     #print(ind)
#     ...:      ...:     env.step(action)
#     ...:      ...:     if k%10==0:
#     ...:      ...:         fig.clear()
#     ...:      ...:         fig.plot(plot[0],plot[1])
#     ...:      ...:         fig.plot(plot_2[0],plot_2[1])
#     ...:      ...:         #fig.plot(plot_peak[0],plot_peak[1],'o')
#     ...:      ...:         fig.plot(plot[0][ind],plot[1][ind],'o')
#     ...:      ...:         fig.plot(plot_2[0][ind],plot_2[1][ind],'o')#fig.plot(
#     ...: plot_peak2_2[0],plot_peak2_2[1],'o')
#     ...:      ...:         plt.pause(0.001)
#     ...:      ...:         env.render()


def optimum_impulse(t,omega,N,n_suckers):
    n_pulse = int(n_suckers/N)
    k = 2*np.pi/N
    alpha =  np.arctan(omega/(k*k))
    beta = 3/2*np.pi
    n0 = (omega*t-alpha+beta)/k
    #  print("reference",n0)
    ids=[]
    ns=[]
    for i in range(n_pulse):       
        n = (n0+N*i)%n_suckers
        # print(n,i,N*i)
        ns.append(n)
        ids.append(int(np.floor(n)))
    # id = int(np.floor(n))
    # print(ns,ids)
    # input()
    return ns,ids

def u0(s,t,omega,N,amplitude,l0):
    k = 2*np.pi/N
    alpha =  np.arctan(omega/(k*k))
    A = amplitude/k * np.cos(alpha)
    u = A*np.cos(omega*t - k*s/l0 -alpha)
    return u


def anal_vel_l0norm(N,omega):
    k = 2*np.pi/N
    amplitude_fraction = 1/x0Fraction
    phase_vel = omega/k
    alpha = np.arctan(omega/(k*k)) #MISTERIOUS 2pi correction here in front of omega
    # reducedOmega = omega/(k*k)
    # cos_alpha = 1/(np.sqrt(1+reducedOmega*reducedOmega))
    return  amplitude_fraction * phase_vel * np.cos(alpha)#cos_alpha