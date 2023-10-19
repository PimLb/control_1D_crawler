#Contains accessory funcitons used for testing, comparing, plotting ecc
# More of a workbook note

import numpy as np

# 1D utilities for tentacle locomotion
zeta = 1
elastic_constant = 1 
carrierMode = 1
x0Fraction = 10
tentacle_length = 30

def u0_cont(t:float,s:float,N:int,omega,optimalShift,carrierMode=1) -> float:
        '''
        N = number of suckers
        '''
        amplitude = tentacle_length/(N*x0Fraction)
        # the k dependent term mimics some time delay in the propagation 
        wavelengthFraction = carrierMode
        # print (wavelengthFraction)
        k = 2*np.pi*wavelengthFraction/(N) #N*x0 but aslo is s*x0 so they simplify out
        # diffusion = elastic_constant/zeta
        # alpha = math.atan(omega*x0*x0/(diffusion*k*k))
        alpha=0 #appears also in equation for  pulse
        alpha = np.arctan(omega*N*N/(2*np.pi))
        # print(alpha)
        # A = amplitude/k * np.cos(alpha)#USELESS SINCE IS JUST A SCALING INDEPENDENT FROM S
        A=1
        return A*np.cos(omega*t - k*s +alpha - optimalShift),A

def plot_Optimalpulse(t,N_suckers,omega):
    # fig.clear()
    tlength = N_suckers #[0-->N_sucker-1 included]
    s = np.arange(1,tlength,0.0001)
    l,max = u0_cont(t,s,N_suckers,omega,optimalShift=2*np.pi/2)
    target = max
    # fig.plot(s,l)
    plot = s,l
    # pulse = np.where(abs(l-(amplitude+x0))<=0.001)
    pulse = np.where(abs(l-target)<=0.000005)
    # print(pulse[0])
    try:
        p = pulse[0][0]
    except:
        p=-1
        pulse=-1
    
    # if pulse[0].size>0:
    # sucker = np.rint(s[pulse][0])#closer one
    
    sucker = np.rint(s[p])#closer one
    # print(s[p],sucker)
    if sucker==N_suckers:
         sucker =0
         # Virtual sucker N-->base
    # print(sucker)
    # fig.plot(s[pulse],l[pulse],'o')
    plot_peaks = s[pulse],l[pulse]
    plot_peak2= s[p],l[p]
    return int(sucker),plot,plot_peaks,plot_peak2
    # else:
    #     plot_peak=s[-1],l[-1]
    #     print(l)
        # return -1,plot,plot_peak


def plot_Optimalpulse2(t,N_suckers,omega):
    #true reference function the above is to spot precisely the shift 3/2pi
    # fig.clear()
    tlength = N_suckers #[0-->8]
    s = np.arange(1,tlength,0.001)
    l,max = u0_cont(t,s,N_suckers,omega,optimalShift=3*np.pi/2)
    target = 0
    # fig.plot(s,l)
    plot = s,l
    # pulse = np.where(abs(l-(amplitude+x0))<=0.001)
    pulse = np.where(abs(l-target)<=0.005)
    # print(pulse[0])
    try:
        p = pulse[0][-1]
    except:
        p=-1
        pulse=-1

    # if pulse[0].size>0:
    # sucker = np.rint(s[pulse][0])#closer one
    sucker = np.rint(s[p])#closer one
    # print(sucker)
    # fig.plot(s[pulse],l[pulse],'o')
    plot_peak = s[pulse],l[pulse]
    plot_peak2= s[p],l[p]
    return int(sucker),plot,plot_peak,plot_peak2


def anal_vel(n,omega=0.1):
    amplitude = tentacle_length/(n*x0Fraction)
    alpha = np.arctan(omega*n*n/(2*np.pi))
    print(alpha,np.cos(alpha))
    return omega*n*carrierMode/(2*np.pi) *amplitude * np.cos(alpha)