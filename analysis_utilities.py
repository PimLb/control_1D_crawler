#Contains accessory funcitons used for testing, comparing, plotting ecc
# More of a workbook note

import numpy as np

# NEW CLEANER
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
    alpha = np.arctan(omega/(k*k))
    # reducedOmega = omega/(k*k)
    # cos_alpha = 1/(np.sqrt(1+reducedOmega*reducedOmega))
    return  amplitude_fraction * phase_vel * np.cos(alpha)#cos_alpha