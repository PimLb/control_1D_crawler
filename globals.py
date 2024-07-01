red = (220, 20, 60)
blue = (30, 144, 255)
green = (0,128,0)
brown = (128,0,0)
dark_violet = (199,21,133)
violet = (219,112,147)

#some optimization..
# from numba import jit
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import pygame
import copy 
import itertools

def interpret_binary(s:tuple):
    return int("".join(str(ele) for ele in s), 2)

def interpret_thernary(s:tuple):
    return int("".join(str(ele) for ele in s), 3)

def make_binary(baseTen_input:int,padding:int):
    '''
    Padding adds digits with 0 in front, for a readable action instruction
    '''
    # print(padding)
    binary_num = [int(i) for i in bin(baseTen_input)[2:]]
    out = [0]*(padding-len(binary_num)) + binary_num
    return out



import re
comment = ['^#','\n']
comment = '(?:% s)' % '|'.join(comment)
class ReadInput(object):
    def __init__(self,input_file,sim_shape=(20,),t_position=100,tLength=10,isOverdamped=True):
        #OBS: t_position useless now maybe remove everywhere this unused concept
        self.sim_shape = sim_shape
        self.t_position = t_position
        self.isOverdamped = isOverdamped
        self.t_length = tLength
        self.min_lr = 0.001
        self.min_epsilon = 0.001
        self.lr_plateau = 0.001
        self.epsilon_plateau = 0.1
        self.convergence = 0.01

        self.scheduling_episodes = 1000
        self.max_episodes = 1500
        self.polExplEpisodes = 200
        self.isGanglia = False
        # self.steps = 10000
        # try:
        #     self._inFile = open(input_fileName,'r')
        # except:
        #     exit("input.prm should be provided in working folder. EXIT")
        self._inFile = input_file
        self.get()
    def get(self):
        lines = self._inFile.readlines()
        
        for line in lines:
            if(re.match(comment,line)):
                continue
            # print(line)
            match_omega = re.match("(omega\s*=\s*)(\d*\.?\d+)",line,flags=re.IGNORECASE)  
            match_tLength = re.match("(tentacle\s*length\s*=\s*)(\d*\.?\d+)",line,flags=re.IGNORECASE)
            match_ns = re.match("(n[\s*|_]suckers\s*=\s*)(\d*\.?\d+)",line,flags=re.IGNORECASE)
            match_isGanglia= re.match("(control\s*center\s*=\s*)(True|False)",line,flags=re.IGNORECASE)
            match_nGanglia = re.match("(n\s*ganglia\s*=\s*)(\d*\.?\d+)",line,flags=re.IGNORECASE)
            match_hive = re.match("(is\s*hive\s*=\s*)(True|False)",line,flags=re.IGNORECASE)
            match_lr = re.match("(min[\s*|_]lr\s*=\s*)(\d*\.?\d+)",line,flags=re.IGNORECASE)
            match_epsilon = re.match("(min[\s*|_]epsilon\s*=\s*)(\d*\.?\d+)",line,flags=re.IGNORECASE)
            match_lrPl = re.match("(Plateau\s*Exploration\s*min[\s*|_]lr\s*=\s*)(\d*\.?\d+)",line,flags=re.IGNORECASE)
            match_epsilonPl = re.match("(Plateau\s*Exploration\s*min[\s*|_]epsilon\s*=\s*)(\d*\.?\d+)",line,flags=re.IGNORECASE)
            match_schedEp = re.match("(scheduling\s*Episodes\s*=\s*)(\d*\.?\d+)",line,flags=re.IGNORECASE)
            match_maxEpisodes = re.match("(max\s*Episodes\s*=\s*)(\d*\.?\d+)",line,flags=re.IGNORECASE)
            match_polExplEpisodes = re.match("(policy\s*exploration\s*episodes\s*=\s*)(\d*\.?\d+)",line,flags=re.IGNORECASE)
            match_polConvergence = re.match("(convergence\s*=\s*)(\d*\.?\d+)",line,flags=re.IGNORECASE)
            gotNGanglia = True
            if match_omega:
                print(line)
                self.omega = float(match_omega.group(2))
                gotOmega=True
            if match_ns:
                print(line)
                # raise ImportError("Missing number of suckers field")
                self.ns = int(match_ns.group(2))
                gotNS=True
            if match_isGanglia:
                gotNGanglia = False
                print(line)
                gotGangliaInfo = True
                # raise ImportError("Missing training mode (architecture/contrtol center)")
                if match_isGanglia.group(2) == 'True':
                    self.isGanglia = True
                else:
                    self.isGanglia = False
            if self.isGanglia:
                gotNGanglia = True
                if match_nGanglia:
                    print(line)
                    # raise ImportError("Provide number of control centers!")
                    self.nGanglia = int(match_nGanglia.group(2)) 
            if  match_hive:
                print(line)
                gotHive=True
                # raise ImportError("Missing update mode (hive not hive)")
                if match_hive.group(2) == 'True':
                    self.isHive = True
                else:
                    self.isHive = False
            if match_lr:
                print(line)
                self.lr = float(match_lr.group(2))
            # else:
            #     print("min lr not provided. Setting default:",self.lr)    
            if match_epsilon:
                print(line)
                self.epsilon = float(match_epsilon.group(2))
            # else:
            #     print("min epsilon not provided. Setting default:",self.epsilon)
            if match_tLength:
                print(line)
                self.t_length = float(match_tLength.group(2))
            # else:
            #     print("Tentacle length set to default")
            #######
            if match_lrPl:
                print(line)
                self.lr_plateau = float(match_lrPl.group(2))
            # else:
            #     print("min lr for plateau exploration not provided. Setting default:",self.lr_plateau)
            if match_epsilonPl:
                print(line)
                self.epsilon_plateau = float(match_epsilonPl.group(2))
            # else:
            #     print("min epsilon for plateau exploration not provided. Setting default:",self.epsilon_plateau)
            if match_schedEp:
                print(line)
                scheduling_episodes = int(match_schedEp.group(2))
                if self.scheduling_episodes != scheduling_episodes:
                    self.scheduling_episodes = scheduling_episodes
                    print("<WARNING>: setting scheduling episodes different from default: ",self.scheduling_episodes)
            if match_maxEpisodes:
                print(line)
                max_episodes = int(match_maxEpisodes.group(2))
                if self.max_episodes != max_episodes:
                    self.max_episodes=max_episodes
                    print("<WARNING>: setting max episodes different from default: ",self.max_episodes)
            if match_polExplEpisodes:
                print(line)
                polExplEpisodes = int(match_polExplEpisodes.group(2))
                if self.polExplEpisodes != polExplEpisodes:
                    self.polExplEpisodes = polExplEpisodes
                    print("<WARNING>: setting policy exploration episodes different from default: ",self.polExplEpisodes)
            if match_polConvergence:
                print(line)
                polConvergence = float(match_polConvergence.group(2))
                if self.convergence != polConvergence:
                    self.convergence = polConvergence
                    print("<WARNING>: setting plateau convergence criterion different from default: ",self.convergence)
                # match_steps = re.match("steps\s*=\s*)(\d*\.?\d+)",line,flags=re.IGNORECASE)
                # if match_steps:
                #     self.match_steps = int(match_steps.group(2))
                #     print("<WARNING>: setting scheduling episodes different from default: ",self.match_steps)
        if gotGangliaInfo and gotNGanglia and  gotHive and gotOmega and gotNS:
            pass
        else:
            raise ImportError("MISSING ESSENTIAL INFO")


