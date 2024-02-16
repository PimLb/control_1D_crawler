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
