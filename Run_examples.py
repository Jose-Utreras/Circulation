import yt
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.interpolate import interp1d
from astropy.table import Table , Column ,vstack,hstack
from Vorticity_Difussion import *
yt.enable_parallelism()

import sys
import os
cmdargs = sys.argv

Ncores=2
directory='Examples'
N=1200
#narr=[0.1,0.25,0.5,0.75,1.0,1.25,1.5]
narr=np.linspace(0.0,1.99,20)
Sarr=np.arange(50,600,50)
SDiff=[1,2,3,4,5,6,8,10,12,14,16,20,25]
#SDiff='None'
par=[]

for n in narr:
    for S in Sarr:
        par.append([n,S])

my_storage = {}

for sto, li in yt.parallel_objects(par, Ncores, storage = my_storage):

    output = save_example(N,li[0],li[1],SDiff,directory)
    sto.result_id = 'n%.1f_'%li[0]+'S%.1f'%li[1]
    sto.result = output
