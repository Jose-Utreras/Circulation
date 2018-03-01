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

narr=np.arange(0.0,1.95,0.1)
Sarr=np.arange(1,1004,50)
SDiff=[0.1,0.2,0.5,1,2,3,4,5,6,8,10,12,14,16,20,25,50,75,100]
N=1200

my_storage = {}

for sto, nd in yt.parallel_objects(narr, Ncores, storage = my_storage):

    output = grid_table_n(N,nd,snarr,sdarr,directory)
    sto.result_id = 'n%.1f_'%nd
    sto.result = output
