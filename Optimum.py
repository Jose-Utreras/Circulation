import yt
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.interpolate import interp1d
from astropy.table import Table , Column ,vstack,hstack
from scipy.ndimage.filters import gaussian_filter as difussion
from astropy.nddata.utils import block_reduce
from Vorticity_Difussion import *
import time
yt.enable_parallelism()
import os
import glob

Ncores=2
Nbins=1
start=1
kmin=200
kmax=700
Nk=16
name='NWR10025'
#if yt.is_root():
#    kt,kd=Circulation_negative_optimum(name,kmin,kmax,Nk)
#    print(kt,kd)
#    write_kt_kd(name,kt,kd)
kk=Circulation_negative_optimum_mpi(name,kmin,kmax,Nk,Ncores)
time.sleep(5)
try:
    kt,kd=kk
    write_kt_kd(name,kt,kd)
    print(kt,kd)
except:
    pass

KK=Circulation_negative_turbulence(name,Nbins,start,kmin,kmax,Nk,Ncores)
try:
    KT,KD=KK
    write_KT_KD(name,KT,KD,Nbins)
    print(KT,KD)
except:
    pass

time.sleep(5)

apply_negative_radial_fit(name,Ncores)

if yt.is_root():
    os.system('rm *temp*.npy  *KK*.npy')
    compare_negative(name,Nbins)
