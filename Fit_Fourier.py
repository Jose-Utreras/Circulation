from Vorticity_Difussion import *
from Enzo_Fields import *
from Turbulence_noise import *
from Turbulence_Fourier import *
import matplotlib.pyplot as plt
import sys

import time
import emcee
import numpy as np
from schwimmbad import MPIPool

def lnlike(theta, mapa, L,res,P25,P50,P75):
    n1,n2,kc,fc = theta
    new_map=add_vorticity_to_map(mapa,L,n1,n2,kc,fc)
    res,Per=Percentile_profiles(new_map,[25,50,75])
    Q25=Per[:,0]
    Q50=Per[:,1]
    Q75=Per[:,2]

    e1=np.mean((Q25-P25)**2/res)/10
    e2=np.mean((Q50-P50)**2/res)/10
    e3=np.mean((Q75-P75)**2/res)/10
    print(e1+e2+e3)
    return -0.5*(e1+e2+e3)

def lnprior(theta,N):
    n1, n2, kc,fc = theta
    if -1 < n1 < 5 and n1 < n2 < 9.0 and 1 < kc < N and 0.5 < fc < 100:
        return 0.0
    return -1e32

def lnprob(theta, mapa,N, L,res,P25,P50,P75):
    lp = lnprior(theta,N)
    if not np.isfinite(lp):
        return -1e32
    return lp + lnlike(theta, mapa, L,res,P25,P50,P75)

mapa=np.load('NWR10025_vort.npy')
#noise_map=get_noise_map(mapa)
#average_map=mapa-noise_map
omega_map=np.load('NWR10025_omeg.npy')
#sigma_map=np.load('NWR10025_sigma.npy')
#mass_map=np.load('NWR10025_mass.npy')

mass_map*=sigma_map.sum()/mass_map.sum()
omega_map*=mapa.sum()/omega_map.sum()
omega_map=correct_map(mapa,omega_map)
#average_map*=mapa.sum()/average_map.sum()

res,Per=Percentile_profiles(mapa,[25,50,75])
P25=Per[:,0]
P50=Per[:,1]
P75=Per[:,2]

with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    ndim, nwalkers = 4, 10
    pos = [[1.33,4,80,11] + np.random.randn(ndim) for i in range(nwalkers)]
    nsteps = 100

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(omega_map,len(omega_map),4e4,res,P25,P50,P75),pool=pool)
    start = time.time()
    sampler.run_mcmc(pos, nsteps)
    end = time.time()
    print(end - start)
