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

def lnlike(theta, mapa, L,res,Neg,Tot):
    n1,n2,kc,fc = theta
    new_map=add_vorticity_to_map(mapa,L,n1,n2,kc,fc)
    r_n,N_n,T_n=Negative_profile(new_map)
    sigma=1/Tot**0.5
    error1=-0.5*(Neg-N_n)**2/(2*sigma**2)
    error2=-0.5*np.log(2*np.pi*sigma**2)
    error=np.sum(error1)+2*len(error1)
    print(np.exp(error),error,len(error1))
    plt.figure(figsize=(10,10))
    plt.errorbar(res,Neg,sigma)
    plt.plot(r_n,N_n)
    plt.xscale('log')
    plt.ylim(0,0.3)
    #plt.savefig('Temp_Figures/Sample_n1_%04.2f' %n1+'_n2_%04.2f' %n2+'_kc_%04d' %int(kc)+'_fc_%04.1f'%fc+'.png')
    plt.savefig('Temp_Figures/Sample_%13.7f.png' %np.exp(error))
    plt.close()
    return error

def lnprior(theta,N):
    n1, n2, kc,fc = theta
    if 0 < n1 < 5 and n1 < n2 < 9.0 and 1 < kc < N and 0.5 < fc < 100:
        return 0.0
    return -1e32

def lnprob(theta, mapa,N, L,res,Neg,Tot):
    lp = lnprior(theta,N)
    if not np.isfinite(lp):
        return -1e32
    return lp + lnlike(theta, mapa, L,res,Neg,Tot)

mapa=np.load('NWR10025_vort.npy')
omega_map=np.load('NWR10025_omeg.npy')

omega_map*=mapa.sum()/omega_map.sum()
omega_map=correct_map(mapa,omega_map)

res,Neg,Tot=Negative_profile(mapa)


with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    ndim, nwalkers = 4, 10
    pos = [[1.33,4,80,11] + np.random.randn(ndim) for i in range(nwalkers)]
    nsteps = 20

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(omega_map,len(omega_map),4e4,res,Neg,Tot),pool=pool)
    start = time.time()
    sampler.run_mcmc(pos, nsteps)
    end = time.time()
    np.save('sampler',sampler)
    print(end - start)
