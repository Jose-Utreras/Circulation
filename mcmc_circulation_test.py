import yt
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.interpolate import interp1d
from astropy.table import Table , Column ,vstack,hstack
from astropy.nddata.utils import block_reduce
import os , sys
import glob
from scipy.optimize import curve_fit
import pickle
from skimage.transform import resize
from common_functions import *
from scipy import fftpack
from yt.utilities.physical_constants import G
from yt.units import pc,Myr,Msun,km,second
from subprocess import check_output
import emcee
def funcion_stds(res,n1,n2,pc,pmin,pmax,dx):
    resolutions=''
    for re in res*dx:
        resolutions+=' '+str(re)
    factor=1.0
    foo = check_output('../C-code/./vsi '+str(n1)+' '+str(n2)+' '+
                   str(pc)+' '+str(pmin)+' '+ str(pmax/2)+ ' ' +str(factor) +' '+resolutions, shell=True).decode("utf-8")
    foo=foo.split('\n')[:-1]
    numbers=[]
    for item in foo:
        numbers.append(float(item))
    numbers=np.array(numbers)*dx**2
    return numbers

def lnlike(theta, x, y, yerr,pmin,pmax,dx):
    n1,n2,pc = theta
    model = funcion_stds(x,n1,n2,pc,pmin,pmax,dx)
    inv_sigma2 = 1.0/yerr**2
    return -0.5*(np.sum((y-model)**2*inv_sigma2 + 2*np.log(yerr)))
def lnprior(theta,pc_lw,pc_up,n1_lw,n1_up,n2_lw,n2_up):
    n1,n2,pc = theta
    if n1_lw < n1 < n1_up and n2_lw < n2 < n2_up and pc_lw < pc < pc_up:
        return 0.0
    return -np.inf
def lnprob(theta, x, y, yerr,pmin,pmax,dx,pc_lw,pc_up,n1_lw,n1_up,n2_lw,n2_up):
    lp = lnprior(theta,pc_lw,pc_up,n1_lw,n1_up,n2_lw,n2_up)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr,pmin,pmax,dx)


eta=2*0.6744897501
kms_to_pcmyr=1.04589549342

cmdargs = sys.argv
L=float(cmdargs[-1])                        # Image size in parsecs
name_file='Test'
difference_map=np.load(name_file+'_vort.npy')     # Vorticity minus smooth function of Omega
difference_map-=np.mean(difference_map)
 # Axisymmetric map at the resolution level

N=len(difference_map)                       # Size image
DX=L/N                                      # Spatial resolution in pc
DA=DX**2                                    # Area of pixel

sigma_t=1.0*kms_to_pcmyr*2*DX               # total error in circulation

p75,p25=np.percentile(difference_map,[75,25])
sigma_v=(p75-p25)/eta


"""
Creating the resolution array which are integers factors
"""
res=list((10**np.linspace(0,np.log10(N),100)+1e-5))
res.sort()
res=np.array(res).astype(int)
res=np.array(list(set(res)))
res.sort()
res=np.array(list(set((N/res).astype(int))))
res.sort()


stds=np.zeros_like(res,dtype=float)
serr=np.zeros_like(res,dtype=float)
for k,R in enumerate(res):
    fake=block_reduce(difference_map, R, func=np.sum)/R**2
    lf=len(fake.ravel())
    stds[k]=np.std(fake.ravel())
stds[-1]=4*stds[-2]
serr=1.5*stds*res/res.max()
sve=0.25*kms_to_pcmyr*2*DX/res**1.5
stot=np.sqrt(sve**2+serr**2)
stot=np.maximum(stot,0.2*stds)

resolutions=''
sigma_array=''
error_array=''
for sa,re,se in zip(stds,res*DX,stot):
    sigma_array+=' '+str(sa)
    resolutions+=' '+str(re)
    error_array+=' '+str(se)

foo = check_output('./reduce '+str(L)+' '+str(N)+ sigma_array + resolutions + error_array, shell=True).decode("utf-8")
foo=foo.split('\n')[:-1]

print(foo)

pc_min,pc_max=float(foo[0].split('\t')[0]),float(foo[0].split('\t')[1])
n1_min,n1_max=float(foo[1].split('\t')[0]),float(foo[1].split('\t')[1])
n2_min,n2_max=float(foo[2].split('\t')[0]),float(foo[2].split('\t')[1])


ndim, nwalkers = 3, 10
pos=np.zeros((10,3))
pos[:,0]=np.random.uniform(n1_min,n1_max,10)
pos[:,1]=np.random.uniform(n2_min,n2_max,10)
pos[:,2]=np.random.uniform(pc_min,pc_max,10)

pmin=4.0/L
pmax=(N-2.0)/(L)
dx=L/(1.0*N)

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(res, stds, stot,pmin,pmax,dx,pc_min,pc_max,n1_min,n1_max,n2_min,n2_max))

sampler.run_mcmc(pos, 1200)

samples = sampler.chain[:, 100:, :].reshape((-1, ndim))

np.save('Test_samples',samples)

"""
Save tables with three columns, circulation, resolution

Nres=0
f_tab=open('Temp_Files/'+name_file+'_tabla_res.txt','w')
for k,R in enumerate(res):
    fake=block_reduce(difference_map, R, func=np.sum)/R**2
    fake=fake.ravel()
    fake=clean_data(fake)
    for cell in fake:
        f_tab.write(str(cell)+'\t'+str(k)+'\n')
        Nres+=1
f_tab.close()

resolutions=''
for re in res*L/N:
    resolutions+=' '+str(re)

print('begin c code')
ftemp=open('temp.txt','w')
ftemp.write('./mcmc ' +name_file+ ' '+str(L)+' '+str(N)+' '+str(Nres)+' '+
        str(sigma_v)+' '+str(sigma_t)+resolutions)
ftemp.close()
#foo = check_output('./mcmc ' +name_file+ ' '+str(L)+' '+str(N)+' '+str(Nres)+' '+
#        str(sigma_v)+' '+str(sigma_t)+resolutions, shell=True).decode("utf-8")
#foo=foo.split('\n')[:-1]

#print(foo)
"""
