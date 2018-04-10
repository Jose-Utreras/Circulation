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

eta=2*0.6744897501
kms_to_pcmyr=1.04589549342

cmdargs = sys.argv
name_file=cmdargs[-2]
L=float(cmdargs[-1])                        # Image size in parsecs


vort_map=np.load(name_file+'_vort.npy')     # Loading vorticity map
omeg_map=np.load(name_file+'_omeg.npy')     # Loading omega map, smooth axisymmetric azimuthal angular velocity
omeg_map=correct_map(vort_map,omeg_map)     # Normalizing omega map to have the same circulation as the vorticity map
difference_map=vort_map-omeg_map            # Vorticity minus smooth function of Omega
difference_map-=np.mean(difference_map)
detailed_map=simple_symmetric_map(vort_map) # Axisymmetric map at the resolution level

N=len(vort_map)                             # Size image
DX=L/N                                      # Spatial resolution in pc
DA=DX**2                                    # Area of pixel

p75,p25=np.percentile(omeg_map-detailed_map,[75,25])

sigma_c=(p75-p25)/eta                       # error in circulation from profile
sigma_s=1.0*kms_to_pcmyr*2*DX               # error in circulation from subgrid turbulence

sigma_t=np.sqrt(sigma_c**2+sigma_s**2)      # total error in circulation

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

"""
Save tables with three columns, circulation, resolution
"""
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

print(sigma_v)
print('begin c code')
ftemp=open('temp.txt','w')
ftemp.write('./mcmc ' +name_file+ ' '+str(L)+' '+str(N)+' '+str(Nres)+' '+
        str(sigma_v)+' '+str(sigma_t)+resolutions)
ftemp.close()
#foo = check_output('./mcmc ' +name_file+ ' '+str(L)+' '+str(N)+' '+str(Nres)+' '+
#        str(sigma_v)+' '+str(sigma_t)+resolutions, shell=True).decode("utf-8")
#foo=foo.split('\n')[:-1]

#print(foo)
