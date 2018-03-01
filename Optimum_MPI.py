from Vorticity_Difussion import *
from Enzo_Fields import *
from Turbulence_noise import *
import matplotlib.pyplot as plt
import sys

yt.enable_parallelism()
import os
import glob

cmdargs = sys.argv
dN=int(cmdargs[-1])
N=int(cmdargs[-2])
Nbins=int(cmdargs[-3])
f2=float(cmdargs[-4])
f1=float(cmdargs[-5])
R_cut=float(cmdargs[-6])
L=float(cmdargs[-7])
name=cmdargs[-8]

coll=False
energy=False
mass=False
radial=False
Ncores=1

names=glob.glob(name+'*_vort.npy')
for i,val in enumerate(names):
    names[i]=val.split('_')[0]
my_storage = {}


Radius=[]
Velocity=[]
Names=[]
for sto, name in yt.parallel_objects(names, Ncores, storage = my_storage):

    output = optimum_rcut(name,L,R_cut,f1,f2,'half','larson',N,dN,coll,energy,mass)
    sto.result_id = name
    sto.result = output

if yt.is_root():
    for fn, vals in sorted(my_storage.items()):
        Radius.append(vals[0])
        Velocity.append(vals[1])
        Names.append(fn)
    Radius=np.array(Radius)
    Velocity=np.array(Velocity)

Rcen=[]
Radius=[]
Velocity=[]
Names=[]
for sto, name in yt.parallel_objects(names, Ncores, storage = my_storage):

    output = a,b,c=optimum_radial_rcut(name,L,R_cut,Nbins,'larson',N,dN,f1,f2,coll,energy,mass)
    sto.result_id = name
    sto.result = output

if yt.is_root():
    for fn, vals in sorted(my_storage.items()):
        Rcen.append(vals[0])
        Velocity.append(vals[1])
        Radius.append(vals[2])
        Names.append(fn)
    Radius=np.array(Radius)
    Velocity=np.array(Velocity)
    print(Radius)
    print(Velocity)
