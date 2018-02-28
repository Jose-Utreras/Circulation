import yt
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.interpolate import interp1d
from astropy.table import Table , Column ,vstack,hstack
from scipy.ndimage.filters import gaussian_filter as difussion
from astropy.nddata.utils import block_reduce
yt.enable_parallelism()
import os
import glob
from scipy.optimize import curve_fit
import pickle
from skimage.transform import resize
from common_functions import *
from scipy import fftpack
from yt.utilities.physical_constants import G
from yt.units import pc,Myr,Msun,km,second

def vk_spectrum(k,k0,kcore):
    A=(k0+kcore)**(-4/3)*k0**2
    return np.piecewise(k,[k>=k0,k<k0],[lambda k: A/k**2, lambda k: (k+kcore)**(-4/3)])

def Negative_resolution(mapa,L,rho):

    N=len(mapa)
    dx=L/N
    res=list((10**np.linspace(0,np.log10(N),100)+1e-5))
    res.sort()
    res=np.array(res).astype(int)
    res=np.array(list(set(res)))
    res.sort()

    res=np.array(list(set((N/res).astype(int))))
    res.sort()
    Negative=np.zeros_like(res,dtype=float)


    for k,R in enumerate(res):
        fake=block_reduce(mapa, R, func=np.sum)
        N1=len(fake)
        radial=radial_map(fake)
        fake=fake.ravel()
        radial=radial.ravel()*L/N1
        try:
            fake=fake[radial<rho]
            Negative[k]=len(fake[fake<0])/len(fake)
        except:
            Negative[k]=0


    return res, Negative ,dx

def Full_Negative(name):
    tab=Table.read('Circulation_data/Full-Percentiles/Negative/'+name+'-Negative',path='data')
    r0=tab['Resolution']*2
    N0=tab['Number']

    return r0,N0

def Block_Negative(name,use_mass=False):
    tab=Table.read('Circulation_data/Percentiles/Negative/'+name+'-Negative',path='data')
    r0=tab['Resolution']
    if use_mass:
        N0=tab['Mass']
    else:
        N0=tab['Number']

    return r0,N0

def larson_noise(N,L,R_cut,V=1):

    noise=np.random.normal(0,10,size=(N,N))
    dx=L/N
    k1 = fftpack.fftfreq(N,d=dx)

    kx,ky = np.meshgrid(k1, k1)
    KX = fftpack.fftshift( kx )
    KY = fftpack.fftshift( ky )
    K=np.sqrt(KX**2+KY**2)

    R=1/(K +0.5/L)

    F1=np.random.uniform(-1,1,size=(N,N))*(K +0.5/L)**(-1.3333)
    #F1=np.random.uniform(-1,1,size=(N,N))*vk_spectrum(K,1.0/R_cut,0.5/L)
    F1[R<R_cut]=0
    F1=fftpack.fftshift( F1 )

    SS=fftpack.ifft2(F1)
    noise=np.real(SS)
    noise-=np.mean(noise)
    noise*=V/np.std(noise)

    noise*=1.02269032

    return noise

def template_larson_noise(N,L,R_cut,eje):
    """
    eje = 'x' or 'y'

    """
    extra=''
    if R_cut>0:
        extra='_Rcut_%04d' % R_cut

    if os.path.isfile('Temp_Files/v'+eje+'_noise_%04d' %N+extra+'.npy'):
        template=np.load('Temp_Files/v'+eje+'_noise_%04d' %N+extra+'.npy')
    else:
        template=larson_noise(N,L,R_cut,1)
        np.save('Temp_Files/v'+eje+'_noise_%04d' %N+extra,template)
    return template

def noise_from_snapshot(name,L,R_cut):
    V=np.load(name+'_sigma_3D.npy')
    N=len(V)
    noise=larson_noise(N,L,R_cut,1)
    V=V*noise
    #V*=1.0/np.std(V)
    V*=1.02269032
    return V

def template_snapshot_noise(name,L,R_cut,eje):
    """
    eje = 'x' or 'y'

    """
    extra=''
    if R_cut>0:
        extra='_Rcut_%04d' % R_cut

    if os.path.isfile('Temp_Files/snapshot_v'+eje+extra+'.npy'):
        template=np.load('Temp_Files/snapshot_v'+eje+extra+'.npy')
    else:
        template=noise_from_snapshot(name,L,R_cut)
        np.save('Temp_Files/snapshot_v'+eje+extra,template)
    return template

def apply_turbulence(name,factor,R_cut,L,save=False,collapse=False):
    vort=np.load(name+'_omeg.npy')
    aux=np.load(name+'_vort.npy')
    vort*=aux.sum()/vort.sum()
    if collapse:
        s1=np.load(name+'_sigma.npy')
        s2=np.load(name+'_mass.npy')
        vort*=s1/s2
    del aux
    N=len(vort)
    dx=L/N
    extra=''
    if R_cut>0:
        extra='_Rcut_%04d' %R_cut
    if os.path.isfile('Temp_Files/vort_noise_%04d'%N+extra+'.npy'):
        VORT=np.load('Temp_Files/vort_noise_%04d'%N+extra+'.npy')
    else:
        vx=template_larson_noise(N+2,4e4,R_cut,'x')
        vy=template_larson_noise(N+2,4e4,R_cut,'y')

        DVX=(vy[1:-1,2:]-vy[1:-1,:-2])/2
        DVY=(vx[2:,1:-1]-vx[:-2,1:-1])/2

        VORT=-DVY+DVX
        VORT*=dx
        if save:
            np.save('Temp_Files/vort_noise_%04d'%N+ extra,VORT)

    return  vort+factor*VORT

def factor_error(vort,VORT,factor,x0,y0,R_cut,energy_loss=False,name=''):
    ff=factor
    if energy_loss:
        ff=energy_loss_map(name,factor,len(VORT))
    new_vort=vort+ff*VORT
    x,y,dx=Negative_resolution(new_vort,4e4,1.5e4)
    X=x0/x0[0]
    Y=y0
    xmin=max(x.min(),X.min())
    xmax=min(x.max(),X.max())

    Y=Y[(X>=xmin)&(X<=xmax)]
    X=X[(X>=xmin)&(X<=xmax)]

    fun=interp1d(x,y)
    y=fun(X)

    plt.plot(X,y)
    plt.plot(X,Y)
    plt.xscale('log')
    plt.savefig('Temp_Figures/Negative_Rcut_%04d' %R_cut+'_factor_%05.2f' %factor+'.png')
    plt.close()

    return np.sqrt(np.mean((y-Y)**2))

def factor_amplitude(vort,VORT,factor,y0,energy_loss=False,name=''):
    ff=factor
    if energy_loss:
        ff=energy_loss_map(name,factor,len(VORT))

    new_vort=vort+ff*VORT
    x,y,dx=Negative_resolution(new_vort,4e4,1.5e4)
    return y[0]-y0

def load_VORT(name,N,L,R_cut,dx,noise):
    dx=L/N
    if noise=='larson':
        vx=template_larson_noise(N+2,L,R_cut,'x')
        vy=template_larson_noise(N+2,L,R_cut,'y')
    else:
        vx=template_snapshot_noise(name,L,R_cut,'x')
        vy=template_snapshot_noise(name,L,R_cut,'y')

    DVX=(vy[1:-1,2:]-vy[1:-1,:-2])/2
    DVY=(vx[2:,1:-1]-vx[:-2,1:-1])/2
    VORT=-DVY+DVX
    VORT*=dx
    return VORT

def error_factor_for_rcut(name,L,R_cut,f1=1000,f2=0,method='half',noise='larson',collapse=False,energy_loss=False,use_mass=False):
    vort=np.load(name+'_omeg.npy')
    aux=np.load(name+'_vort.npy')
    vort*=aux.sum()/vort.sum()
    del aux
    if collapse:
        s1=np.load(name+'_sigma.npy')
        s2=np.load(name+'_mass.npy')
        vort*=s1/s2
    N=len(vort)
    dx=L/N
    VORT=load_VORT(name,N,L,R_cut,dx,noise)
    x0,y0=Block_Negative(name,use_mass)
    if noise!='larson':
        vort=vort[1:-1,1:-1]
    if method=='half':
        F1=f1
        F2=f2
        delta1 = factor_amplitude(vort,VORT,F1,y0[0],energy_loss,name)
        delta2 = factor_amplitude(vort,VORT,F2,y0[0],energy_loss,name)
        while delta1<0:
            F1*=1.1
            delta1 = factor_amplitude(vort,VORT,F1,y0[0],energy_loss,name)
        while delta2>0:
            F2*=0.9
            delta2 = factor_amplitude(vort,VORT,F2,y0[0],energy_loss,name)

        while (np.abs(F1-F2)/(F1+F2)>0.005):
            F3=0.5*(F1+F2)

            delta3 = factor_amplitude(vort,VORT,F3,y0[0],energy_loss,name)

            if delta1*delta3<0:
                F2=F3
                delta2=delta3
            else:
                F1=F3
                delta1=delta3
    if method=='newton':


        F1=0.5*(f1+f2)
        F2=1.1*F1
        F3=1.2*F1
        aux=F3*2
        d1=factor_amplitude(vort,VORT,F1,y0[0])**2
        d2=factor_amplitude(vort,VORT,F2,y0[0])**2
        while (np.abs(aux-F3)/(aux+F3)>0.005):
            aux=F3
            d3=factor_amplitude(vort,VORT,F3,y0[0])**2

            DD=(d3-d2)*(F3-F1)*(F2-F1)
            DD/=d3*(F2-F1)-d2*(F3-F1)+d1*(F3-F2)*2

            F1=F2
            F2=F3
            F3=F3-DD
            d1=d2
            d2=d3
    error=factor_error(vort,VORT,F3,x0,y0,R_cut,energy_loss,name)

    return error,F3

def optimum_rcut(name,L,Rin,f1,f2,method,noise,N,dN,collapse=False,energy_loss=False,use_mass=False):

    F1=f1
    F2=f2
    ERR=np.zeros(N)
    RC=np.zeros(N)
    FC=np.zeros(N)
    for k in range(N):
        R=Rin+k*dN
        d,FM=error_factor_for_rcut(name,L,R,F1,F2,method,noise,collapse,energy_loss,use_mass)
        ERR[k]=d
        RC[k]=R
        FC[k]=FM
        F1=FM*0.9
        F2=FM*1.1

    Rest=np.average(RC,weights=1.0/(ERR-0.5*ERR.min())**4)
    Fest=np.average(FC,weights=1.0/(ERR-0.5*ERR.min())**4)
    donde=np.where(ERR==ERR.min())
    print(RC[donde])
    if RC[0]==RC[donde]:
        print(name+' IS LOWER')
    if RC[-1]==RC[donde]:
        print(name+' IS HIGHER')
    return Rest,Fest

def radial_rcut(name,L,R_cut,Nbins,noise,f1=50,f2=0,collapse=False,energy_loss=False,use_mass=False):
    vort=np.load(name+'_omeg.npy')
    aux=np.load(name+'_vort.npy')
    vort*=aux.sum()/vort.sum()
    del aux
    if collapse:
        s1=np.load(name+'_sigma.npy')
        s2=np.load(name+'_mass.npy')
        vort*=s1/s2
    N=len(vort)
    dx=L/N
    VORT=load_VORT(name,N,L,R_cut,dx,noise)
    redges=Block_Negative_redges(N,Nbins)
    x=[]
    y=[]
    for j in range(Nbins):
        element=interp_negative_bin(name,redges[j]*dx,redges[j+1]*dx,use_mass)
        x.append(element[0])
        y.append(element[1])
    x=np.array(x)
    y=np.array(y)
    ERR=np.zeros(Nbins)
    FAC=np.zeros(Nbins)
    for j in range(Nbins):
        F1=f1
        F2=f2
        if energy_loss:
            aux=energy_loss_map(name,F1,N)
            n1 = Negative_fraction_radii(vort+aux*VORT,Nbins,redges[j],redges[j+1])-y[j][0]
            aux=energy_loss_map(name,F2,N)
            n2 = Negative_fraction_radii(vort+aux*VORT,Nbins,redges[j],redges[j+1])-y[j][0]
            del aux
        else:
            n1 = Negative_fraction_radii(vort+F1*VORT,Nbins,redges[j],redges[j+1])-y[j][0]
            n2 = Negative_fraction_radii(vort+F2*VORT,Nbins,redges[j],redges[j+1])-y[j][0]
        while n1<0:
            F1*=1.1
            if energy_loss:
                aux=energy_loss_map(name,F1,N)
                n1 = Negative_fraction_radii(vort+aux*VORT,Nbins,redges[j],redges[j+1])-y[j][0]
                del aux
            else:
                n1 = Negative_fraction_radii(vort+F1*VORT,Nbins,redges[j],redges[j+1])-y[j][0]
        while n2>0:
            F2*=0.9
            if energy_loss:
                aux=energy_loss_map(name,F2,N)
                n2 = Negative_fraction_radii(vort+aux*VORT,Nbins,redges[j],redges[j+1])-y[j][0]
                del aux
            else:
                n2 = Negative_fraction_radii(vort+F2*VORT,Nbins,redges[j],redges[j+1])-y[j][0]

        while ((F1-F2)/(F1+F2)>0.005):

            FM=0.5*(F1+F2)
            if energy_loss:
                aux=energy_loss_map(name,FM,N)
                nm = Negative_fraction_radii(vort+aux*VORT,Nbins,redges[j],redges[j+1])-y[j][0]
                del aux
            else:
                nm=Negative_fraction_radii(vort+FM*VORT,Nbins,redges[j],redges[j+1])-y[j][0]
            if n1*nm>0:
                F1=FM
                n1=nm
            else:
                F2=FM
                n2=nm
        FAC[j]=FM
        if energy_loss:
            aux=energy_loss_map(name,FM,N)
            res, rcen, Neg, redges = Block_Negative_radii(vort+aux*VORT,Nbins)
        else:
            res, rcen, Neg, redges = Block_Negative_radii(vort+FM*VORT,Nbins)

        ERR[j]=error_two_curves(x[j]*2,res*dx,y[j],Neg[j,:])

    return FAC,ERR

def optimum_radial_rcut(name,L,R_cut,Nbins,noise,N,dN,f1=50,f2=0,collapse=False,energy_loss=False,use_mass=False):
    Radius=np.zeros(N)
    Factors=np.zeros((N,Nbins))
    Errors=np.zeros((N,Nbins))
    redges = np.linspace(0,0.75,Nbins+1)
    rcen   = 0.25*(redges[:-1]+redges[1:])*L

    for k in range(N):
        R=R_cut+k*dN
        Radius[k]=R
        x,y=radial_rcut(name,L,R,Nbins,noise,f1,f2,collapse,energy_loss,use_mass)
        Factors[k,:]=x
        Errors[k,:]=y

    fx=np.zeros(Nbins)
    rx=np.zeros(Nbins)
    for j in range(Nbins):
        err=Errors[:,j]
        minimo=np.where(err==err.min())[0][0]
        fx[j]=Factors[minimo,j]
        rx[j]=Radius[minimo]

    return rcen,fx,rx

def error_two_curves(x1,x2,y1,y2):
    xmin=max(x1.min(),x2.min())
    xmax=min(x1.max(),x2.max())

    n1=len(x1[(x1>xmin)&(x1<xmax)])
    n2=len(x2[(x2>xmin)&(x2<xmax)])

    if n1>=n2:
        y1=y1[(x1>xmin)&(x1<xmax)]
        x1=x1[(x1>xmin)&(x1<xmax)]

        fun=interp1d(x2,y2)
        yout=fun(x1)
        err=np.sqrt(np.mean((y1-yout)**2))
    else:
        y2=y2[(x2>xmin)&(x2<xmax)]
        x2=x2[(x2>xmin)&(x2<xmax)]

        fun=interp1d(x1,y1)
        yout=fun(x2)
        err=np.sqrt(np.mean((y2-yout)**2))
    return err

def Negative_fraction_radii(mapa,Nbins,rmin,rmax):

    radial=radial_map(mapa)
    radial=radial.ravel()

    radial=radial.ravel()
    mapr=mapa.ravel()
    ring=(rmin<radial)&(rmax>radial)
    temp=mapr[ring]
    try:
        Negative=len(temp[temp<0])/len(temp)
    except:
        Negative=0.0
    return Negative

def Block_Negative_radii(mapa,Nbins):
    N   = len(mapa)
    res = list((10**np.linspace(0,np.log10(N),100)+1e-5))
    res.sort()
    res = np.array(res).astype(int)
    res = np.array(list(set(res)))
    res.sort()

    res=np.array(list(set((N/res).astype(int))))
    res.sort()

    radial=radial_map(mapa)
    radial=radial.ravel()

    if Nbins>1:
        #redges = np.linspace(0,0.5*N,Nbins)
        #redges = np.insert(redges,len(redges),radial.max())
        redges = np.linspace(0,0.5*N*0.75,Nbins+1)
    else:
        #redges=np.array([0,radial.max()])
        redges=np.array([0,0.5*N*0.75])
    rcen   = 0.5*(redges[:-1]+redges[1:])

    Negative=np.zeros((Nbins,len(res)))
    for k,R in enumerate(res):
        fake=block_reduce(mapa, R, func=np.sum)
        N1=len(fake)
        radial=radial_map(fake)
        fake=fake.ravel()
        radial=radial.ravel()*N/N1

        for j in range(Nbins):
            ring=(redges[j]<radial)&(redges[j+1]>radial)
            temp=fake[ring]
            try:
                Negative[j][k]=len(temp[temp<0])/len(temp)
            except:
                Negative[j][k]=0.0
    return res, rcen, Negative, redges

def Block_Negative_redges(N,Nbins):
    if Nbins>1:
        redges = np.linspace(0,0.5*N*0.75,Nbins+1)
    else:
        redges=np.array([0,0.5*N*0.75])
    return redges

def interp_negative_bin(name,rmin,rmax,use_mass):
    f    = open("Temp_Files/"+name+"-All-Radial", "rb")
    res  = pickle.load(f)
    rcen = pickle.load(f)
    Neg  = pickle.load(f)
    Mass  = pickle.load(f)
    f.close()
    if use_mass:
        Neg=Mass
    redges=0.5*(rcen[1:]+rcen[:-1])

    redges=np.insert(redges,0,0)
    redges=np.insert(redges,len(redges),redges[-1]+rcen[2]-rcen[1])

    Rmin=rmin
    Rmax=rmax
    if rmin<redges.min():
        Rmin=redges.min()
    if rmax>redges.max():
        Rmax=redges.max()

    il=np.where(redges<=Rmin)[0][-1]
    iu=np.where(redges>=Rmax)[0][0]-1

    f0=(redges[il+1]-Rmin)/(redges[il+1]-redges[il])
    fn=(redges[iu+1]-Rmax)/(redges[iu+1]-redges[iu])

    if il==iu:
        return res,Neg[il,:]

    pesos=np.ones(iu-il+1)
    pesos[0]=f0
    pesos[-1]=fn

    indices=np.arange(il,iu+1,1)
    indices2=indices-il
    for ij,lm in zip(indices,indices2):
        pesos[lm]*=rcen[ij]
    pesos=pesos/pesos.sum()

    Nj=len(res)
    temp=np.zeros(Nj)
    for ij in range(Nj):
        temp[ij]=(Neg[il:iu+1,ij]*pesos).sum()

    return res, temp

def energy_loss(name,L):
    if os.path.isfile(name+'_energy_loss.npy'):
        EL=np.load(name+'_energy_loss.npy')
        return EL
    sigma=np.load(name+'_sigma.npy')
    mass=np.load(name+'_mass.npy')
    omega=np.load(name+'_omeg.npy')
    vort=np.load(name+'_vort.npy')

    N=len(omega)
    dx=L/N
    A=dx**2

    mass=mass*Msun/pc**2
    sigma=sigma*Msun/pc**2

    vort=vort*pc**2/Myr
    omega=omega*pc**2/Myr
    A=A*pc**2

    v2=0.977813106*omega**2*(1.0- vort*mass/(omega*sigma))/A
    v2.convert_to_units('km**2/s**2')
    v2[v2<np.percentile(v2,0.1)]=np.percentile(v2,1)
    v2[v2>np.percentile(v2,99)]=np.percentile(v2,99)
    noise=np.random.uniform(-1,1,(N,N))*v2

    np.save(name+'_energy_loss',noise)
    return noise

def energy_loss_map(name,factor,N):
    ff=factor

    EL=np.load(name+'_energy_loss.npy')
    ff=np.ones((N,N))*ff
    ff=ff**2-EL
    ff[ff<0]=0
    ff=np.sqrt(ff)
    return ff
