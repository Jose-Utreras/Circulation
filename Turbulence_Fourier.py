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

import time
import emcee
from schwimmbad import MPIPool

def vk_spectrum(k,k0,kcore):
    A=(k0+kcore)**(-4/3)*k0**2
    return np.piecewise(k,[k>=k0,k<k0],[lambda k: A/k**2, lambda k: (k+kcore)**(-4/3)])

def Noise(N,n1,n2,kc):
    dx=1/N
    k1 = fftpack.fftfreq(N,d=dx)

    kx,ky = np.meshgrid(k1, k1)
    KX = fftpack.fftshift( kx )
    KY = fftpack.fftshift( ky )
    K=np.sqrt(KX**2+KY**2)
    K[K<1]=0.5


    F1=fftpack.fft2(np.random.normal(0,1,size=(N,N)))
    F1*=two_slopes(K,n1,n2,kc)
    F1[K<1]=0

    F1=fftpack.fftshift( F1 )

    SS=fftpack.ifft2(F1)
    noise=np.real(SS)
    noise-=np.mean(noise)
    noise/=np.std(noise)

    return noise

def Vorticity(N,n1,n2,kc):
    vx=Noise(N+2,n1,n2,kc)
    vy=Noise(N+2,n1,n2,kc)
    dx=1.0/N
    DVX=(vy[1:-1,2:]-vy[1:-1,:-2])/2
    DVY=(vx[2:,1:-1]-vx[:-2,1:-1])/2
    VORT=-DVY+DVX
    VORT/=dx
    return VORT

def add_vorticity(mapa,n1,n2,kc,factor):
    N=len(mapa)
    new_map=mapa+factor*Vorticity(N,n1,n2,kc)
    return new_map

def add_vorticity_to_map(mapa,L,n1,n2,kc,factor):
    N=len(mapa)
    dA=(L/N)**2
    new_map=mapa+factor*Vorticity(N,n1,n2,kc)*1.02269032*dA/L
    return new_map

def Percentile_profile(mapa,PP):

    N=len(mapa)
    res=list((10**np.linspace(0,np.log10(N),100)+1e-5))
    res.sort()
    res=np.array(res).astype(int)
    res=np.array(list(set(res)))
    res.sort()

    res=np.array(list(set((N/res).astype(int))))
    res.sort()
    Per=np.zeros_like(res,dtype=float)


    for k,R in enumerate(res):
        fake=block_reduce(mapa, R, func=np.sum)
        N1=len(fake)
        fake=fake.ravel()
        try:

            Per[k]=np.percentile(fake,PP)
        except:
            Per[k]=0


    return res, Per

def Percentile_profiles(mapa,PP):

    N=len(mapa)
    res=list((10**np.linspace(0,np.log10(N),100)+1e-5))
    res.sort()
    res=np.array(res).astype(int)
    res=np.array(list(set(res)))
    res.sort()

    res=np.array(list(set((N/res).astype(int))))
    res.sort()

    N1=len(res)
    N2=len(PP)
    Per=np.zeros((N1,N2),dtype=float)


    for k,R in enumerate(res):
        fake=block_reduce(mapa, R, func=np.sum)/R**2
        N1=len(fake)
        fake=fake.ravel()
        for j in range(N2):
            try:
                Per[k][j]=np.percentile(fake,PP[j])
            except:
                Per[k][j]=0


    return res, Per

def two_slopes(k,n1,n2,k0):
    A=k0**(n2-n1)
    return np.piecewise(k,[k<k0,k>=k0],[lambda k: 1.0/k**n1, lambda k: A/k**n2])

def one_slope(k,n1):
    return 1.0/k**n1

def sigma_resolution(mapa):
    N=len(mapa)
    res=list((10**np.linspace(0,np.log10(N),100)+1e-5))
    res.sort()
    res=np.array(res).astype(int)
    res=np.array(list(set(res)))
    res.sort()

    res=np.array(list(set((N/res).astype(int))))
    res.sort()
    Sig=np.zeros_like(res,dtype=float)
    eta=2*0.6744897501

    for k,R in enumerate(res):
        fake=block_reduce(mapa, R, func=np.sum)
        fake=fake.ravel()
        try:
            #Sig[k]=(np.percentile(fake,75)-np.percentile(fake,25))/eta
            Sig[k]=np.std(fake)
        except:
            Sig[k]=0

    return res, Sig

def step_spacing(Number,vmin,vmax,func_obs,func_pro):
    zeta=np.linspace(vmin,vmax,Number)
    yobs=func_obs(zeta)/simpson_array(zeta,func_obs(zeta))
    ypro=func_pro(zeta)/simpson_array(zeta,func_pro(zeta))
    yfft=fft(yobs)/fft(ypro)
    aux=yfft.imag**2+yfft.real**2

    zeta=zeta[~np.isnan(aux)]
    aux=aux[~np.isnan(aux)]

    z1=zeta[aux==aux.max()]
    z2=zeta[(aux-0.5*aux.max())**2==((aux-0.5*aux.max())**2).min()]
    est=np.abs((z2-z1)/np.sqrt(2*np.log(2)))
    try:
        est=est[0]
    except:
        pass
    try:
        popt, pcov = curve_fit(gaussian_step, zeta, aux,p0=[aux.max(),np.mean(zeta),est,0])


        err=np.sqrt(np.mean((gaussian_step(zeta,*popt)-aux)**2))/(aux.max()-aux.min())
        factor=(np.std(aux[1:]-aux[:-1])/np.std(aux))

        print(Number,err,factor,err*factor)
        err*=factor
        #plt.plot(zeta,aux)
        #plt.plot(zeta,gaussian_step(zeta,*popt))
        #plt.show()
        #plt.close()


    except:
        err=1e4

    return err

def best_spacing(vmin,vmax,func_obs,func_pro):

    x1=100
    step=1
    x=[]
    y=[]
    for k in range(200):
        f1=step_spacing(x1,vmin,vmax,func_obs,func_pro)
        x.append(x1)
        y.append(f1)
        x1+=step
    y=np.array(y)
    x=np.array(x)

    x=x[y<np.percentile(y,50)]
    y=y[y<np.percentile(y,50)]

    plt.plot(x,y,'.')
    minimos=np.where(x[1:]-x[:-1]!=step)[0]
    minimos=np.array(minimos)
    minimos=np.insert(minimos,len(minimos),len(x))
    minimos=np.insert(minimos,0,0)

    x=x[minimos[0]:minimos[1]+1]
    y=y[minimos[0]:minimos[1]+1]


    return x[y==y.min()][0]

def get_noise_map(mapa):
    N=len(mapa)
    R=2*radial_map_N(N,N)/N
    h=2*(np.percentile(mapa,75)-np.percentile(mapa,25))/N**(1.0/3.0)
    Nbins=np.percentile(mapa,99.9)-np.percentile(mapa,0.1)
    Nbins=2*int(Nbins/h)
    Nbins=min(Nbins,int(N/3))
    Redges=np.linspace(0,R.max(),Nbins+1)
    Rcen=0.5*(Redges[1:]+Redges[:-1])

    A=np.zeros(Nbins)
    B=np.zeros(Nbins)
    C=np.zeros(Nbins)
    for k in range(Nbins):
        ring=(Redges[k]<=R)&(R<Redges[k+1])
        yaux=mapa[ring].ravel()
        raux=R[ring].ravel()
        yaux=yaux[raux.argsort()]
        raux=raux[raux.argsort()]
        popt, pcov = curve_fit(square_function, raux, yaux)
        A[k]=popt[0]
        B[k]=popt[1]
        C[k]=popt[2]

    xtest=np.linspace(0,1.6,100000)
    ytest=np.zeros_like(xtest)

    for k in range(Nbins+1):
        if k==0:
            x2=Rcen[k]
            kregion=xtest<x2
            a2,b2,c2=A[k],B[k],C[k]
            d2=x2-xtest[kregion]
            f2=a2*xtest[kregion]**2+b2*xtest[kregion]+c2
            ytest[kregion]=f2

        elif k==Nbins:
            x1=Rcen[k-1]
            kregion=x1<xtest
            a1,b1,c1=A[k-1],B[k-1],C[k-1]
            d1=xtest[kregion]-x1
            f1=a1*xtest[kregion]**2+b1*xtest[kregion]+c1
            ytest[kregion]=f1
        else:
            x1=Rcen[k-1]
            x2=Rcen[k]
            kregion=(x1<xtest)&(x2>xtest)
            a1,b1,c1=A[k-1],B[k-1],C[k-1]
            a2,b2,c2=A[k],B[k],C[k]

            d1=xtest[kregion]-x1
            d2=x2-xtest[kregion]
            D=d1+d2
            f1=a1*xtest[kregion]**2+b1*xtest[kregion]+c1
            f2=a2*xtest[kregion]**2+b2*xtest[kregion]+c2
            ytest[kregion]=(f1*d2+f2*d1)/D



    f3=interp1d(xtest,ytest)
    Y3=np.zeros(Nbins)

    for k in range(Nbins):
        ring=(Redges[k]<=R)&(R<Redges[k+1])
        yaux=mapa[ring]
        raux=R[ring]


        Y3[k]=np.std(yaux-f3(raux))

    Rcen=np.insert(Rcen,0,2*Rcen[0]-Rcen[1])
    Rcen=np.insert(Rcen,len(Rcen),np.sqrt(2)*1.1)

    Y3=Y3/np.median(Y3)
    Y3=np.insert(Y3,0,2*Y3[0]-Y3[1])
    Y3=np.insert(Y3,len(Y3),Y3[-1])
    F3=interp1d(Rcen,Y3)


    mapa3=(mapa-f3(R))/F3(R)
    mapa3-=np.mean(mapa3)
    x1=1.0
    x2=1.1
    x3=1.2

    f1=np.std(mapa-x1*mapa3)
    f2=np.std(mapa-x2*mapa3)

    for k in range(10):
        f3=np.std(mapa-x3*mapa3)
        df=(f3-f2)/(x3-x2)
        ddf=2*(df-(f2-f1)/(x2-x1))/(x3-x1)
        step=-df/ddf
        x1=x2
        x2=x3
        x3=x3+step

        f1=f2
        f2=f3

    return mapa3*x3

def percentile_error(average,L,n1,n2,kc,factor,P25,P50,P75):
    new_map=add_vorticity_to_map(average,L,n1,n2,kc,factor)
    res,Per=Percentile_profiles(new_map,[25,50,75])
    Q25=Per[:,0]
    Q50=Per[:,1]
    Q75=Per[:,2]

    e1=np.mean((Q25-P25)**2)
    e2=np.mean((Q50-P50)**2)
    e3=np.mean((Q75-P75)**2)

    return e1+e2+e3

def vorticity_from_circulation(mapa):
    rad,ome=profile_from_circulation(mapa)
    n=np.zeros_like(rad)
    n[1:-1]=(np.log(ome[2:])-np.log(ome[:-2]))/(np.log(rad[2:])-np.log(rad[:-2]))
    try:
        h1=np.log(rad[1])-np.log(rad[0])
        h2=np.log(rad[2])-np.log(rad[0])
        f0=np.log(ome[0])
        f1=np.log(ome[1])
        f2=np.log(ome[2])
        n[0]=(h2**2*(f1-f0)-h1**2*(f2-f0))/(h2*h1*(h2-h1))
    except:
        h2=np.log(rad[2])-np.log(rad[1])
        f1=np.log(ome[1])
        f2=np.log(ome[2])
        n[0]=(f2-f1)/h2
    h1=np.log(rad[-1])-np.log(rad[-2])
    h2=np.log(rad[-1])-np.log(rad[-3])
    f0=np.log(ome[-1])
    f1=np.log(ome[-2])
    f2=np.log(ome[-3])
    n[-1]=-(h2**2*(f1-f0)-h1**2*(f2-f0))/(h2*h1*(h2-h1))

def Negative_profile(mapa):

    N=len(mapa)
    res=list((10**np.linspace(0,np.log10(N),100)+1e-5))
    res.sort()
    res=np.array(res).astype(int)
    res=np.array(list(set(res)))
    res.sort()

    res=np.array(list(set((N/res).astype(int))))
    res.sort()
    Negative=np.zeros_like(res,dtype=float)
    Total=np.zeros_like(res,dtype=float)


    for k,R in enumerate(res):
        fake=block_reduce(mapa, R, func=np.sum)
        N1=len(fake)
        fake=fake.ravel()
        try:
            Negative[k]=len(fake[fake<0])/len(fake)
        except:
            Negative[k]=0
        Total[k]=len(fake)


    return res, Negative,Total

def Profile_sigma_analytic(n1,n2,L,kc,N,M):
    dx=L/N
    k1 = fftpack.fftfreq(N,d=dx)

    kx,ky = np.meshgrid(k1, k1)
    KX = fftpack.fftshift( kx )
    KY = fftpack.fftshift( ky )
    K=np.sqrt(KX**2+KY**2)
    Wv=two_slopes(K,n1,n2,kc)
    Wv[K<2/L]=0

    w=3*np.ones((N,N))
    w[::3,0]-=1
    w[0,0]=1
    w[-1,0]=1

    w[0,::3]-=1
    w[0,0]=1
    w[0,-1]=1

    for j in range(N-1):
        w[:,j]=w[:,0]*w[0,j]
    h=KX[0,1]-KX[0,0]

    factor=Wv/((KX+1e-16)*(KY+1e-16))
    n=1
    delta=L/N
    W1=factor*np.sin(n*KX*delta)*np.sin(n*KY*delta)/delta**2
    S1=np.sqrt(np.sum(w*W1**2*9*h**2/64))

    deviations=np.zeros(M)
    deltas=np.exp(np.linspace(0,np.log(N),M))*L
    for k,delta in enumerate(deltas/N):
        W2=factor*np.sin(n*KX*delta)*np.sin(n*KY*delta)/delta**2
        S2=np.sqrt(np.sum(w*W2**2*9*h**2/64))
        deviations[k]=S2/S1

    return deltas,deviations

def mcmc_std(mapa,mpi=False):
    def lnlike(theta, x0, y0, N):
        n1,n2,kc = theta
        x1,y1=sigma_resolution(Noise(N,n1,n2,kc))
        y1=y1/x1**2
        y1+=np.mean(y0)-np.mean(y1)
        error=np.sum((y1-y0)**2)
        plt.plot(x0,y0)
        plt.plot(x1,y1)
        plt.title(str(error))
        plt.xscale('log')
        plt.savefig('Temp_Figures/n1_%04.2f' %n1+'_n2_%04.2f' %n2+'_kc_%04d' %int(kc)+'.png')
        plt.close()
        return -0.5*error

    def lnprior(theta, N):
        n1, n2, kc = theta
        if 0 < n1 < 5 and n1 < n2 < 9.0 and 1 < kc < N:
            return 0.0
        return -1e32

    def lnprob(theta, x0, y0, N):
        lp = lnprior(theta, N)
        if not np.isfinite(lp):
            return -1e32
        return lp + lnlike(theta, x0,y0,N)

    new_map=mapa/np.std(mapa)
    x0,y0=sigma_resolution(new_map)
    y0=y0/x0**2
    N=len(new_map)
    if mpi:
        with MPIPool() as pool:
            if not pool.is_master():
                pool.wait()
                sys.exit(0)

            ndim, nwalkers = 3, 10
            pos = [[1.33,4,N/4] + np.random.randn(ndim) for i in range(nwalkers)]
            nsteps = 10

            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x0,y0,N),pool=pool)
            start = time.time()
            sampler.run_mcmc(pos, nsteps)
            end = time.time()
            return sampler
    else:
        ndim, nwalkers = 3, 20
        pos = [[1.33,4,N] + np.random.randn(ndim) for i in range(nwalkers)]
        for p in pos:
            p[2]*=np.random.uniform(0,1)
        print(pos)
        nsteps = 100

        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x0,y0,N))
        start = time.time()
        sampler.run_mcmc(pos, nsteps)
        end = time.time()
        return sampler

def gamma_p(p,n1,n2,pc,pmin,pmax):
    B=1
    A=B*pc**(n2-n1)
    return np.piecewise(p,[p<pmin,(pmin<p)&(pc<pc),(p>=pc)&(p<pmax),p>pmax],
        [lambda p:0, lambda p: B/p**n1, lambda p: A/p**n2, lambda p : 0])

def gamma_delta(p,q,n1,n2,delta,pc,pmin,pmax):
    rp=np.sqrt(p**2+q**2)
    return np.sin(np.pi*p*delta)*np.sin(np.pi*q*delta)*gamma_p(rp,n1,n2,pc,pmin,pmax)/((p+1e-16)*(q+1e-16)*np.pi**2*delta**2)
