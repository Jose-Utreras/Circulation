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

def sample(i,j,R2,xmax,ymax):
    R=R2+0.5
    r=R*(np.random.uniform(0,1)**0.5)
    theta=np.random.uniform(0,2*np.pi)
    I=i+int(r*np.cos(theta))
    J=j+int(r*np.sin(theta))
    I=max(0,min(xmax-1,I))
    J=max(0,min(ymax-1,J))
    return I,J

def spatial_noise(image,R):
    xmax,ymax=np.shape(image)
    new_image=np.zeros_like(image)

    for i in range(xmax):
        for j in range(ymax):

            I,J=sample(i,j,R,xmax,ymax)
            new_image[i,j]=image[I,J]
    return new_image

def spatial_noise_radial(image,R2):
    Radius=R2[0]
    Size=R2[1]
    if Radius[0]>=1e-4:
        Radius=np.insert(Radius,0,0)
        Radius=np.insert(Radius,len(Radius),2*Radius[-1])
        Size=np.insert(Size,0,Size[0])
        Size=np.insert(Size,len(Size),Size[-1])

    frad=interp1d(Radius,Size)
    xmax,ymax=np.shape(image)
    new_image=np.zeros_like(image)
    print(Radius)
    print(Size)
    for i in range(xmax):
        for j in range(ymax):
            r_ij=np.sqrt((i-0.5*xmax)**2+(j-0.5*ymax)**2)
            R3=frad(r_ij)
            I,J=sample(i,j,R3,xmax,ymax)
            new_image[i,j]=image[I,J]
    return new_image

def example(N,n,S,direc):

    bool1=os.path.isfile(direc+'vx_N%04d'%N+'_n%.1f'%n+'.npy')
    bool2=os.path.isfile(direc+'vy_N%04d'%N+'_n%.1f'%n+'.npy')
    bool3=os.path.isfile(direc+'vort_N%04d'%N+'_n%.1f'%n+'.npy')
    if bool1&bool2&bool3:
        Vx=np.load(direc+'vx_N%04d'%N+'_n%.1f'%n+'.npy')
        Vy=np.load(direc+'vy_N%04d'%N+'_n%.1f'%n+'.npy')
    else:
        Omega=np.ones((N,N))

        X=np.array(list(np.reshape(range(N),(1,N)))*N)

        Y=X.T
        X=np.reshape(X,(N,N))-0.5*N
        Y=np.reshape(Y,(N,N))-0.5*N
        R=np.sqrt(X**2+Y**2)

        Omega/=(R+0.2)**n
        Omega/=((2-n)*Omega.sum())
        Vort=(2-n)*Omega
        Vx=-Omega*Y
        Vy=Omega*X
        np.save(direc+'vx_N%04d'%N+'_n%.1f'%n,Vx)
        np.save(direc+'vy_N%04d'%N+'_n%.1f'%n,Vy)
        np.save(direc+'vort_N%04d'%N+'_n%.1f'%n,Vort[1:-1,1:-1])
    Vx=spatial_noise(Vx,S)
    Vy=spatial_noise(Vy,S)
    DVX=(Vy[1:-1,2:]-Vy[1:-1,:-2])/2
    DVY=(Vx[2:,1:-1]-Vx[:-2,1:-1])/2

    Vort2=-DVY+DVX
    return Vort2

def save_example(N,n,S_noise,S_diff,directory):
    if directory[-1]=='/':
        pass
    else:
        directory+='/'
    if os.path.isfile(directory+'noise_N%04d'%N+'_n%.1f'%n+'_R%03d'%S_noise+'.npy'):
        print('noise file exists',N,n,S_noise)
        Vort2=np.load(directory+'noise_N%04d'%N+'_n%.1f'%n+'_R%03d'%S_noise+'.npy')
    else:
        Vort2=example(N,n,S_noise,directory)
        np.save(directory+'noise_N%04d'%N+'_n%.1f'%n+'_R%03d'%S_noise,Vort2)
    if S_diff=='None':
        pass
    else:
        for sd in S_diff:
            if os.path.isfile(directory+'difussion_N%04d'%N+'_n%.1f'%n+'_R%03d'%S_noise+'_D'+temp+'.npy'):
                pass
            else:
                Vort=difussion(Vort2,sd,mode='wrap')
                temp='%02d'%sd
                np.save(directory+'difussion_N%04d'%N+'_n%.1f'%n+'_R%03d'%S_noise+'_D'+temp,Vort)
                del Vort
    return True

def fit_examples():
    lista=glob.glob('Examples/difussion*')
    lista.sort()
    NN=len(lista)
    n=np.zeros(NN)
    R=np.zeros(NN)
    D=np.zeros(NN)
    A=np.zeros(NN)
    mu=np.zeros(NN)
    sig=np.zeros(NN)

    for k,li in enumerate(lista):
        if k%10==0:
            print(k,NN)
        name=li
        name=name.split('_')
        n[k]=float(name[-3][1:])
        R[k]=int(name[-2][1:])
        D[k]=int(name[-1][1:3])
        mapa=np.load(li)
        res,Neg=Circulation_Negative(mapa)
        x=res[Neg>0]
        y=Neg[Neg>0]
        try:
            f1=interp1d(y,x)
            x50=f1(y[0]/2)
            s=(x50-1)/(np.sqrt(2*np.log(2)))
            popt, pcov = curve_fit(gaussian, x , y, p0=[y[0],x[0],s],bounds=(-np.inf,[np.inf,1,np.inf]))

            A[k]=popt[0]
            mu[k]=popt[1]
            sig[k]=popt[2]
            if A[k]>0.5:
                plt.plot(x,y,'b')
                plt.plot(x,gaussian(x,*popt),'r:')
                plt.show()
                plt.close()
        except:
            A[k]=0
            mu[k]=0
            sig[k]=0
    tabla=Table()
    tabla['n']=Column(n)
    tabla['R']=Column(R)
    tabla['D']=Column(D)
    tabla['A']=Column(A)
    tabla['mu']=Column(mu)
    tabla['sig']=Column(sig)
    tabla.write('Examples/Table_Parameters',path='data',format='hdf5',overwrite=True)

def grid_table(N,narr,snarr,sdarr,directory):
    if directory[-1]=='/':
        pass
    else:
        directory+='/'

    NN=len(narr)*len(snarr)*len(sdarr)
    ii=-1
    n=np.zeros(NN)
    R=np.zeros(NN)
    D=np.zeros(NN)
    A=np.zeros(NN)
    mu=np.zeros(NN)
    sig=np.zeros(NN)

    for nd in narr:
        for S_noise in snarr:
            if os.path.isfile(directory+'noise_N%04d'%N+'_n%.1f'%nd+'_R%03d'%S_noise+'.npy'):
                Vort=np.load(directory+'noise_N%04d'%N+'_n%.1f'%nd+'_R%03d'%S_noise+'.npy')
            else:
                Vort=example(N,nd,S_noise,directory)
            for sd in sdarr:
                ii+=1

                n[ii]=nd
                R[ii]=S_noise
                D[ii]=sd
                temp='%02d'%sd
                if os.path.isfile(directory+'difussion_N%04d'%N+'_n%.1f'%nd+'_R%03d'%S_noise+'_D'+temp+'.npy'):
                    Vort2=np.load(directory+'difussion_N%04d'%N+'_n%.1f'%nd+'_R%03d'%S_noise+'_D'+temp+'.npy')
                else:
                    Vort2=difussion(Vort,sd,mode='wrap')
                res,Neg=Circulation_Negative(Vort2)
                x=res[Neg>0]
                y=Neg[Neg>0]
                try:
                    f1=interp1d(y,x)
                    x50=f1(y[0]/2)
                    s=(x50-1)/(np.sqrt(2*np.log(2)))
                    popt, pcov = curve_fit(gaussian, x , y, p0=[y[0],x[0],s],bounds=(-np.inf,[np.inf,1,np.inf]))

                    A[ii]=popt[0]
                    mu[ii]=popt[1]
                    sig[ii]=popt[2]
                    del popt,pcov,s,x50,f1
                except:
                    A[ii]=0
                    mu[ii]=0
                    sig[ii]=0
                del res,Neg,x,y
                print(ii*100.0/NN)

    tabla=Table()
    tabla['n']=Column(n)
    tabla['R']=Column(R)
    tabla['D']=Column(D)
    tabla['A']=Column(A)
    tabla['mu']=Column(mu)
    tabla['sig']=Column(sig)
    tabla.write('Examples/Table_Parameters_Grid',path='data',format='hdf5',overwrite=True)

def grid_table_n(N,nd,snarr,sdarr,directory):
    if directory[-1]=='/':
        pass
    else:
        directory+='/'

    NN=len(snarr)*len(sdarr)
    ii=-1
    R=np.zeros(NN)
    D=np.zeros(NN)
    A=np.zeros(NN)
    mu=np.zeros(NN)
    sig=np.zeros(NN)

    for S_noise in snarr:
        if os.path.isfile(directory+'noise_N%04d'%N+'_n%.1f'%nd+'_R%03d'%S_noise+'.npy'):
            Vort=np.load(directory+'noise_N%04d'%N+'_n%.1f'%nd+'_R%03d'%S_noise+'.npy')
        else:
            Vort=example(N,nd,S_noise,directory)
        for sd in sdarr:
            ii+=1
            R[ii]=S_noise
            D[ii]=sd
            temp='%02d'%sd
            if os.path.isfile(directory+'difussion_N%04d'%N+'_n%.1f'%nd+'_R%03d'%S_noise+'_D'+temp+'.npy'):
                Vort2=np.load(directory+'difussion_N%04d'%N+'_n%.1f'%nd+'_R%03d'%S_noise+'_D'+temp+'.npy')
            else:
                Vort2=difussion(Vort,sd,mode='wrap')
            res,Neg=Circulation_Negative(Vort2)
            x=res[Neg>0]
            y=Neg[Neg>0]
            try:
                f1=interp1d(y,x)
                x50=f1(y[0]/2)
                s=(x50-1)/(np.sqrt(2*np.log(2)))
                popt, pcov = curve_fit(gaussian, x , y, p0=[y[0],x[0],s],bounds=(-np.inf,[np.inf,1,np.inf]))
                A[ii]=popt[0]
                mu[ii]=popt[1]
                sig[ii]=popt[2]
                del popt,pcov,s,x50,f1
            except:
                A[ii]=0
                mu[ii]=0
                sig[ii]=0
            del res,Neg,x,y
            print(nd,ii*100.0/NN)

    tabla=Table()
    tabla['R']=Column(R)
    tabla['D']=Column(D)
    tabla['A']=Column(A)
    tabla['mu']=Column(mu)
    tabla['sig']=Column(sig)
    tabla.write('Examples/Table_Parameters_Grid_n%.1f'%nd,path='data',format='hdf5',overwrite=True)
    return 'done'

def merge_grid_tables():
    tablas=glob.glob('Examples/Table_Parameters_Grid_n*')

    tablas.sort()

    tabla_0=Table()

    for tabname in tablas:
        tabla=Table.read(tabname,path='data')

        NN=len(tabla)
        n=float(tabname.split('n')[-1])*np.ones(NN)

        tabla['n']=Column(n)
        tabla_0=vstack([tabla_0,tabla])
    tabla_0.write('Examples/Table_Parameters_Grid_Merged',path='data',format='hdf5',overwrite=True)

def prediction_negative(Ao,muo,sigo):
    tabla=Table.read('Examples/Table_Parameters_Grid_Merged',path='data')
    R=tabla['R']
    D=tabla['D']
    n=tabla['n']

    A=tabla['A']
    mu=tabla['mu']
    sig=tabla['sig']


    wA=np.percentile(A,84)-np.percentile(A,16)
    wm=np.percentile(mu,84)-np.percentile(mu,16)
    ws=np.percentile(sig,84)-np.percentile(sig,16)

    A/=wA
    mu/=wm
    sig/=ws

    Ao/=wA
    muo/=wm
    sigo/=ws

    Po=[Ao,muo,sigo]

    P=[]
    Q=[]
    for i,j,k in zip(A,mu,sig):
        P.append([i,j,k])
    for i,j,k in zip(n,R,D):
        Q.append([i,j,k])
    P=np.array(P)
    Q=np.array(Q)
    Distances=np.zeros_like(A)
    for k,p in enumerate(P):
        Distances[k]=distance(p,Po)

    Quadrants=np.array(Heaviside(mu-muo)+Heaviside(A-Ao)*2+Heaviside(sig-sigo)*4,dtype=int)

    nq=len(set(Quadrants))
    cube_n=np.zeros(nq)
    cube_R=np.zeros(nq)
    cube_D=np.zeros(nq)
    cube_l=np.zeros(nq)

    for j,k in enumerate(set(Quadrants)):
        qua=Quadrants==k
        dmin=Distances[qua].min()
        ver=Distances[qua]==dmin
        cube_n[j]=n[qua][ver]
        cube_R[j]=R[qua][ver]
        cube_D[j]=D[qua][ver]
        cube_l[j]=1.0/dmin**3

    n_predict=(cube_n*cube_l).sum()/cube_l.sum()
    R_predict=(cube_R*cube_l).sum()/cube_l.sum()
    D_predict=(cube_D*cube_l).sum()/cube_l.sum()

    return n_predict,R_predict,D_predict

def pnn(Ao,muo,sigo,n):
    tabla=Table.read('Examples/Table_Parameters_Grid_n%.1f' %n,path='data')

    R=tabla['R']
    D=tabla['D']

    A=tabla['A']
    mu=tabla['mu']
    sig=tabla['sig']


    wA=np.percentile(A,84)-np.percentile(A,16)
    wm=np.percentile(mu,84)-np.percentile(mu,16)
    ws=np.percentile(sig,84)-np.percentile(sig,16)

    A/=wA
    mu/=wm
    sig/=ws

    Ao/=wA
    muo/=wm
    sigo/=ws

    Po=[Ao,muo,sigo]

    P=[]
    Q=[]
    for i,j,k in zip(A,mu,sig):
        P.append([i,j,k])
    for i,j in zip(R,D):
        Q.append([i,j])
    P=np.array(P)
    Q=np.array(Q)
    Distances=np.zeros_like(A)

    for k,p in enumerate(P):
        Distances[k]=distance(p,Po)

    Quadrants=np.array(Heaviside(mu-muo)+Heaviside(A-Ao)*2+Heaviside(sig-sigo)*4,dtype=int)

    nq=len(set(Quadrants))
    cube_R=np.zeros(nq)
    cube_D=np.zeros(nq)
    cube_l=np.zeros(nq)

    for j,k in enumerate(set(Quadrants)):
        qua=Quadrants==k
        dmin=Distances[qua].min()
        ver=Distances[qua]==dmin
        cube_R[j]=R[qua][ver]
        cube_D[j]=D[qua][ver]
        cube_l[j]=1.0/dmin

    R_predict=(cube_R*cube_l).sum()/cube_l.sum()
    D_predict=(cube_D*cube_l).sum()/cube_l.sum()

    return R_predict,D_predict

def prediction_negative_n(Ao,muo,sigo,n):
    n1=int(n*10)/10
    n2=n1+np.sign(n-n1)/10
    n1=np.abs(n1)
    n2=np.abs(n2)
    n=np.abs(n)
    w1=n2-n
    w2=n-n1

    w1=w1/(w1+w2)
    w2=1-w1

    r1,d1=pnn(Ao,muo,sigo,n1)
    r2,d2=pnn(Ao,muo,sigo,n2)

    return r1*w1+r2*w2,d1*w1+d2*w2

def save_slope(name):
    vort=np.load(name+'_omeg.npy')
    vd1=np.diag(vort)
    N=len(vd1)
    x=np.array(range(len(vd1[int(N/2):])))
    x=np.sqrt(2)*x*2e4/x.max()
    y=vd1[int(N/2):]
    y=y[x<1.5e4]
    x=x[x<1.5e4]
    y=y[x>0]
    x=x[x>0]
    p=np.polyfit(np.log(x),np.log(y),4)
    x=np.sqrt(2)*x*2e4/x.max()
    x=x[x>0]

    n=np.polyval([p[0]*4,p[1]*3,p[2]*2,p[3]],np.log(x))

    x=np.insert(x,0,0)
    n=np.insert(n,0,0)
    x=np.insert(x,len(x),1e5)
    n=np.insert(n,len(n),n[-1])
    np.savetxt(name+'_slope.txt', (x,n))
    return True

def get_slope(name,rmin,rmax):
    X,N=np.loadtxt(name+'_slope.txt' ,ndmin=2, unpack=True).T
    fN=interp1d(X,N)
    rmin=100
    rmax=200
    radius=np.linspace(rmin,rmax,100)
    Nest=fN(radius)

    NF=np.average(Nest,weights=radius)

    return NF

def get_velocities(name):
    if os.path.isfile(name+'_vx.npy')&os.path.isfile(name+'_vy.npy'):
        vx=np.load(name+'_vx.npy')
        vy=np.load(name+'_vy.npy')

    else:
        Model=np.load(name+'_omeg.npy')
        Data=np.load(name+'_vort.npy')
        xmax,ymax=np.shape(Model)

        X=np.array(list(np.reshape(range(xmax),(1,xmax)))*xmax)
        Y=X.T

        X=np.reshape(X,(xmax,xmax))-np.mean(X)
        Y=np.reshape(Y,(xmax,xmax))-np.mean(Y)
        R=np.sqrt(X**2+Y**2)

        TTest=np.diag(Model)
        TRadius=np.diag(R)

        N=len(TRadius)
        if N%2==0:
            TRadius=np.insert(TRadius,0,0)
            TTest=np.insert(TTest,0,2*TTest[0]-TTest[1])
        TRadius=TRadius[int((N-1)/2):]
        TTest=TTest[int((N-1)/2):]

        TOmega=np.zeros_like(TTest)

        for i in range(len(TTest)):
            TOmega[i]=simps(y=TTest[:i+1]*TRadius[:i+1],dx=np.sqrt(2.0))

        TOmega[0]=0
        TOmega+=TTest[0]/2
        TOmega/=TRadius**2
        TOmega[0]=2*TOmega[1]-TOmega[2]

        f1=interp1d(TRadius,TOmega)
        Omega=f1(R)
        vx=-Omega*Y
        vy=Omega*X
        np.save(name+'_vx',vx)
        np.save(name+'_vy',vy)

    return vx,vy

def difussion_image(name,R):

    vx,vy=get_velocities(name)

    switch=1
    try:
        len(R)
    except:
        switch=0

    if switch==0:
        VX=spatial_noise(vx,R)
        VY=spatial_noise(vy,R)
    else:
        VX=spatial_noise_radial(vx,R)
        VY=spatial_noise_radial(vy,R)

    DVX=(VY[1:-1,2:]-VY[1:-1,:-2])/2
    DVY=(VX[2:,1:-1]-VX[:-2,1:-1])/2

    VORT=-DVY+DVX

    return VORT

def dispersion_image(name,vel):

    vx,vy=get_velocities(name)

    N=len(vx)


    conv=0.977813106
    dx=4.0e4/N
    vturb=vel*dx/conv

    noise=np.random.normal(0,vturb,size=(N,N))
    #noise=resize(noise,(N,N),order=0)

    VX=vx+noise
    VY=vy+noise

    DVX=(VY[1:-1,2:]-VY[1:-1,:-2])/2
    DVY=(VX[2:,1:-1]-VX[:-2,1:-1])/2

    VORT=-DVY+DVX

    return VORT

def optimum_R(name):
    if os.path.isfile(name+'_optimum_R'):
        print('File already exists')
        return 0
    c1=1e10
    c2=1e9
    k=0.5

    Data=np.load(name+'_vort.npy')
    xmax,ymax=np.shape(Data)
    Data=Data[1:-1,1:-1]
    h1,b1=np.histogram(Data.ravel(),100,range=(np.percentile(Data.ravel(),0.5),np.percentile(Data.ravel(),99.5)))
    while c1>c2:
        k+=0.5
        c1=c2
        VORT=difussion_image(name,k)
        h2,b2=np.histogram(VORT.ravel(),100,range=(np.percentile(Data.ravel(),0.5),np.percentile(Data.ravel(),99.5)))
        c2=((h1-h2)**2).sum()/1e10
    rreal=k-0.5
    c1=1e10
    c2=1e9
    k=0.5
    while c1>c2:
        k+=0.5
        c1=c2
        VORT=difussion_image(name,k)
        h2,b2=np.histogram(VORT.ravel(),100,range=(np.percentile(Data.ravel(),0.5),np.percentile(Data.ravel(),99.5)))
        h2[h2==0]=1
        c2=((np.log(h1)-np.log(h2))**2).sum()
    rlog=k-0.5
    return [rreal,rlog,40000/xmax]

def optimum_radii(name,Nbins):
    if os.path.isfile(name+'-%04d'%Nbins):
        print('File already exists')
        return 0
    N=Nbins
    k=0.5
    counter=0
    Data=np.load(name+'_vort.npy')
    Data=Data[1:-1,1:-1]
    xmax,ymax=np.shape(Data)

    X=np.array(list(np.reshape(range(xmax),(1,xmax)))*xmax)

    Y=X.T
    X=np.reshape(X,(xmax,xmax))-np.mean(X)
    Y=np.reshape(Y,(xmax,xmax))-np.mean(Y)
    R=np.sqrt(X**2+Y**2)
    Data=Data.ravel()
    R=R.ravel()
    rmin=R.min()
    rmax=R.max()/np.sqrt(2)
    Redges=np.linspace(rmin,rmax,N+1)
    Rcen=0.5*(Redges[1:]+Redges[:-1])
    C1=1e10*np.ones(N)
    C2=1e9*np.ones(N)
    B1=1e10*np.ones(N)
    B2=1e9*np.ones(N)
    K1array=np.zeros(N)
    K2array=np.zeros(N)

    while counter < 2*N:
        k+=0.2
        VORT=difussion_image(name,k)
        VORT=VORT.ravel()
        print(counter)
        for i in range(N):
            region=((R>Redges[i])&(R<Redges[i+1]))
            C1[i]=C2[i]
            lim_inf=np.percentile(Data[region],0.5)
            lim_sup=np.percentile(Data[region],99.5)
            h1,b1=np.histogram(Data[region],50,range=(lim_inf,lim_sup))
            h2,b2=np.histogram(VORT[region],50,range=(lim_inf,lim_sup))
            C2[i]=((h1-h2)**2).sum()/1e10

            if (C2[i]>C1[i])&(K1array[i]==0):
                K1array[i]=k-0.2
                counter+=1
            B1[i]=B2[i]
            h2[h2==0]=1
            h1[h1==0]=1
            B2[i]=((np.log(h1)-np.log(h2))**2).sum()
            if (B2[i]>B1[i])&(K2array[i]==0):
                K2array[i]=k-0.2
                counter+=1

    return [K1array,K2array,Rcen,40000/(xmax+2)]

def apply_radial_noise(name,Nbins):
    tab=Table.read(name+'-%04d'%Nbins,path='data')
    Radius=tab['Radius']
    R_lin=tab['R_lin']
    R_log=tab['R_log']

    Radius=np.insert(Radius,0,0)
    R_lin=np.insert(R_lin,0,R_lin[0])
    R_log=np.insert(R_log,0,R_log[0])

    Radius=np.insert(Radius,len(Radius),Radius[-1]*1.5)
    R_lin=np.insert(R_lin,len(R_lin),R_lin[-1])
    R_log=np.insert(R_log,len(R_log),R_log[-1])


    Data=np.load(name+'_vort.npy')
    xmax,ymax=np.shape(Data)
    del Data

    R_lin*=xmax*1.0/40000.0
    R_log*=xmax*1.0/40000.0


    V_lin=difussion_image(name,[Radius,R_lin])
    np.save(name+'_lin',V_lin)
    del V_lin
    print('done linear')
    V_log=difussion_image(name,[Radius,R_log])
    np.save(name+'_lin',V_log)
    del V_log
    print('done log')
    return 'done'

def optimum_turb_diff(name,log=True):
    if os.path.isfile(name+'_optimum_turb_diff'):
        print('File already exists')
        return 0

    Data=np.load(name+'_vort.npy')
    Data=Data[1:-1,1:-1]
    D05=np.percentile(Data.ravel(),0.5)
    D95=np.percentile(Data.ravel(),99.5)
    h1,b1=np.histogram(Data.ravel(),50,range=(D05,D95))
    del Data
    if log:
        h1[h1==0]=1
    k=0.5
    c1=1e10
    c2=1e9
    CMIN=1e10
    while c1>c2:
        c1=c2
        VORT=difussion_image(name,k)
        kd=0.025
        sup=1e10
        for kk in range(20):

            VORT2=difussion(VORT,kd,mode='wrap')
            h2,b2=np.histogram(VORT2.ravel(),50,range=(D05,D95))
            if log:
                h2[h2==0]=1
                c3=((np.log(h1)-np.log(h2))**2).sum()
            else:
                c3=((h1-h2)**2).sum()/1e10

            if c3<sup:
                sup=c3
                KD=kd
            kd+=0.025
            del VORT2
        c2=sup

        if c2<CMIN:
            CMIN=c2
            KTM=k
            KDM=KD
        k+=0.2
    return KTM,KDM

def apply_turb_diff(name,KTM,KDM):
    VORT=difussion_image(name,KTM)
    VORT=difussion(VORT,KDM,mode='wrap')
    return VORT

def optimum_turb_diff_radii(name,Nbins,log=True):
    if os.path.isfile(name+'turb-diff-%04d'%Nbins):
        print('File already exists')
        return 0
    N=Nbins
    k=0.5
    counter=0
    Data=np.load(name+'_vort.npy')
    Data=Data[1:-1,1:-1]
    xmax,ymax=np.shape(Data)

    X=np.array(list(np.reshape(range(xmax),(1,xmax)))*xmax)

    Y=X.T
    X=np.reshape(X,(xmax,xmax))-np.mean(X)
    Y=np.reshape(Y,(xmax,xmax))-np.mean(Y)
    R=np.sqrt(X**2+Y**2)
    Data=Data.ravel()
    R=R.ravel()
    rmin=R.min()
    rmax=R.max()/np.sqrt(2)
    Redges=np.linspace(rmin,rmax,N+1)
    Rcen=0.5*(Redges[1:]+Redges[:-1])
    B1=1e10*np.ones(N)
    B2=1e9*np.ones(N)
    KTM=np.zeros(N)
    KDM=np.zeros(N)
    CMIN=1e10*np.ones(N)
    while counter < N:

        VORT=difussion_image(name,k)


        for i in range(N):
            region=((R>Redges[i])&(R<Redges[i+1]))
            lim_inf=np.percentile(Data[region],0.5)
            lim_sup=np.percentile(Data[region],99.5)
            h1,b1=np.histogram(Data[region],50,range=(lim_inf,lim_sup))
            if log:
                h1[h1==0]=1

            B1[i]=B2[i]
            kd=0.025
            sup=1e10
            for kk in range(100):
                VORT2=difussion(VORT,kd,mode='wrap')
                h2,b2=np.histogram(VORT2.ravel(),50,range=(lim_inf,lim_sup))
                if log:
                    h2[h2==0]=1
                    c3=((np.log(h1)-np.log(h2))**2).sum()
                else:
                    c3=((h1-h2)**2).sum()

                if c3<sup:
                    sup=c3
                    KD=kd
                kd+=0.025
                del VORT2

            B2[i]=sup
            if B2[i]<CMIN[i]:
                CMIN[i]=B2[i]
                KTM[i]=k
                KDM[i]=KD
            del sup
            k+=0.2

            if (B2[i]>B1[i]):
                counter+=1

    return [KTM,KDM,Rcen,40000/(xmax+2)]

def Circulation_block_example(name):
    print(name)
    temp=name.split('/')[-1]
    temp=temp.split('.npy')[0]
    mapa=np.load(name)
    N=len(mapa)
    res=list((10**np.linspace(0,np.log10(N),100)+1e-5))
    res.sort()
    res=np.array(res).astype(int)
    res=np.array(list(set(res)))
    res.sort()

    res=np.array(list(set((N/res).astype(int))))
    res.sort()
    for R in [res[0]]:
        if os.path.isfile('Difussion/Block-'+temp+'-%05d'%R):

            print('File already exists')  #Skip code if file already exists

        else:
            original=np.load(name)
            fake=block_reduce(original, R, func=np.sum)
            del original
            Nf=len(fake)
            tabla=Table()
            tabla['circulation'] = Column(np.array(fake.ravel()),  description='vorticity')
            tabla.write('Difussion/Block-'+temp+'-%05d' %R,path='data',format='hdf5')
            del tabla
    return True

def maximum_width():
    original=glob.glob('Examples/*vort*')
    original.sort()
    slopes=[]
    for ori in original:
        num=ori.split('n')[-2]
        num=num[:-1]
        slopes.append(num)
        del num

    for exponent in slopes:
        #if os.path.isfile('Examples/Widths_n'+exponent):
        #    pass
        #else:
        if True:
            original=glob.glob('Examples/*vort*n'+exponent+'*')
            original=original[0]

            vort=np.load(original)
            vort=vort.ravel()
            Width0=np.percentile(vort,84)-np.percentile(vort,16)
            Width0=np.array(Width0)
            Med0=np.percentile(vort,50)
            print(exponent)
            print(Width0)
            print(Med0)

            numbers=np.arange(4,130,4)

            X=[]
            Y=[]
            Z=[]
            W=[]
            for number in numbers:
                print(number)
                original=glob.glob('Examples/*noise*n'+exponent+'*R%03d*'%number)
                original=original[0]

                vert=original.split('_')[-1]
                vert=vert.split('.')[0]
                vert=float(vert[1:])

                vort=np.load(original)
                Width=np.percentile(vort,84)-np.percentile(vort,16)
                WidthW=Width-Width0
                WidthM=Width/Med0
                X.append(vert)
                Y.append(Width)
                Z.append(WidthW)
                W.append(WidthM)
            X=np.array(X)
            Y=np.array(Y)
            Z=np.array(Z)
            W=np.array(W)

            tabla=Table()
            tabla['radius_turb']    = Column(X)
            tabla['Width']          = Column(Y)
            tabla['WidthW']         = Column(Z)
            tabla['WidthM']         = Column(W)

            tabla.write('Examples/Widths_n'+exponent,path='data',format='hdf5',overwrite=True)

        #Width0[Width0<1e-09]=1e9
        #Width0[Width0>10]=Width0.min()

def difussion_width(exponent):

    original=glob.glob('Examples/*vort*n%03.1f*'%exponent)
    original=original[0]
    vort=np.load(original)
    vort=vort.ravel()
    Width0=np.percentile(vort,84)-np.percentile(vort,16)


    noises=['R008','R016','R032','R064']
    numbers=np.array([0.2,0.3,0.4,0.5,0.6,0.8,1,2,3,4,5,6,8,10,12,16])

    tabla=Table()
    tabla['r_dif']=Column(numbers)
    for noise in noises:
        original=glob.glob('Examples/*noise*n%03.1f*'%exponent+noise+'*')
        original=original[0]
        vort=np.load(original)
        vort=vort.ravel()
        Width1=np.percentile(vort,84)-np.percentile(vort,16)
        Width1=Width1-Width0

        Y=np.zeros_like(numbers)
        counter=-1
        for number in numbers:
            counter+=1
            original=glob.glob('Examples/*difussion*n%03.1f*'%exponent+noise+'*D%06.2f*'%number)
            original=original[0]

            vort=np.load(original)
            vort=vort.ravel()
            Width=np.percentile(vort,84)-np.percentile(vort,16)
            Width=(Width-Width0)/Width1
            Y[counter]=Width
        tabla[noise]=Column(Y)
    tabla.write('Examples/Widths_diff_n%03.1f'%exponent,path='data',format='hdf5',overwrite=True)

def f_base(exponent):
    base=10**exponent
    base=base*(4**(-exponent/2.0)-21**(-exponent/2.0))
    base=base/(2**(1.5*exponent))
    return base

def f_width(x,c,d):
    a=1.0/(1+1/c)**d
    return a*(1+x/c)**d

def circulation_width_table(exponent):
    Parameters=[]
    base=f_base(exponent)
    original=glob.glob('Difussion/*vort*n%03.1f*'%exponent)
    original.sort()
    Size0=[]
    Width0=[]
    for original_i in original:
        size=float(original_i.split('-')[-1])
        tabla=Table.read(original_i,path='data')
        vort=tabla['circulation']/size**2
        Width0.append(np.percentile(vort,50))
        Size0.append(size)
    Width0=np.array(Width0)
    print(Width0)
    numbers=np.arange(2,128,2)

    for number in numbers:
        original=glob.glob('Difussion/*noise*n%03.1f'%exponent+'*R%03d*'%number)
        original.sort()
        vert=original[0].split('_')[-1]
        vert=vert.split('-')[0]
        vert=float(vert[1:])

        Size=[]
        Width=[]
        for original_i in original:
            size=float(original_i.split('-')[-1])
            tabla=Table.read(original_i,path='data')
            vort=tabla['circulation']/size**2
            Width.append(np.percentile(vort,84)-np.percentile(vort,16))
            Size.append(size)
        Size=np.array(Size)
        Width=np.array(Width)/Width0
        Width=Width[Size<100]
        Size=Size[Size<100]
        Width=Width-base
        maximo=Width.max()
        Width=Width/maximo
        popt, pcov = curve_fit(f_width,Size, Width,p0=[1,-1.5])
        print(popt,exponent)
        Parameters.append([vert,maximo,*popt])

    Parameters=np.array(Parameters)
    print(Parameters)
    tabla=Table()
    tabla['rt']     =   Column(Parameters[:,0])
    tabla['max']    =   Column(Parameters[:,1])
    tabla['ro']     =   Column(Parameters[:,2])
    tabla['beta']   =   Column(Parameters[:,3])

    tabla.write('Tables/Width-Function-n%03.1f'%exponent+'-wo-difussion',path='data',format='hdf5',overwrite=True)
    return True

def Circulation_Percentiles(mapa):

    N=len(mapa)
    res=list((10**np.linspace(0,np.log10(N),100)+1e-5))
    res.sort()
    res=np.array(res).astype(int)
    res=np.array(list(set(res)))
    res.sort()

    res=np.array(list(set((N/res).astype(int))))
    res.sort()
    P16=np.zeros_like(res,dtype=float)
    P25=np.zeros_like(res,dtype=float)
    P50=np.zeros_like(res,dtype=float)
    P75=np.zeros_like(res,dtype=float)
    P84=np.zeros_like(res,dtype=float)

    for k,R in enumerate(res):
        fake=block_reduce(mapa, R, func=np.sum)
        fake=fake.ravel()/R**2
        P16[k]=np.percentile(fake,16)
        P25[k]=np.percentile(fake,25)
        P50[k]=np.percentile(fake,50)
        P75[k]=np.percentile(fake,75)
        P84[k]=np.percentile(fake,84)

    return res,P16,P25,P50,P75,P84

def Circulation_Negative(mapa):

    N=len(mapa)
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
        radial=radial.ravel()*4.0e4/N1
        try:
            fake=fake[radial<1.5e4]
            Negative[k]=len(fake[fake<0])/len(fake)
        except:
            Negative[k]=0


    return res, Negative

def Circulation_sigma(mapa):

    N=len(mapa)
    res=list((10**np.linspace(0,np.log10(N),100)+1e-5))
    res.sort()
    res=np.array(res).astype(int)
    res=np.array(list(set(res)))
    res.sort()

    res=np.array(list(set((N/res).astype(int))))
    res.sort()
    Negative=np.zeros_like(res,dtype=float)

    L=np.zeros_like(res,dtype=float)

    for k,R in enumerate(res):
        fake=block_reduce(mapa, R, func=np.sum)
        N1=len(fake)
        L[k]=N1
        radial=radial_map(fake)
        fake=fake.ravel()
        radial=radial.ravel()*4.0e4/N1
        try:
            fake=fake[radial<1.5e4]
            Negative[k]=np.percentile(fake,84)-np.percentile(fake,16)
        except:
            Negative[k]=0
    Negative[np.isnan(Negative)]=0


    return res, Negative, L

def Circulation_optimum(name):

    Data=np.load(name+'_vort.npy')
    Data=Data[1:-1,1:-1]
    re,P1,P2,P5,P7,P8=Circulation_Percentiles(Data)
    Suma=Data.sum()
    Data=np.load(name+'_omeg.npy')
    Data=Data[1:-1,1:-1]
    qe,Q1,Q2,Q5,Q7,Q8=Circulation_Percentiles(Data)

    k=200.0

    kd=0.1

    while k<1000:
        kd=kd*0.99
        kd=np.round(kd,1)
        VORT=difussion_image(name,k)
        VORT*=Data.sum()/VORT.sum()
        ERROR=1e10
        while kd<100:

            VORT2=difussion(VORT,kd,mode='wrap')

            res,P16,P25,P50,P75,P84=Circulation_Percentiles(VORT2)

            err1=((P8-P1)/P5)-((P84-P16)/P50)
            err2=((P7-P2)/P5)-((P75-P25)/P50)

            err1=np.sqrt(np.mean(err1[res<100]**2))
            err2=np.sqrt(np.mean(err2[res<100]**2))
            err=np.sqrt(err1**2+err2**2)*100
            print(err)
            if err<ERROR:
                ERROR=err
            elif err<1000 :
                plt.plot(re,(P8-P1)/P5,color="#38c3ff",linestyle='-')
                plt.plot(re,(P7-P2)/P5,color="#ffdb38",linestyle='-')

                plt.plot(qe,(Q8-Q1)/Q5,color="#38c3ff",linestyle='--')
                plt.plot(qe,(Q7-Q2)/Q5,color="#ffdb38",linestyle='--')

                VORT2=difussion(VORT,kd-0.1,mode='wrap')

                res,P16,P25,P50,P75,P84=Circulation_Percentiles(VORT2)
                plt.plot(res,(P84-P16)/P50,color="#38c3ff",linestyle=':')
                plt.plot(res,(P75-P25)/P50,color="#ffdb38",linestyle=':')
                plt.ylim(0,5)
                plt.xscale('log')
                plt.title('error_%08.4f' %err)
                plt.savefig('Temp_Files/Figures/kt_%05.2f' %k+'_kd_%05.2f'%kd+'.png')
                plt.close()
                break
            del res,P16,P25,P50,P75,P84



            kd+=.01

        k*=1.025

def scaling(VORT,kd):
    VORT2=difussion(VORT,kd,mode='wrap')

    res,Neg=Circulation_Negative(VORT2)

    return Neg[0]

def Circulation_negative_optimum(name,kmin,kmax,Nk):
    print('Circulation_negative_optimum')
    Data=np.load(name+'_vort.npy')
    dx=4.0e4/len(Data)

    tab=Table.read('Circulation_data/Full-Percentiles/Negative/'+name+'-Negative',path='data')
    r0=tab['Resolution']
    N0=tab['Number']

    r0=np.insert(r0,0,0)
    r0=np.insert(r0,len(r0),1.0e5)
    N0=np.insert(N0,0,N0[0])
    N0=np.insert(N0,len(N0),0)
    Ne=interp1d(r0,N0)
    karray=np.exp(np.linspace(np.log(kmin),np.log(kmax),Nk))
    kd1=1
    kd2=50
    KD=[]
    factor=karray[1]/karray[0]
    minimo=1e10
    for nk,k in enumerate(karray):
        print('kt =',k)
        VORT=difussion_image(name,k)

        N1=scaling(VORT,kd1)-N0[0]
        N2=scaling(VORT,kd2)-N0[0]
        if N1*N2<0:
            thresh=1e3
            while thresh>2e-3:
                aux=scaling(VORT,0.5*(kd2+kd1))-N0[0]

                if N1*aux<0:
                    kd2=0.5*(kd2+kd1)
                else:
                    kd1=0.5*(kd2+kd1)

                thresh=np.abs((1-kd2/kd1)/(1+kd2/kd1))

            VORT2=difussion(VORT,kd2,mode='wrap')
            res,Neg=Circulation_Negative(VORT2)
            res=res*dx
            err=(Neg-Ne(res))**2
            err=np.sqrt(np.mean(err))
            KD.append(kd2)
            if err<minimo:
                minimo=err
            else:
                break
            if k==karray.max():
                break
            plt.plot(tab['Resolution'],tab['Number'])

            plt.plot(res,Neg,color="#38c3ff",linestyle=':')
            plt.xscale('log')
            plt.title('error_%08.4f' %err)
            plt.axvline(kd2*dx,color='k',linestyle=':')
            plt.xlim(30,5e3)

            plt.savefig('Temp_Files/Figures/Negative_kt_%06.2f' %k+'_kd_%05.2f'%kd2+'.png')
            plt.close()

            kd1=kd1/(1+factor*2)
            kd2=kd2*(1+factor*2)
    return karray[nk-1],KD[-2]

def Circulation_Negative_radii(mapa,Nbins):
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

def Circulation_negative_fits(name,k,Nbins,start):
    print('KT = ',k)
    Data=np.load(name+'_vort.npy')
    dx=4.0e4/len(Data)

    kd=start

    #print('applying noise')
    VORT=difussion_image(name,k)
    switch =-1*np.ones(Nbins)
    kdarray=np.zeros(Nbins)
    erarray=np.zeros(Nbins)
    while switch.sum()<Nbins:
        VORT1=difussion(VORT,kd,mode='wrap')
        res,rcen,Neg,redges=Circulation_Negative_radii(VORT1,Nbins)
        for j in range(Nbins):

            x,y=interp_negative_bin(name,dx*redges[j],dx*redges[j+1])
            if (Neg[j,:][0]-y[0]<0) & (switch[j]<0):
                switch[j]=1
                kdarray[j]=kd
                raux=res*dx
                Naux=Neg[j,:]
                raux=np.insert(raux,0,0)
                Naux=np.insert(Naux,0,Naux[0])
                func=interp1d(raux,Naux)
                z=func(x)
                XMAX=max(x[np.where(y>0)].max(),x[np.where(z>0)].max())
                XMIN=max(x[np.where(y>0)].min(),x[np.where(z>0)].min())
                xnew=np.exp(np.linspace(np.log(XMIN),np.log(XMAX),100))
                fy=interp1d(x,y)
                fz=interp1d(x,z)
                ynew=fy(xnew)
                znew=fz(xnew)
                znew*=ynew[0]/znew[0]
                err=(znew-ynew)**2
                err=np.mean(err)
                err=np.sqrt(err)
                erarray[j]=err


                plt.title('error_%08.4f' %err)
                plt.plot(x,y,'b-')
                plt.plot(x,z*y[0]/z[0],'r:')

                x=x[y>0]
                y=y[y>0]
                dx=x[0]
                x=x/dx
                f1=interp1d(y,x)
                x50=f1(y[0]/2)
                s=(x50-1)/(np.sqrt(2*np.log(2)))
                popt, pcov = curve_fit(gaussian, x , y, p0=[y[0],x[0],s],bounds=(-np.inf,[np.inf,1.01,np.inf]))
                plt.plot(res,gaussian(res/dx,*popt),'k-.')
                print(j,prediction_negative(*popt)[1])
                plt.xscale('log')
                plt.xlim(30,5e3)
                #plt.yscale('log')
                plt.ylim(ymin=0.0)
                plt.savefig('Temp_Files/Figures/Negative_j_%02d' %j +'_kt_%06.2f' %k+'_kd_%05.2f'%kd+'.png')
                plt.close()

                del raux,Naux,err,z

        kd+=0.1
    start=kdarray.min()
    return kdarray , erarray ,rcen*dx

def Circulation_negative_turbulence(name,Nbins,start,kmin,kmax,Nk,Ncores):
    ktarray=np.exp(np.linspace(np.log(kmin),np.log(kmax),Nk))
    my_storage = {}
    kdmatrix=[]
    errmatrix=[]
    for sto, kt in yt.parallel_objects(ktarray, Ncores, storage = my_storage):

        output = Circulation_negative_fits(name,kt,Nbins,start)
        sto.result_id = kt
        sto.result = output


    if yt.is_root():
        for fn, vals in sorted(my_storage.items()):
            kdmatrix.append(vals[0])
            errmatrix.append(vals[1])
            rcen = vals[2]
        kdmatrix=np.array(kdmatrix)
        errmatrix=np.array(errmatrix)

        KT=np.zeros(Nbins)
        KD=np.zeros(Nbins)
        for j in range(Nbins):
            optimo=np.where(errmatrix[:,j]==errmatrix[:,j].min())
            KD[j]=kdmatrix[optimo,j]
            KT[j]=ktarray[optimo]

        KK=np.array([KT,KD])
        np.save(name+'_KK',KK)
        return KT,KD

def temporal_turbulence(name,k):
    VORT=difussion_image(name,k)

    np.save(name+'_temp_%07.2f' %k,VORT)
    return name+'_temp_%07.2f' %k+'.npy'

def apply_negative_radial_fit(name,Ncores):

    KK=np.load(name+'_KK.npy')
    KT=KK[0]
    KD=KK[1]

    Nbins=len(KT)
    my_storage = {}
    maps=[]
    print('computing optimization')
    indices=range(Nbins)
    for sto, kt in yt.parallel_objects(set(KT), Ncores, storage = my_storage):

        output = temporal_turbulence(name,kt)
        sto.result_id = output
        sto.result = kt

    print('applying optimization')
    my_storage2 = {}
    for sto, ij in yt.parallel_objects(indices, Ncores, storage = my_storage2):
        mapa=apply_temp(name,KT,KD,ij,Nbins)
        sto.result_id = ij
        sto.result = mapa
    if yt.is_root():
        for fn, vals in sorted(my_storage2.items()):
            if ij==0:
                mapa=vals
            else:
                mapa+=vals
        np.save(name+'_negative_fit_%02d' %Nbins,mapa)

def apply_temp(name,KT,KD,ij,Nbins):

    kt      = KT[ij]
    kd      = KD[ij]
    temp    = np.load(name+'_temp_%07.2f' %kt+'.npy')
    temp2   = difussion(temp,kd,mode='wrap')
    temp3   = np.zeros_like(temp2)
    N       = len(temp2)
    radial  = radial_map(temp2)

    del temp

    if Nbins>1:
        redges = np.linspace(0,0.5*N,Nbins)
        redges = np.insert(redges,len(redges),radial.max())
    else:
        redges=np.array([0,radial.max()])

    ring    = (redges[ij]<radial)&(radial<redges[ij+1])
    temp3[ring]=temp2[ring]
    return temp3

def Circulation_negative_optimum_dispersion(name):

    Data=np.load(name+'_vort.npy')
    dx=4.0e4/len(Data)

    tab=Table.read('Circulation_data/Percentiles/Negative/NWR10025-Negative',path='data')
    r0=tab['Resolution']
    N0=tab['Number']

    r0=np.insert(r0,0,0)
    r0=np.insert(r0,len(r0),1.0e5)
    N0=np.insert(N0,0,N0[0])
    N0=np.insert(N0,len(N0),0)
    Ne=interp1d(r0,N0)

    k=1


    minimo=1e10
    while k<25:
        kd1=1
        kd2=30
        VORT=dispersion_image(name,k,150)

        N1=scaling(VORT,kd1)-N0[0]
        N2=scaling(VORT,kd2)-N0[0]
        print(k,kd1,kd2,N1,N2)
        if N1*N2<0:
            switch=True
            thresh=1e3
            while thresh>2e-3:
                aux=scaling(VORT,0.5*(kd2+kd1))-N0[0]

                if N1*aux<0:
                    kd2=0.5*(kd2+kd1)
                else:
                    kd1=0.5*(kd2+kd1)

                thresh=np.abs((1-kd2/kd1)/(1+kd2/kd1))

            VORT2=difussion(VORT,kd2,mode='wrap')
            res,Neg=Circulation_Negative(VORT2)
            res=res*dx
            err=(Neg-Ne(res))**2
            err=np.sqrt(np.mean(err))

            if err<minimo:
                minimo=err
            else:
                break
            plt.plot(tab['Resolution'],tab['Number'])

            plt.plot(res,Neg,color="#38c3ff",linestyle=':')
            plt.xscale('log')
            plt.title('error_%08.4f' %err)
            plt.axvline(kd2*dx,color='k',linestyle=':')


            plt.savefig('Temp_Files/Figures/Negative_kt_%06.2f' %k+'_kd_%05.2f'%kd2+'.png')
            plt.close()

            kd1=kd1/1.2
            kd2=kd2*1.2

        k+=0.1

def interp_negative(name,R):
    f = open("Temp_Files/"+name+"-All-Radial", "rb")
    res  = pickle.load(f)
    rcen = pickle.load(f)
    Neg  = pickle.load(f)
    f.close()
    if R<rcen.min():
        return res, Neg[0,:]
    if R>rcen.max():
        return res, Neg[-1,:]
    up=np.where(rcen>R)[0][0]
    low=np.where(rcen<R)[0][-1]
    inter=(R-rcen[low])*Neg[up,:]+(rcen[up]-R)*Neg[low,:]
    inter=inter/(rcen[up]-rcen[low])
    return res,inter

def interp_negative_bin(name,rmin,rmax):
    f    = open("Temp_Files/"+name+"-All-Radial", "rb")
    res  = pickle.load(f)
    rcen = pickle.load(f)
    Neg  = pickle.load(f)
    f.close()

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

def compare_negative(name,Nbins):
    tab=Table.read('Circulation_data/Full-Percentiles/Negative/'+name+'-Negative',path='data')
    r0=tab['Resolution']
    N0=tab['Number']

    r0=np.insert(r0,0,0)
    r0=np.insert(r0,len(r0),1.0e5)
    N0=np.insert(N0,0,N0[0])
    N0=np.insert(N0,len(N0),0)

    mapa=np.load(name+'_negative_fit_%02d' %Nbins+'.npy')
    res,Neg=Circulation_Negative(mapa)
    dx=4.0e4/(len(mapa)+2)
    plt.plot(res*dx,Neg,'b:')
    plt.plot(r0,N0,'r-')
    plt.xscale('log')
    plt.xlim(30,5e3)
    plt.savefig('Temp_Files/Figures/'+name+'_fit_%02d' %Nbins+'.png')
    plt.close()

def write_kt_kd(name,kt,kd):
    if os.path.isfile('Negative_Fit'):
        pass
    else:
        f1=open('Negative_Fit','w')
        f1.close()
    empty = os.stat('Negative_Fit').st_size == 0
    if empty:
        f1=open('Negative_Fit','a')
        f1.write(name+"\t %07.2f \t" %kt+ "%07.2f \n " %kd)
        f1.close()
    else:
        f1=open('Negative_Fit','r')
        switch=True
        for x in f1:
            xp=x.split('\t')
            if xp[0]==name:
                switch=False

        f1.close()
        if switch:
            f1=open('Negative_Fit','a')
            f1.write(name+"\t %07.2f \t" %kt+ "%07.2f \n " %kd)
            f1.close()

def write_KT_KD(name,KT,KD,Nbins):
    if os.path.isfile(name+'_Negative_Fit'):
        pass
    else:
        f1=open(name+'_Negative_Fit','w')
        f1.close()
    empty = os.stat(name+'_Negative_Fit').st_size == 0

    if empty:
        f1=open(name+'_Negative_Fit','a')
        f1.write('%02d' %Nbins)
        for kt in KT:
            f1.write('\t %07.2f' %kt)
        for kt in KD:
            f1.write('\t %07.2f' %kt)
        f1.write('\n')
        f1.close()
    else:
        number='%02d' %Nbins
        f1=open(name+'_Negative_Fit','r')
        switch=True
        for x in f1:
            xp=x.split('\t')
            if xp[0]==number:
                switch=False

        f1.close()
        if switch:
            f1=open(name+'_Negative_Fit','a')
            f1.write('%02d' %Nbins)
            for kt in KT:
                f1.write('\t %07.2f' %kt)
            for kt in KD:
                f1.write('\t %07.2f' %kt)
            f1.write('\n')
            f1.close()

def Circulation_negative_optimum_mpi(name,kmin,kmax,Nk,Ncores):
    ktarray=np.exp(np.linspace(np.log(kmin),np.log(kmax),Nk))
    my_storage = {}
    kdarray=[]
    erarray=[]
    for sto, kt in yt.parallel_objects(ktarray, Ncores, storage = my_storage):

        output = Circulation_negative_optimum_thread(name,kt)
        sto.result_id = kt
        sto.result = output

    if yt.is_root():
        for fn, vals in sorted(my_storage.items()):
            kdarray.append(vals[0])
            erarray.append(vals[1])
        kdarray=np.array(kdarray)
        erarray=np.array(erarray)

        optimo=np.where(erarray==erarray.min())
        return ktarray[optimo],kdarray[optimo]

def Circulation_negative_optimum_thread(name,k):
    print('Circulation_negative_optimum_thread')
    Data=np.load(name+'_vort.npy')
    dx0=4.0e4/len(Data)

    tab=Table.read('Circulation_data/Full-Percentiles/Negative/'+name+'-Negative',path='data')
    r0=tab['Resolution']*2
    N0=tab['Number']

    r0=np.insert(r0,0,0)
    r0=np.insert(r0,len(r0),1.0e5)
    N0=np.insert(N0,0,N0[0])
    N0=np.insert(N0,len(N0),0)
    Ne=interp1d(r0,N0)
    kd1=1
    kd2=50

    print('kt =',k)
    VORT=difussion_image(name,k)

    N1=scaling(VORT,kd1)-N0[0]
    N2=scaling(VORT,kd2)-N0[0]
    if N1*N2<0:
        thresh=1e3
        while thresh>2e-3:
            aux=scaling(VORT,0.5*(kd2+kd1))-N0[0]

            if N1*aux<0:
                kd2=0.5*(kd2+kd1)
            else:
                kd1=0.5*(kd2+kd1)

            thresh=np.abs((1-kd2/kd1)/(1+kd2/kd1))

        VORT2=difussion(VORT,kd2,mode='wrap')
        res,Neg=Circulation_Negative(VORT2)
        res=res*dx0
        aNeg=Ne(res)
        XMAX=max(res[np.where(Neg>0)].max(),res[np.where(aNeg>0)].max())
        XMIN=max(res[np.where(Neg>0)].min(),res[np.where(aNeg>0)].min())
        xnew=np.exp(np.linspace(np.log(XMIN),np.log(XMAX),1000))
        fy=interp1d(res,Neg)
        fz=interp1d(res,aNeg)
        ynew=fy(xnew)
        znew=fz(xnew)
        ynew*=znew[0]/ynew[0]
        err=(ynew-znew)**2
        err=np.sqrt(np.mean(err))

        n_slope=get_slope(name,0,1.5e4)
        plt.plot(tab['Resolution']*2,tab['Number'])
        x=tab['Resolution']
        y=tab['Number']
        x=x[y>0]
        y=y[y>0]
        dx=x[0]
        x=2*x/dx
        f1=interp1d(y,x)
        x50=f1(y[0]/2)
        s=(x50-1)/(np.sqrt(2*np.log(2)))
        popt, pcov = curve_fit(gaussian, x , y, p0=[y[0],x[0]/2.0,s],bounds=(-np.inf,[np.inf,1.01,np.inf]))
        plt.plot(res,gaussian(res/dx0,*popt),'k-.')
        print(*popt)
        print(prediction_negative_n(*popt,n_slope))
        plt.plot(res,Neg,color="#38c3ff",linestyle=':')
        plt.xscale('log')
        plt.title('error_%08.4f' %err)
        plt.axvline(kd2*dx,color='k',linestyle=':')
        plt.xlim(30,5e3)
        plt.savefig('Temp_Files/Figures/Negative_kt_%06.2f' %k+'_kd_%05.2f'%kd2+'.png')
        plt.close()


    return kd2,err
