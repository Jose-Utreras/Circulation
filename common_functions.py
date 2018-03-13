import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import fileinput
from scipy import fftpack
from scipy.interpolate import interp1d
from scipy.fftpack import fft, ifft
from scipy.optimize import curve_fit
from scipy.stats import kurtosis,skew


def Heaviside(x):
    return np.piecewise(x,[x<0,x>=0], [lambda x: 0, lambda x: 1])

def distance(p,q):
    return np.sqrt((p[0]-q[0])**2+(p[1]-q[1])**2+(p[2]-q[2])**2)

def colorplot(number,n):
    dz=np.linspace(0,number,number+1)

    norm = plt.Normalize()
    colors = plt.cm.jet(norm(dz))
    return colors[n]

def gaussian(x, a, mu, sig):
    return a*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def gaussian_step(x, a, mu, sig,step):
    return step+a*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def radial_map(mapa):
    xmax,ymax=np.shape(mapa)

    X=np.array(list(np.reshape(range(xmax),(1,xmax)))*xmax)

    Y=X.T
    X=np.reshape(X,(xmax,xmax))-(xmax-1)/2
    Y=np.reshape(Y,(xmax,xmax))-(ymax-1)/2
    R=np.sqrt(X**2+Y**2)

    return R

def radial_map_N(N1,N2):
    xmax,ymax=N1,N2

    X=np.array(list(np.reshape(range(xmax),(1,xmax)))*xmax)

    Y=X.T
    X=np.reshape(X,(xmax,xmax))-(xmax-1)/2
    Y=np.reshape(Y,(xmax,xmax))-(ymax-1)/2
    R=np.sqrt(X**2+Y**2)

    return R

def change_word_infile(filename,text_to_search,replacement_text):
    with fileinput.FileInput(filename, inplace=True, backup='.bak') as file:
        for line in file:
            print(line.replace(text_to_search, replacement_text), end='')

def Fourier_2D(mapa,L):
    N=len(mapa)
    noise=np.random.normal(0,10,size=(N,N))
    dx=L/N
    k1 = fftpack.fftfreq(N,d=dx)

    kx,ky = np.meshgrid(k1, k1)
    KX = fftpack.fftshift( kx )
    KY = fftpack.fftshift( ky )
    K=np.sqrt(KX**2+KY**2)

    F1=fftpack.fft2( mapa )
    F1=fftpack.fftshift( F1 )

    return K,F1

def Fourier_map(mapa,k,Nbins):
    kedges=np.linspace(k.min(),k.max(),Nbins+1)
    kbins=0.5*(kedges[1:]+kedges[:-1])
    Profile=np.zeros(Nbins)
    for i in range(Nbins):
        ring=(kedges[i]<k)&(k<kedges[i+1])
        Profile[i]=np.median(mapa[ring])
    return kbins,Profile

def power_law_map(n,N,rc):

    Omega=np.ones((N,N))

    X=np.array(list(np.reshape(range(N),(1,N)))*N)

    Y=X.T
    X=np.reshape(X,(N,N))-0.5*(N-1)
    Y=np.reshape(Y,(N,N))-0.5*(N-1)
    R=np.sqrt(X**2+Y**2)
    R=2*R/N
    Omega/=(R+rc)**n
    Omega/=Omega[int(N/2)][-1]
    Vort=(2-n)*Omega

    return Vort

def Knumber(mapa,L):
    N=len(mapa)
    noise=np.random.normal(0,10,size=(N,N))
    dx=L/N
    k1 = fftpack.fftfreq(N,d=dx)

    kx,ky = np.meshgrid(k1, k1)
    KX = fftpack.fftshift( kx )
    KY = fftpack.fftshift( ky )
    K=np.sqrt(KX**2+KY**2)
    return K

def map_profile(mapa):
    N=len(mapa)
    R=2*radial_map_N(N,N)/N
    h=2*(np.percentile(mapa,75)-np.percentile(mapa,25))/N**(1.0/3.0)
    Nbins=np.percentile(mapa,99)-np.percentile(mapa,1)
    Nbins=4*int(Nbins/h)

    Redges=np.linspace(0,R.max(),Nbins+1)
    Rcen=0.5*(Redges[1:]+Redges[:-1])
    hist=np.zeros(Nbins)
    hstd=np.zeros(Nbins)
    weights=np.zeros(Nbins)

    for k in range(Nbins):
        ring=(Redges[k]<=R)&(R<Redges[k+1])
        hist[k]=np.mean(mapa[ring])
        hstd[k]=np.std(mapa[ring])
        weights[k]+=len(mapa[ring].ravel())


    hstd=np.insert(hstd,0,hstd[0])
    hist=np.insert(hist,0,2*hist[0]-hist[1])
    Rcen=np.insert(Rcen,0,0)
    weights=np.insert(weights,0,1)

    hist=np.insert(hist,len(hist),hist.min())
    Rcen=np.insert(Rcen,len(Rcen),np.sqrt(2)*1.5)
    hstd=np.insert(hstd,len(hstd),hstd[-1])
    weights=np.insert(weights,len(weights),1)

    n_seed=np.log(h.max()/hist[hist>0].min())/np.log(Rcen.max()/Rcen[Rcen>0].min())
    popt, pcov = curve_fit(two_functions, Rcen[(Rcen>0)&(hist>0)], hist[(Rcen>0)&(hist>0)],
        p0=[np.mean(hist),n_seed,n_seed ,np.mean(Rcen)])
    temp=two_functions(Rcen,*popt)
    bad_values=(temp==np.isnan)|(temp==np.inf)|(temp==-np.inf)
    temp[bad_values]=hist[bad_values]
    temp=(hist+temp)/2
    #temp[temp<0]=0
    return Rcen,temp,hstd,weights

def standard_deviation_from_map(mapa):
    N=len(mapa)
    R=2*radial_map_N(N,N)/N
    h=2*(np.percentile(mapa,75)-np.percentile(mapa,25))/N**(1.0/3.0)
    Nbins=np.percentile(mapa,99)-np.percentile(mapa,1)
    Nbins=int(Nbins/h)

    Redges=np.linspace(0,R.max(),Nbins+1)
    Rcen=0.5*(Redges[1:]+Redges[:-1])
    hstd=np.zeros(Nbins)
    weights=np.zeros(Nbins)
    wskew=np.zeros(Nbins)
    sskew=np.zeros(Nbins)

    for k in range(Nbins):
        ring=(Redges[k]<=R)&(R<Redges[k+1])
        hstd[k]=np.std(mapa[ring])
        weights[k]+=len(mapa[ring].ravel())
        nn=len(mapa[ring])
        wskew[k]=skew(mapa[ring])
        sskew[k]=np.sqrt(6*nn*(nn-1)/((nn-2)*(nn+1)*(nn+3)))


    return Rcen,hstd,weights,wskew,sskew

def get_distribution(mapa):
    new=mapa.ravel()
    h=np.percentile(new,75)-np.percentile(new,25)
    h*=2
    h/=len(new)**(1.0/3.0)
    Nbins=np.percentile(new,99.9)-np.percentile(new,0.1)
    Nbins=int(Nbins/h)

    h,bins=np.histogram(new,Nbins,normed=True)
    center=0.5*(bins[1:]+bins[:-1])

    center=center[h>0]
    h=h[h>0]
    """
    g=simpson_F(center,h)

    f1=interp1d(g/g[-1],center)

    thresh=f1(0.9999)

    new_h=h[center>thresh]
    new_c=center[center>thresh]

    lx=np.log(new_c)
    ly=np.log(new_h)

    m=np.sum((ly-ly[0])*(lx-lx[0]))/np.sum((lx-lx[0])**2)
    h[center>thresh]=new_h[0]*(new_c/new_c[0])**m
    """
    h=np.insert(h,0,0)
    center=np.insert(center,0,bins[0])

    h=np.insert(h,0,0)
    center=np.insert(center,0,-1e10)

    h=np.insert(h,len(h),0)
    center=np.insert(center,len(center),bins[-1])

    h=np.insert(h,len(h),0)
    center=np.insert(center,len(center),1e10)

    func=interp1d(center,h)
    return func, Nbins

def simpson_array(x,y):
    w=3*np.ones_like(y)
    w[::3]-=1
    w[0]=1
    w[-1]=1

    dx=3*(x[1]-x[0])/8
    suma=(y*w).sum()*dx
    return suma

def simpson_F(x,y):
    w=3*np.ones_like(y)
    w[::3]-=1
    w[0]=1
    w[-1]=1

    dx=3*(x[1]-x[0])/8
    suma=np.cumsum(y*w)*dx
    return suma

def map_from_profile(R,y,N):

    fun=interp1d(R,y)
    Radius=2*radial_map_N(N,N)/N
    return fun(Radius)

def two_functions(x,A,n1,n2,x0):
    B=A*x0**(-n1+n2)
    return np.piecewise(x,[x<x0,x>=x0],[lambda x: A/x**n1, lambda x: B/x**n2])

def square_function(x,a,b,c):
    return np.polyval([a,b,c],x)
