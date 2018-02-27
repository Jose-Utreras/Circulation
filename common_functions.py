import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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

def radial_map(mapa):
    xmax,ymax=np.shape(mapa)

    X=np.array(list(np.reshape(range(xmax),(1,xmax)))*xmax)

    Y=X.T
    X=np.reshape(X,(xmax,xmax))-xmax/2
    Y=np.reshape(Y,(xmax,xmax))-ymax/2
    R=np.sqrt(X**2+Y**2)

    return R
