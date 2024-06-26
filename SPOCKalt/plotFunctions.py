import numpy as np
import pandas as pd
import rebound



def farey_sequence(n):
    """Return the nth Farey sequence as order pairs of the form (N,D) where `N' is the numerator and `D' is the denominator."""
    a, b, c, d = 0, 1, 1, n
    sequence=[(a,b)]
    while (c <= n):
        k = int((n + b) / d)
        a, b, c, d = c, d, (k*c-a), (k*d-b)
        sequence.append( (a,b) )
    return sequence

def resonant_period_ratios(min_per_ratio,max_per_ratio,order):
    """Return the period ratios of all resonances up to order 'order' between 'min_per_ratio' and 'max_per_ratio' """
    if min_per_ratio < 0.:
        raise AttributeError("min_per_ratio of {0} passed to resonant_period_ratios can't be < 0".format(min_per_ratio))
    if max_per_ratio >= 1.:
        raise AttributeError("max_per_ratio of {0} passed to resonant_period_ratios can't be >= 1".format(max_per_ratio))
    minJ = int(np.floor(1. /(1. - min_per_ratio)))
    maxJ = int(np.ceil(1. /(1. - max_per_ratio)))
    res_ratios=[(minJ-1,minJ)]
    for j in range(minJ,maxJ):
        res_ratios = res_ratios + [ ( x[1] * j - x[1] + x[0] , x[1] * j + x[0]) for x in farey_sequence(order)[1:] ]
    res_ratios = np.array(res_ratios)
    msk = np.array( list(map( lambda x: min_per_ratio < x[0]/float(x[1]) < max_per_ratio , res_ratios )) )
    return res_ratios[msk]

import rebound
def ratio_list(sim: rebound.Simulation, i1, i2):
    maxorder = 5
    ps = sim.particles
    #print('test')
    n1 = ps[i1].n
    n2 = ps[i2].n

    m1 = ps[i1].m/ps[0].m
    m2 = ps[i2].m/ps[0].m

    Pratio = n2/n1
    #next want to try not to abreviate to closest

    delta = 0.03
    if Pratio < 0 or Pratio > 1: # n < 0 = hyperbolic orbit, Pratio > 1 = orbits are crossing
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    minperiodratio = max(Pratio-delta, 0.)
    maxperiodratio = min(Pratio+delta, 0.99) # too many resonances close to 1
    res = resonant_period_ratios(minperiodratio,maxperiodratio, order=maxorder)
    return res


def ratio(sim: rebound.Simulation, i1, i2):
    ps = sim.particles
    real = ps[i1].P/ ps[i2].P
    which = int
    val = [10000,10]
    #print(ratio_list(sim,i1,i2))
    for i,each in enumerate(ratio_list(sim,i1,i2)):
        if np.abs((each[0]/each[1])-real)<np.abs((val[0]/val[1])-real):
            #which = i
            val = each
    return val

# def getTheta(sim, i1,i2, rat):
#     ps = sim.particles
#     #rat = ratio(sim, i1,i2)
#     #print(rat)
#     #evec = (ps[i2].e*np.cos(ps[i2].pomega)-ps[i1].e*np.cos(ps[i1].pomega), ps[i2].e*np.sin(ps[i2].pomega)-ps[i1].e*np.sin(ps[i1].pomega))
#     #pomega = np.arctan(evec[1]/evec[0])
#     evec2 = ps[i2].e*np.exp(1j*ps[i2].pomega)
#     evec1 = ps[i1].e*np.exp(1j*ps[i1].pomega)
#     erel = evec2-evec1
#     pomegarel=np.angle(erel)
#     #theta = (rat[1]*ps[i2].l) -(rat[0]*ps[i1].l)-(lambda x : x+(2*np.pi) if x<0 else x )(pomegarel)
#     theta = (rat[1]*ps[i2].l) -(rat[0]*ps[i1].l)-(rat[1]-rat[0])*pomegarel

#    return theta

def getPomega(sim, i1, i2):
    ps = sim.particles
    evec2 = ps[i2].e*np.exp(1j*ps[i2].pomega)
    evec1 = ps[i1].e*np.exp(1j*ps[i1].pomega)
    erel = evec2-evec1
    pomegarel=np.angle(erel)
    return pomegarel

def retTheta(la,lb, pomegarel, Pratio: list):
    maxorder = 5
    delta = 0.03
    minperiodratio = max(Pratio-delta, 0.)
    maxperiodratio = min(Pratio+delta, 0.99) # too many resonances close to 1
    res = resonant_period_ratios(minperiodratio,maxperiodratio, order=maxorder)
    val = [10000000,10]
    for i,each in enumerate(res):
        if np.abs((each[0]/each[1])-Pratio)<np.abs((val[0]/val[1])-Pratio):
            #which = i
            
            val = each
    
    theta = (val[1]*lb) -(val[0]*la)-(val[1]-val[0])*pomegarel

    return np.mod(theta, 2*np.pi)

def calcTheta(la,lb,pomegarel, val):
    theta = (val[1]*lb) -(val[0]*la)-(val[1]-val[0])*pomegarel
    return np.mod(theta, 2*np.pi)

def calcThetaMod2(la,lb,pomegarel, val):
    theta = (val[1]*lb) -(val[0]*la)-(val[1]-val[0])*pomegarel
    return np.mod(theta, 2*np.pi)
def calcThetaMod1(la,lb,pomegarel, val):
    theta = (val[1]*lb) -(val[0]*la)-(val[1]-val[0])*pomegarel
    return np.mod(np.mod(theta, 2*np.pi)-np.pi,2*np.pi)

def getval( Pratio: list):
    maxorder = 5
    delta = 0.03
    minperiodratio = Pratio-delta
    maxperiodratio = Pratio+delta # too many resonances close to 1
    if maxperiodratio >.999:
        maxperiodratio =.999
    res = resonant_period_ratios(minperiodratio,maxperiodratio, order=maxorder)
    val = [10000000,10]
    for i,each in enumerate(res):
        if np.abs((each[0]/each[1])-Pratio)<np.abs((val[0]/val[1])-Pratio):
            #which = i
            
            val = each
    
    

    return val


def get_data(sim, Nint, Nout):
    '''gets dataframe '''
    times = np.linspace(0,Nint,Nout)
    data = pd.DataFrame()
    theta12 = np.zeros(Nout)
    theta23 = np.zeros(Nout)
    p2p1 = np.zeros(Nout)
    p3p2 = np.zeros(Nout)
    e1 = np.zeros(Nout)
    e2 = np.zeros(Nout)
    e3 = np.zeros(Nout)
    l1 = np.zeros(Nout)
    l2 = np.zeros(Nout)
    l3 = np.zeros(Nout)
    pomegarel12 = np.zeros(Nout)
    pomegarel23 = np.zeros(Nout)
    theta12 = np.zeros(Nout)
    theta23 = np.zeros(Nout)
    
    
    ps = sim.particles
    for i,each in enumerate(times):
        p2p1[i] = ((ps[2].P/ps[1].P))
        p3p2[i]=((ps[3].P/ps[2].P))
        e1[i]=(ps[1].e)
        e2[i]=(ps[2].e)
        e3[i]=(ps[3].e)
        l1[i]=(ps[1].l)
        l2[i]=(ps[2].l)
        l3[i]=(ps[3].l)
        pomegarel12[i]=(getPomega(sim,1,2))
        pomegarel23[i]=(getPomega(sim,2,3))

        sim.integrate(each, exact_finish_time=0)

    Pratio12 = 1/np.median(p2p1)
    Pratio32 = 1/np.median(p3p2)
    pval12 = getval(Pratio12)
    pval32 = getval(Pratio32)
    #print(pomegarel12)
    for x in range(Nout):
        theta12[x]=calcTheta(l1[x],l2[x],pomegarel12[x],pval12)
        theta23[x]=calcTheta(l2[x],l3[x],pomegarel23[x],pval32)
    
    data=pd.DataFrame({'time':times,'p2/p1':p2p1,'p3/p2':p3p2,'theta12':theta12,'theta23':theta23,'e1':e1,'e2':e2,'e3':e3})
    return data,pval12,pval32


# def plot_data(sim,Nout,Nint):
#     data = get_data(sim,Nout,Nint)
#     data.plot.scatter(x="p2/p1", y="p3/p2",s=2, c="time", colormap="copper", alpha=.5)
#     data.plot.scatter(x="time", y="theta12",figsize=(14,7),s=1)
#     data.plot.scatter(x="time", y="theta23",figsize=(14,7),s=1)
#     # data.plot.scatter(x="time", y="rat",figsize=(14,7),s=1)

#     data.plot(x='time',y=['e1','e2','e3'],figsize=(14,7))
#     rebound.OrbitPlot(sim)

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
def plot_data(sim,Nout,Nint):
    figure = plt.figure(figsize=[20,35])
    gs = GridSpec(4, 2, figure=figure)
    #gs.update(wspace = .1, hspace = .1)
    
    data = get_data(sim,Nout,Nint)
    ax1 = plt.subplot(gs[0,0])
    data.plot.scatter(ax = ax1,x="p2/p1", y="p3/p2",s=2, c="time", colormap="copper", alpha=.5)
    ax2 = plt.subplot(gs[1,:2])
    data.plot.scatter(ax=ax2,x="time", y="theta12",s=1)
    ax3 = plt.subplot(gs[2,:2])
    data.plot.scatter(ax = ax3,x="time", y="theta23",s=1)
    ax4 = plt.subplot(gs[3,:2])
    data.plot(ax=ax4,x='time',y=['e1','e2','e3'])
    ax5 = plt.subplot(gs[0,1])
    ax5.set_aspect('equal')
    rebound.OrbitPlot(sim,fig=figure, ax=ax5,ylim=[-3,3],xlim=[-3,3])

    return figure

    