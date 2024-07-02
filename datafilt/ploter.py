import numpy as np
import rebound
import sys
#Intigration/simsetup.py
#import SPOCKalt
sys.path.insert(1, '..')
#print(path)
from SPOCKalt import *
sys.path.insert(1, '../SPOCKalt')
#Intigration/simsetup.py
from SPOCKalt import featureKlassifier
from SPOCKalt import simsetup
import plotFunctions


def getRatL( Pratio: list,orderL):
    maxorder = max(orderL)
    delta = 0.03
    minperiodratio = Pratio-delta
    maxperiodratio = Pratio+delta # too many resonances close to 1
    if maxperiodratio >.999:
        maxperiodratio =.999
    res = plotFunctions.resonant_period_ratios(minperiodratio,maxperiodratio, order=maxorder)
    resList = [[]]*len(orderL)
    
    for each in res:
        resList[each[1]-each[0]-1] = resList[each[1]-each[0]-1]+[each]
    finals = []
    #print(resList)
    for eachO in resList:
        if len(eachO) !=0:
            val = [10000000,10]
            for i,each in enumerate(eachO):
                if np.abs((each[0]/each[1])-Pratio)<np.abs((val[0]/val[1])-Pratio):
                    val = each
            finals.append(val)
    return finals
    


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
        pomegarel12[i]=(plotFunctions.getPomega(sim,1,2))
        pomegarel23[i]=(plotFunctions.getPomega(sim,2,3))

        sim.integrate(each, exact_finish_time=0)

    Pratio12 = 1/np.median(p2p1)
    Pratio23 = 1/np.median(p3p2)
    OrderL = range(1,11)
    #OrderL=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,]
    rat12 = getRatL(Pratio12,OrderL)
    thetalist12 = [[np.nan]*Nout]*len(rat12)
    #print(len(thetalist12[0]))
    for i,r in enumerate(rat12):
        for x in range(Nout):
                #
                # print(r)
                thetalist12[i][x]=plotFunctions.calcTheta(l1[x],l2[x],pomegarel12[x],r)
        #print(thetalist12[i])
        thetalist12[i]= np.unwrap(thetalist12[i])
    
    stds12 = list(map(np.std,thetalist12))
    which12 = stds12.index(min(stds12))
    theta12 = thetalist12[which12]
    pval12 = rat12[which12]
    

    rat23 = getRatL(Pratio23,OrderL)
    thetalist23 = [[np.nan]*Nout]*len(rat23)
    for i,r in enumerate(rat23):
        for x in range(Nout):
                thetalist23[i][x]=plotFunctions.calcTheta(l2[x],l3[x],pomegarel23[x],r)
        thetalist23[i]= np.unwrap(thetalist23[i])
    stds23 = list(map(np.std,thetalist23))
    which23 = stds23.index(min(stds23))
    theta23 = thetalist23[which23]
    pval23 = rat23[which23]
    
    
    
    
    #theta12 = np.mod(theta12,2*np.pi)
    #theta23 = np.mod(theta23,2*np.pi)
    # print(np.log10(np.std(theta12)/1.8))
    # print(np.log10(np.std(theta23)/1.8))
    

    data=pd.DataFrame({'time':times,'p2/p1':p2p1,'p3/p2':p3p2,'theta12':theta12,'theta23':theta23,'e1':e1,'e2':e2,'e3':e3})
    return data,pval12,pval23

import pandas as pd


initial = pd.read_csv('../modeldata/originalCondAllData.csv')
dataset = pd.read_csv('../modeldata/trydifOrdSTD.csv')


import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
def get_plot(num,Nout=5000,Nint=100000):
    sim = simsetup.get_simList(initial.iloc[num,2:])
    simsetup.init_sim_parameters(sim)
    figure = plt.figure(figsize=[20,35])
    gs = GridSpec(4, 2, figure=figure)
    #gs.update(wspace = .1, hspace = .1)
    ps = sim.particles
    et12 = np.abs((ps[2].e*np.exp(1j*ps[2].pomega))-(ps[1].e*np.exp(1j*ps[1].pomega)))/((ps[2].a-ps[1].a)/ps[2].a)
    et23 = np.abs((ps[3].e*np.exp(1j*ps[3].pomega))-(ps[2].e*np.exp(1j*ps[2].pomega)))/((ps[3].a-ps[2].a)/ps[3].a)

    
    data, res12, res23 = get_data(sim,Nout,Nint)
    ax1 = plt.subplot(gs[0,0])
    ax1.set_title('threeBRfillfac:' +str(dataset['threeBRfillfac'][num]))
    data.plot.scatter(ax = ax1,x="p2/p1", y="p3/p2",s=2, c="time", colormap="copper", alpha=.35)
    ax2 = plt.subplot(gs[1,:2])
    ax2.set_title(str(res12[1])+':'+str(res12[0])+'   fracOfCross:'+str(et12))
    data.plot.scatter(ax=ax2,x="time", y="theta12",s=1)
    ax3 = plt.subplot(gs[2,:2])
    ax3.set_title(str(res23[1])+':'+str(res23[0])+'   fracOfCross:'+str(et23))
    data.plot.scatter(ax = ax3,x="time", y="theta23",s=1)
    ax4 = plt.subplot(gs[3,:2])
    data.plot(ax=ax4,x='time',y=['e1','e2','e3'])
    ax5 = plt.subplot(gs[0,1])
    ax5.set_title(str(num))
    ax5.set_aspect('equal')
    rebound.OrbitPlot(sim,fig=figure, ax=ax5,ylim=[-3,3],xlim=[-3,3])
    #plt.savefig(f'imgs/'+str(num)+'h.png')
    #plt.show(False)


def insimPlot(sim,Nout=5000,Nint=100000):
    simsetup.init_sim_parameters(sim)
    figure = plt.figure(figsize=[20,35])
    gs = GridSpec(4, 2, figure=figure)
    #gs.update(wspace = .1, hspace = .1)
    ps = sim.particles
    et12 = np.abs((ps[2].e*np.exp(1j*ps[2].pomega))-(ps[1].e*np.exp(1j*ps[1].pomega)))/((ps[2].a-ps[1].a)/ps[2].a)
    et23 = np.abs((ps[3].e*np.exp(1j*ps[3].pomega))-(ps[2].e*np.exp(1j*ps[2].pomega)))/((ps[3].a-ps[2].a)/ps[3].a)

    
    data, res12, res23 = get_data(sim,Nout,Nint)
    ax1 = plt.subplot(gs[0,0])
    #ax1.set_title('threeBRfillfac:' +str(dataset['threeBRfillfac'][num]))
    data.plot.scatter(ax = ax1,x="p2/p1", y="p3/p2",s=2, c="time", colormap="copper", alpha=.35)
    ax2 = plt.subplot(gs[1,:2])
    ax2.set_title(str(res12[1])+':'+str(res12[0])+'   fracOfCross:'+str(et12))
    data.plot.scatter(ax=ax2,x="time", y="theta12",s=1)
    ax3 = plt.subplot(gs[2,:2])
    ax3.set_title(str(res23[1])+':'+str(res23[0])+'   fracOfCross:'+str(et23))
    data.plot.scatter(ax = ax3,x="time", y="theta23",s=1)
    ax4 = plt.subplot(gs[3,:2])
    data.plot(ax=ax4,x='time',y=['e1','e2','e3'])
    ax5 = plt.subplot(gs[0,1])
    #ax5.set_title(str(num))
    ax5.set_aspect('equal')
    rebound.OrbitPlot(sim,fig=figure, ax=ax5,ylim=[-1,1],xlim=[-1,1])