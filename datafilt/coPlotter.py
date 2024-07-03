import numpy as np
import pandas as pd
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

dataset = pd.read_csv('../modeldata/zfixed3brfill.csv')


initial = pd.read_csv('../modeldata/originalCondAllData.csv')

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
    

def calcTheta(la,lb,pomegarel, val):
    theta = (val[1]*lb) -(val[0]*la)-(val[1]-val[0])*pomegarel
    return np.mod(theta, 2*np.pi)



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
    e1v = np.zeros(Nout,dtype='complex')
    e2v = np.zeros(Nout,dtype='complex')
    e3v = np.zeros(Nout,dtype='complex')
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
        e1v[i]=ps[1].e*np.exp(1j*ps[1].pomega)
        e2v[i]=ps[2].e*np.exp(1j*ps[2].pomega)
        e3v[i]=ps[3].e*np.exp(1j*ps[3].pomega)





        sim.integrate(each, exact_finish_time=0)

    Pratio12 = 1/np.median(p2p1)
    Pratio23 = 1/np.median(p3p2)
    OrderL = range(1,11)
    rat12 = getRatL(Pratio12,OrderL)
    thetalist12 = [[np.nan]*Nout]*len(rat12)
    #print(len(thetalist12[0]))
    for i,r in enumerate(rat12):
        for x in range(Nout):
                #
                # print(r)
                thetalist12[i][x]=calcTheta(l1[x],l2[x],pomegarel12[x],r)
        #print(thetalist12[i])
        thetalist12[i]= np.unwrap(thetalist12[i])
    
    stds12 = list(map(np.std,thetalist12))
    which12 = stds12.index(min(stds12))
    theta12 = thetalist12[which12]
    pval12 = rat12[which12]
###########
    theta12df = pd.DataFrame()
    theta12df['time']=times
    for each in range(pval12[1]-pval12[0]+1):
        c1 = each
        c2 = pval12[1]-pval12[0]-each
        theta12df['-'+str(c1)+'w1-'+str(c2)+'w2']=np.unwrap((pval12[1]*l2) -(pval12[0]*l1))-(c1*np.angle(e1v))-(c2*np.angle(e2v))
        #(c1*np.angle(e1v))-(c2*np.angle(e2v))
    

    rat23 = getRatL(Pratio23,OrderL)
    thetalist23 = [[np.nan]*Nout]*len(rat23)
    for i,r in enumerate(rat23):
        for x in range(Nout):
                thetalist23[i][x]=calcTheta(l2[x],l3[x],pomegarel23[x],r)
        thetalist23[i]= np.unwrap(thetalist23[i])
    stds23 = list(map(np.std,thetalist23))
    which23 = stds23.index(min(stds23))
    theta23 = thetalist23[which23]
    pval23 = rat23[which23]

    theta23df = pd.DataFrame()
    theta23df['time']=times
    for each in range(pval23[1]-pval23[0]+1):
        c1 = each
        c2 = pval23[1]-pval23[0]-each
        theta23df['-'+str(c1)+'w1-'+str(c2)+'w2']=np.unwrap((pval23[1]*l2) -(pval23[0]*l1))-(c1*np.angle(e2v))-(c2*np.angle(e3v))


    # #FIXME
    # plotdf = pd.DataFrame()
    # for i, each in enumerate(rat23):
    #     plotdf[str(rat23[i][1])+':'+str(rat23[i][0])+'    std:'+str(np.log10(stds23[i]/1.8))]= thetalist23[i]
    
    # plotdf.plot(figsize=(10,10))
    
    
    
    # print(np.log10(np.std(theta12)/1.8))
    # print(np.log10((np.std(theta23))/1.8))
    # print((np.median(theta23)))
    # print((np.mean(theta23)))
    #theta12 = np.mod(theta12,2*np.pi)
    #theta23 = np.mod(theta23,2*np.pi)
    #'erel12':np.abs(e2v-e1v), 'erel23':np.abs(e3v-e2v)
    #'erel12':erel12, 'erel23':erel23

    data=pd.DataFrame({'time':times,'p2/p1':p2p1,'p3/p2':p3p2,'theta12':theta12,'theta23':theta23,'e1':e1,'e2':e2,'e3':e3, 'erel12':np.abs(e2v-e1v), 'erel23':np.abs(e3v-e2v)})
    return data,pval12,pval23, theta12df, theta23df




import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
def get_plot(num,Nout=5000,Nint=100000):
    sim = simsetup.get_simList(initial.iloc[num,2:])
    simsetup.init_sim_parameters(sim)
    figure = plt.figure(figsize=[20,45])
    gs = GridSpec(5, 2, figure=figure)
    #gs.update(wspace = .1, hspace = .1)
    
    data, res12, res23, theta12df,theta23df = get_data(sim,Nout,Nint)
    ax1 = plt.subplot(gs[0,0])
    ax1.set_title('threeBRfillfac:' +str(dataset['threeBRfillfac'][num]))
    data.plot.scatter(ax = ax1,x="p2/p1", y="p3/p2",s=2, c="time", colormap="copper", alpha=.35)
    ax2 = plt.subplot(gs[1,:2])
    ax2.set_title(str(res12[1])+':'+str(res12[0]))
    ax2.set_ylim([data['theta12'].min(),data['theta12'].max()])

    data.plot(ax=ax2,x="time", y="theta12",linestyle='none',marker='.',markersize=2,c='red')
    theta12df.plot(ax=ax2,x=theta12df.columns[0],y=theta12df.columns[1:],marker='.',linestyle='none',markersize=1)

    #for i,each in enumerate(theta12df.columns[1:]):
        #theta12df.plot.scatter(ax=ax2,x=theta12df.columns[0],y=theta12df.columns[i],s=1)
    #,y=list(theta12df.columns[1:]
    ax3 = plt.subplot(gs[2,:2])
    ax3.set_title(str(res23[1])+':'+str(res23[0]))
    #print(data['theta23'].min())
    ax3.set_ylim([data['theta23'].min(),data['theta23'].max()])
    #ax3.set_ylim=[0,9]
    data.plot(ax = ax3,x="time", y="theta23",linestyle='none',marker='.',markersize=2,c='red')
    theta23df.plot(ax=ax3,x=theta23df.columns[0],y=theta23df.columns[1:],marker='.',linestyle='none',markersize=1)

    ax4 = plt.subplot(gs[3,:2])
    data.plot(ax=ax4,x='time',y=['e1','e2','e3'])
    ax5 = plt.subplot(gs[0,1])
    ax5.set_title(str(num))
    ax5.set_aspect('equal')
    ax6 = plt.subplot(gs[4,:2])
    data.plot(ax=ax6, x='time', y=['erel12','erel23'])
    rebound.OrbitPlot(sim,fig=figure, ax=ax5,ylim=[-3,3],xlim=[-3,3])
    plt.savefig(f'imgs/'+str(num)+'CoPlot.png')
    #plt.show(False)