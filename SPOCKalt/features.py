from collections import OrderedDict
import numpy as np
from celmech.nbody_simulation_utilities import set_time_step,align_simulation
from celmech.nbody_simulation_utilities import get_simarchive_integration_results
from celmech.disturbing_function import laplace_b
import plotFunctions
import math

import firstOrder3BRfill as fff
class Trio:
    def __init__(self):
        '''initializes new set of features.
        
            each list of the key is the series of data points, second dict is for final features
        
        '''
    #innitialize running list 
        self.runningList = OrderedDict()
        self.runningList['time']=[]
        self.runningList['MEGNO']=[]
        self.runningList['threeBRfill']=[]
        for each in ['near','far','outer']:
            self.runningList['EM'+each]=[]
            self.runningList['EP'+each]=[]
            self.runningList['MMRstrength'+each]=[]
            self.runningList['twoMMRstrength'+each]=[]
            self.runningList['MMRstrengthW'+each]=[]
            self.runningList['twoMMRstrengthW'+each]=[]
        
        
        self.runningList['Prat12']=[]
        self.runningList['l1']=[]
        self.runningList['l2']=[]
        self.runningList['pomega12']=[]
        self.runningList['Prat23']=[]
        self.runningList['l3']=[]
        self.runningList['pomega23']=[]
        self.runningList['erel12']=[]
        self.runningList['erel23']=[]

    #returned features
        self.features = OrderedDict()

        for each in ['near','far','outer']:
            self.features['EMcross'+each]= np.nan
            self.features['EMfracstd'+each]= np.nan
            self.features['EPstd'+each]= np.nan
            self.features['MMRstrength'+each]= np.nan
            self.features['twoMMRstrength'+each]= np.nan
            self.features['MMRstrengthW'+each]=np.nan
            self.features['MMRstrengthWMAX'+each]=np.nan
            self.features['twoMMRstrengthW'+each]=np.nan
            self.features['twoMMRstrengthWMAX'+each]=np.nan

        self.features['MEGNO']= np.nan
        self.features['MEGNOstd']= np.nan
        self.features['threeBRfillfac']= np.nan
        self.features['threeBRfillstd']= np.nan
        self.features['chiSec'] = np.nan
        self.features['ThetaSTD'] = np.nan
        self.features['p2/1'] = np.nan
        self.features['p3/2'] = np.nan
        self.features['logInstT3BR']=np.nan
        self.features['Zval12']=np.nan
        self.features['Zcrit12']=np.nan
        self.features['Zval23']=np.nan
        self.features['Zcrit23']=np.nan
        self.features['IntZval12']=np.nan
        self.features['IntZval23']=np.nan

        
        


    def fillVal(self, Nout):
        '''fills with nan values
        
            Arguments: 
                Nout: number of datasets collected'''
        for each in self.runningList.keys():
            self.runningList[each] = [np.nan]*Nout

    def getNum(self):
        '''returns number of features collected as ran'''
        return len(self.runningList.keys())

    def populateData(self, sim, trio, pairs, minP,i):
        '''populates the runningList data dictionary for one time step.
        
            user must specify how each is calculated and added
        '''
        ps = sim.particles
        
        for q, [label, i1, i2] in enumerate(pairs):
            m1 = ps[i1].m
            m2 = ps[i2].m
            e1x, e1y = ps[i1].e*np.cos(ps[i1].pomega), ps[i1].e*np.sin(ps[i1].pomega)
            e2x, e2y = ps[i2].e*np.cos(ps[i2].pomega), ps[i2].e*np.sin(ps[i2].pomega)
            self.runningList['time'][i]= sim.t/minP
            self.runningList['EM'+label][i]= np.sqrt((e2x-e1x)**2 + (e2y-e1y)**2)
            self.runningList['EP'+label][i] = np.sqrt((m1*e1x + m2*e2x)**2 + (m1*e1y + m2*e2y)**2)/(m1+m2)
            MMRs = find_strongest_MMR(sim, i1, i2)
            self.runningList['MMRstrength'+label][i] = MMRs[2]
            self.runningList['twoMMRstrength'+label][i] = MMRs[6]
            MMRW = MMRwidth(sim, MMRs[3], i1,i2)
            self.runningList['MMRstrengthW'+label][i]=MMRW[0]
            MMRWtwo = MMRwidth(sim, MMRs[7], i1,i2)
            self.runningList['twoMMRstrengthW'+label][i]=MMRWtwo[0]
        self.runningList['threeBRfill'][i]= threeBRFillFac(sim, trio)
        self.runningList['MEGNO'][i]= sim.megno()
        

        self.runningList['Prat12'][i]= ps[1].P/ps[2].P
        self.runningList['Prat23'][i]= ps[2].P/ps[3].P
        self.runningList['l1'][i]=ps[1].l
        self.runningList['l2'][i]=ps[2].l
        self.runningList['l3'][i]=ps[3].l
        self.runningList['pomega12'][i]=getPomega(sim,1,2)
        self.runningList['pomega23'][i]=getPomega(sim,2,3)
        self.runningList['erel12'][i]=(ps[2].e*np.exp(1j*ps[2].pomega))-(ps[1].e*np.exp(1j*ps[1].pomega))
        self.runningList['erel23'][i]=(ps[3].e*np.exp(1j*ps[3].pomega))-(ps[2].e*np.exp(1j*ps[2].pomega))


    def startingFeatures(self, sim, pairs):
        '''used to initialize/add to the features that only depend on initial conditions'''
        
        #only applies to one
        ps  = sim.particles
        for [label, i1, i2] in pairs:
            self.features['EMcross'+label] =  (ps[i2].a-ps[i1].a)/ps[i1].a
        

    
        #FIXME
        #this wont work for abstracted data but will be used for testing purposes
        #from Eritas&Tamayo 2024

        #equation 11
        e12 = 1- (ps[1].a/ps[2].a)
        e13 = 1- (ps[1].a/ps[3].a)
        e23 = 1- (ps[2].a/ps[3].a)

        #23
        eta = (e12/e13)-(e23/e13)

        #equation 25

        mu = (ps[3].m-ps[1].m)/(ps[1].m+ps[3].m)
        chi23 = (1+eta)**3 *(3-eta)*(1+mu)
        chi12 = (1-eta)**3 *(3+eta)*(1-mu)
        self.features['p2/1'] = ps[2].P/ps[1].P
        self.features['p3/2'] = ps[3].P/ps[2].P

        self.features['chiSec']= chi12/(chi23+chi12)

        trio = [1,2,3]
        b0, b1,b2,b3 = ps[0], ps[trio[0]], ps[trio[1]], ps[trio[2]]
        m0,m1,m2,m3 = b0.m,b1.m,b2.m,b3.m
        ptot = None

        #semim
        a12 =(b1.a/b2.a)
        a23 = (b2.a/b3.a)

        #equation 43
        d12 = 1- a12
        d23 = 1- a23

        #equation 45
        d = (d12*d23)/(d12+d23)

        #equation 19
        mu12 = b1.P/b2.P
        mu23 = b2.P/b3.P

        #equation 21
        eta = (mu12*(1-mu23))/(1-(mu12*mu23))

        #equation 53
        eMpow2 = (m1*m3 + m2*m3*(a12**(-2))+m1*m2*(a23**2)*((1-eta)**2))/(m0**2)

        #equation 59
        dov = ((42.9025)*(eMpow2)*(eta*((1-eta)**3)))**(0.125)

        #equation 60

        ptot = (dov/d)**4
        ptot = abs(ptot)

        p1 = -np.log10((16*(2**.5)*(3.47)*math.sqrt(eMpow2*eta*(1-eta)))/3)
        p2 = np.log10((ptot**(-1.5))*(1/(1-(ptot**(-1)))))
        p3 = math.sqrt(-np.log(1-(ptot**(-1))))

        self.features['logInstT3BR']=p1+p2+p3

        #FIXME
        v12,v23,v13 = imagMaxErel(sim,trio)
        self.features['Zval12']=np.abs(v12/math.sqrt(2))
        self.features['Zcrit12']=getZcrit(sim,1,2)
        self.features['Zval23']=np.abs(v23/math.sqrt(2))
        self.features['Zcrit23']=getZcrit(sim,2,3)




    



    def fill_features(self, args):
        '''fills the final set of features that are returned to the ML model.
            
            Each feature is filled depending on some combination of runningList features and initial condition features
        '''
        Norbits = args[0]
        Nout = args[1]
        trios = args[2] #
        #print(args)

        if not np.isnan(self.runningList['MEGNO']).any(): # no nans
            self.features['MEGNO']= np.median(self.runningList['MEGNO'][-int(Nout/10):]) # smooth last 10% to remove oscillations around 2
            self.features['MEGNOstd']= np.std(self.runningList['MEGNO'][int(Nout/5):])

        self.features['threeBRfillfac']= np.median(self.runningList['threeBRfill'])
        self.features['threeBRfillstd']= np.std(self.runningList['threeBRfill'])


        for label in ['near', 'far', 'outer']: #would need to remove outer here
            self.features['MMRstrength'+label] = np.median(self.runningList['MMRstrength'+label])
            self.features['twoMMRstrength'+label]= np.median(self.runningList['twoMMRstrength'+label])
            self.features['EMfracstd'+label]= np.std(self.runningList['EM'+label])/ self.features['EMcross'+label]
            self.features['EPstd'+label]= np.std(self.runningList['EP'+label])
            self.features['MMRstrengthW'+label]=np.median(self.runningList['MMRstrengthW'+label])
            self.features['MMRstrengthWMAX'+label]=max(self.runningList['MMRstrengthW'+label])
            self.features['twoMMRstrengthW'+label]=np.median(self.runningList['twoMMRstrengthW'+label])
            self.features['twoMMRstrengthWMAX'+label]=max(self.runningList['twoMMRstrengthW'+label])

        self.features['ThetaSTD'] = min([self.getThetaSTD('1','2',Nout),self.getThetaSTD('2','3',Nout)])
        self.features['IntZval12']= np.abs(max(self.runningList['erel12'])/math.sqrt(2))
        self.features['IntZval23']= np.abs(max(self.runningList['erel23'])/math.sqrt(2))


        
            


    def getThetaSTD(self,b1,b2,Nout):

        try:
            OrderL=[1,2,3,4,5]
            ratList = getRatL(np.median(self.runningList['Prat'+b1+b2]),OrderL)
            thetalist = [[np.nan]*Nout]*len(ratList)
            
            for i,r in enumerate(ratList):
                for x in range(Nout):
                    thetalist[i][x]=calcTheta(self.runningList['l'+b1][x],self.runningList['l'+b2][x],self.runningList['pomega'+b1+b2][x],r)
                thetalist[i]= np.unwrap(thetalist[i])
            
            stds = list(map(np.std, thetalist))
            which = stds.index(min(stds))
            
            

            return  min(stds)
        
            #self.features['nearThetaSTD'] = np.std(thetalist2)
        except:
            return np.nan

# Pratio23 = 1/np.median(p3p2)
#     OrderL=[1,2,3,4,5]
#     rat12 = getRatL(Pratio12,OrderL)
#     thetalist12 = [[np.nan]*Nout]*len(rat12)
#     #print(len(thetalist12[0]))
#     for i,r in enumerate(rat12):
#         for x in range(Nout):
#                 #
#                 # print(r)
#                 thetalist12[i][x]=plotFunctions.calcTheta(l1[x],l2[x],pomegarel12[x],r)
#         #print(thetalist12[i])
#         thetalist12[i]= np.unwrap(thetalist12[i])
    
#     stds12 = list(map(np.std,thetalist12))
#     which12 = stds12.index(min(stds12))
#     theta12 = thetalist12[which12]
#     pval12 = rat12[which12]

class initialTrio:
    def __init__(self):
        self.features = OrderedDict()

        for each in ['near','far','outer']:
            self.features['EMcross'+each]= np.nan
            self.features['MMRstrength'+each]= np.nan
            self.features['twoMMRstrength'+each]= np.nan
            self.features['MMRstrengthW'+each]=np.nan
            #self.features['MMRstrengthWMAX'+each]=np.nan
            
            self.features['twoMMRstrengthW'+each]=np.nan
            
            #self.features['twoMMRstrengthWMAX'+each]=np.nan
            
            

        self.features['MEGNO']= np.nan
        self.features['3BRfirstfillfac']= np.nan
        self.features['threeBRfillfac']= np.nan
        self.features['chiSec'] = np.nan
        self.features['eccMag'] = np.nan
        self.features['eccDir'] = np.nan
        self.features['pomegastd'] = np.nan
        self.features['p2/1'] = np.nan
        self.features['p3/2'] = np.nan



        self.features['Zval12']=np.nan
        self.features['Zcrit12']=np.nan
        self.features['Zval23']=np.nan
        self.features['Zcrit23']=np.nan

        for x in range(1,4):
            self.features['hillRad'+str(x)] = np.nan
            self.features['e'+str(x)] = np.nan
            self.features['a'+str(x)] = np.nan
            self.features['P'+str(x)] = np.nan
            self.features['pomega'+str(x)] = np.nan
            self.features['Omega'+str(x)] = np.nan
            self.features['w'+str(x)] = np.nan
            self.features['m'+str(x)] = np.nan









    def initial_features(self,sim, pairs, trio):
        '''used to initialize/add to the features that only depend on initial conditions'''
        
        #only applies to one
        ps  = sim.particles
        MMRWdata = []
        for [label, i1, i2] in pairs:
            self.features['EMcross'+label] =  (ps[i2].a-ps[i1].a)/ps[i1].a
            m1 = ps[i1].m
            m2 = ps[i2].m
            e1x, e1y = ps[i1].e*np.cos(ps[i1].pomega), ps[i1].e*np.sin(ps[i1].pomega)
            e2x, e2y = ps[i2].e*np.cos(ps[i2].pomega), ps[i2].e*np.sin(ps[i2].pomega)
            self.features['EM'+label]= np.sqrt((e2x-e1x)**2 + (e2y-e1y)**2)
            self.features['EP'+label] = np.sqrt((m1*e1x + m2*e2x)**2 + (m1*e1y + m2*e2y)**2)/(m1+m2)
            MMRs = find_strongest_MMR(sim, i1, i2)
            self.features['MMRstrength'+label] = MMRs[2]
            self.features['twoMMRstrength'+label] = MMRs[6]
            MMRWdata.append([i1,i2,MMRs[3],'',label])
            MMRWdata.append([i1,i2,MMRs[7],'two',label])

        fillMaxWidth(self,sim,trio,MMRWdata)
        
        self.features['threeBRfillfac']= threeBRFillFac(sim, trio)
        self.features['MEGNO']= sim.megno() 

        self.features['3BRfirstfillfac']= fff.getfirst3brfill(sim,trio)


        
         #FIXME

        self.features['pomegastd']= np.std([ps[1].pomega, ps[2].pomega,ps[3].pomega])
        #this wont work for abstracted data but will be used for testing purposes
        #from Eritas&Tamayo 2024

        #equation 11
        e12 = 1- (ps[1].a/ps[2].a)
        e13 = 1- (ps[1].a/ps[3].a)
        e23 = 1- (ps[2].a/ps[3].a)

        #23
        eta = (e12/e13)-(e23/e13)

        #equation 25

        mu = (ps[3].m-ps[1].m)/(ps[1].m+ps[3].m)
        chi23 = (1+eta)**3 *(3-eta)*(1+mu)
        chi12 = (1-eta)**3 *(3+eta)*(1-mu)

        self.features['chiSec']= chi12/(chi23+chi12)
        e1x, e1y = ps[1].e*np.cos(ps[1].pomega), ps[1].e*np.sin(ps[1].pomega)
        e2x, e2y = ps[2].e*np.cos(ps[2].pomega), ps[2].e*np.sin(ps[2].pomega)
        e3x, e3y = ps[3].e*np.cos(ps[3].pomega), ps[3].e*np.sin(ps[3].pomega)
        ex = (e1x*ps[1].m+e2x*ps[2].m+e3x*ps[3].m)/(ps[1].m+ps[2].m+ps[3].m)
        ey = (e1y*ps[1].m+e2y*ps[2].m+e3y*ps[3].m)/(ps[1].m+ps[2].m+ps[3].m)
        self.features['eccMag']=(ex**2+ey**2)**(.5)
        self.features['eccDir']=np.arctan(ey/ex)

        self.features['p2/1'] = ps[2].P/ps[1].P
        self.features['p3/2'] = ps[3].P/ps[2].P


        for x in trio:
            self.features['hillRad'+str(x)] = ps[x].rhill
            self.features['e'+str(x)] = ps[x].e
            self.features['a'+str(x)] = ps[x].a
            self.features['P'+str(x)] = ps[x].P
            self.features['pomega'+str(x)] = ps[x].pomega
            self.features['Omega'+str(x)] = ps[x].Omega
            self.features['w'+str(x)] = ps[x].omega
            self.features['m'+str(x)] = ps[x].m
#FIXME
        v12,v23,v13 = imagMaxErel(sim,trio)
        self.features['Zval12']=np.abs(v12/math.sqrt(2))
        self.features['Zcrit12']=getZcrit(sim,1,2)
        self.features['Zval23']=np.abs(v23/math.sqrt(2))
        self.features['Zcrit23']=getZcrit(sim,2,3)
        



def getZcrit(sim, i1, i2):
    ps = sim.particles
    p1 = ps[i1]
    p2 = ps[i2]
    t1 = ((p2.a-p1.a)/p2.a)/math.sqrt(2)
    m1 = p1.m/ps[0].m
    m2 = p2.m/ps[0].m
    exp = -2.2*((m1+m2)**(1/3))*((p2.a/(p2.a-p1.a))**(4/3))

    return t1**exp


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
    for eachO in resList:
        if len(eachO) !=0:
            val = [10000000,10]
            for i,each in enumerate(eachO):
                if np.abs((each[0]/each[1])-Pratio)<np.abs((val[0]/val[1])-Pratio):
                    val = each
            finals.append(val)
    return finals

#if len(eachO) !=0:
            # val = [10000000,10]
            # for i,each in enumerate(eachO):
            #     if np.abs((each[0]/each[1])-Pratio)<np.abs((val[0]/val[1])-Pratio):
            #         val = each
            # finals.append(val)





def calcTheta(la,lb,pomegarel, val):
    theta = (val[1]*lb) -(val[0]*la)-(val[1]-val[0])*pomegarel
    return np.mod(theta, 2*np.pi)

def getPomega(sim, i1, i2):
    ps = sim.particles
    evec2 = ps[i2].e*np.exp(1j*ps[i2].pomega)
    evec1 = ps[i1].e*np.exp(1j*ps[i1].pomega)
    erel = evec2-evec1
    pomegarel=np.angle(erel)
    return pomegarel


def fillMaxWidth(obj:initialTrio,sim, trio,data):
    e12,e23,e13 = maxErel(sim,trio)
    relecc = [np.nan, e12, e23, e13]
    Ak = [0., 0.84427598, 0.75399036, 0.74834029, 0.77849985, 0.83161366] #from tamayo&hadden 2024
    ps = sim.particles
    for each in data:
        i1,i2,Prat,pre,label = each
        if Prat is np.nan or type(Prat) == float or np.nan in Prat:
            obj.features[pre+'MMRstrengthW'+label]= scaleWidth
        else:

            erel = relecc[trio.index(i1)+trio.index(i2)]
            a,b=Prat
            ec = (ps[i2].a-ps[i1].a)/ps[i2].a #crossing excentricity
            mu = (ps[i1].m+ps[i2].m)/ps[0].m #planet mass star mass ratio
            order = b-a
            etilde = np.linalg.norm(erel)/ec
            dP = 3*Ak[order]*np.sqrt(mu * (etilde**order))
            scaleWidth = dP/(a/b)
            obj.features[pre+'MMRstrengthW'+label]= scaleWidth




def getEigMode(sim, trio):
    '''returns the three eigen modes for a three body system when given trion.
        
        return: ecom, e13, emin, [chi12t,chi23t]
    '''
    #FIXME
    [i1,i2,i3] = trio
    ps = sim.particles
    p1, p2, p3 = ps[i1], ps[i2], ps[i3]

    ecom, e13, emin = [np.nan, np.nan],[np.nan, np.nan],[np.nan, np.nan]

    m1, m2, m3 = p1.m, p2.m, p3.m
    m_tot = m1 + m2 + m3
    mu1, mu2, mu3 = m1/m_tot, m2/m_tot, m3/m_tot
    
    #alpha is semi major axis ratio
    alpha12, alpha23 = p1.a/p2.a, p2.a/p3.a
    alpha13 = alpha12*alpha23
    #crossing ecc in Appendix A
    ec12 = alpha12**(-1/4)*alpha23**(3/4)*alpha23**(-1/8)*(1-alpha12)
    ec23 = alpha23**(-1/2)*alpha12**(1/8)*(1-alpha23)
    ec13 = alpha13**(-1/2)*(1-alpha13)
    #made up constant
    eta = (ec12 - ec23)/ec13
    chi12 = mu1*(1-eta)**3*(3+eta)
    chi23 = mu3*(1+eta)**3*(3-eta)
    chi12t = chi12/(chi12+chi23)
    chi23t = chi23/(chi12+chi23)

    
    #rel ecc vec
    e1x, e2x, e3x = [p.e*np.cos(p.pomega) for p in [p1,p2,p3]]
    e1y, e2y, e3y = [p.e*np.sin(p.pomega) for p in [p1,p2,p3]]

    emin = np.array([chi23t*(e3x-e2x)-chi12t*(e2x-e1x), chi23t*(e3y-e2y)-chi12t*(e2y-e1y)])
    e13 = np.array([e3x-e1x, e3y-e1y])
    ecom = np.array([(mu1*e1x+mu2*e2x+mu3*e3x), (mu1*e1y+mu2*e2y+mu3*e3y)])

    return ecom, e13, emin, [chi12t,chi23t]



def maxErel(sim, trio):
    '''given a sim and trio, returns the abs of maximum excentricity vector for each body, note, using 2nd body is massless assumption'''

    ecom, e13, emin, [chi12t, chi23t] = getEigMode(sim, trio)

    absEcom = np.sqrt(np.sum(ecom**2))
    absE13 = np.sqrt(np.sum(e13**2))
    absEmin = np.sqrt(np.sum(emin**2))

    e12M = chi23t*absE13 + absEmin
    e23M = chi12t*absE13 + absEmin

    return e12M, e23M, absE13

def imagMaxErel(sim, trio):
    ecom, e13, emin, [chi12t, chi23t] = getEigMode(sim, trio)

    v13 = complex(e13[0],e13[1])
    vm = complex(emin[0],emin[1])
    v12 = chi23t*v13-vm
    v23 = chi12t*v13+vm
    return v12,v23,v13
# def first3BRfill(sim,trio):
    


def MMRwidth(sim,Prat, i1,i2):
    '''calculates the MMR width per tamayo&hadden 2024 equation 19
    
    returns dP/P'''
    if Prat is np.nan or type(Prat) == float or np.nan in Prat:
        #print(Prat)
        return 0, 0
    

    ps = sim.particles

    Ak = [0., 0.84427598, 0.75399036, 0.74834029, 0.77849985, 0.83161366] #from tamayo&hadden 2024
    a,b=Prat

    ec = (ps[i2].a-ps[i1].a)/ps[i2].a #crossing excentricity

    mu = (ps[i1].m+ps[i2].m)/ps[0].m #planet mass star mass ratio

    order = b-a

    erel = [ps[i2].e*np.cos(ps[i2].pomega)-ps[i1].e*np.cos(ps[i1].pomega),ps[i2].e*np.sin(ps[i2].pomega)-ps[i1].e*np.sin(ps[i1].pomega)]

    etilde = np.linalg.norm(erel)/ec
#mag
    dP = 3*Ak[order]*np.sqrt(mu * (etilde**order))

    realPrat = ps[i1].P/ps[i2].P
    inwidth = dP< abs(realPrat-(a/b))
    scaleWidth = dP/(a/b)
    

    return scaleWidth, inwidth


def threeBRFillFac(sim, trio):
    '''calculates the 3BR filling factor in acordance to petit20'''
    ps = sim.particles
    b0, b1,b2,b3 = ps[0], ps[trio[0]], ps[trio[1]], ps[trio[2]]
    m0,m1,m2,m3 = b0.m,b1.m,b2.m,b3.m
    ptot = None

    #semim
    a12 =(b1.a/b2.a)
    a23 = (b2.a/b3.a)

    #equation 43
    d12 = 1- a12
    d23 = 1- a23

    #equation 45
    d = (d12*d23)/(d12+d23)

    #equation 19
    mu12 = b1.P/b2.P
    mu23 = b2.P/b3.P

    #equation 21
    eta = (mu12*(1-mu23))/(1-(mu12*mu23))

    #equation 53
    eMpow2 = (m1*m3 + m2*m3*(a12**(-2))+m1*m2*(a23**2)*((1-eta)**2))/(m0**2)

    #equation 59
    dov = ((42.9025)*(eMpow2)*(eta*((1-eta)**3)))**(0.125)

    #equation 60

    ptot = (dov/d)**4

    return abs(ptot)






    ######################### Taken from celmech github.com/shadden/celmech

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
##########################

# sorts out which pair of planets has a smaller EMcross, labels that pair inner, other adjacent pair outer
# returns a list of two lists, with [label (near or far), i1, i2], where i1 and i2 are the indices, with i1 
# having the smaller semimajor axis

#taken from original spock
####################################################
def find_strongest_MMR(sim, i1, i2):
    #originally 2, trying with 5th order now
    maxorder = 5
    ps = sim.particles
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

    # Calculating EM exactly would have to be done in celmech for each j/k res below, and would slow things down. This is good enough for approx expression
    EM = np.sqrt((ps[i1].e*np.cos(ps[i1].pomega) - ps[i2].e*np.cos(ps[i2].pomega))**2 + (ps[i1].e*np.sin(ps[i1].pomega) - ps[i2].e*np.sin(ps[i2].pomega))**2)
    EMcross = (ps[i2].a-ps[i1].a)/ps[i1].a

    j, k, maxstrength, res1 = np.nan, np.nan, 0, np.nan 
    j2, k2, maxstrength2, res2 = np.nan, np.nan, 0, np.nan 
    
    for a, b in res:
        nres = (b*n2 - a*n1)/n1
        if nres == 0:
            s = np.inf # still want to identify as strongest MMR if initial condition is exatly b*n2-a*n1 = 0
        else:
            s = np.abs(np.sqrt(m1+m2)*(EM/EMcross)**((b-a)/2.)/nres)
        
        if s > maxstrength2 and not np.isnan(s) :
            j2 = b
            k2 = b-a
            maxstrength2 = s
            res2 = [a,b]
            if maxstrength2> maxstrength:
                j,j2 = swap(j,j2)
                k,k2 = swap(k,k2)
                res1, res2 = swap(res1,res2)
                maxstrength, maxstrength2 = swap(maxstrength, maxstrength2)
    

    # if maxstrength == 0:
    #     maxstrength = np.nan
    # if maxstrength2 == 0:
    #     maxstrength2 = np.nan

    return j, k, maxstrength, res1, j2, k2, maxstrength2, res2, res
#############################################

def swap(a,b):
    return b,a