#%%


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from xgboost.sklearn import XGBRegressor
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import hyperopt
import sys
sys.path.append('../spock/')
#from simsetup import get_sim
#from modelfitting import ROC_curve, stable_unstable_hist, calibration_plot, unstable_error_fraction
try:
    plt.style.use('paper')
except:
    pass

import rebound
import numpy as np
import pandas as pd
import warnings
from multiprocessing import Process
import sys
from collections import OrderedDict
from multiprocessing import Pool
sys.path.insert(1, '..')
#print(path)
from SPOCKalt import *
sys.path.insert(1, '../SPOCKalt')
#Intigration/simsetup.py
from SPOCKalt import featureKlassifier
from SPOCKalt import simsetup
#%%
dataset = pd.read_csv('../modeldata/zfixed3brfill.csv')
initial = pd.read_csv('../modeldata/originalCondAllData.csv')
#%%
plot = dataset
plot['dup']=plot[['threeBRfillfac','instability_time','EMcrossnear']].duplicated()
plot = plot.drop(plot[plot['dup']==True].index)

plot = plot.drop(plot[plot['threeBRfillfac']<13].index)
plot = plot.drop(plot[plot['index']==113762].index)

one = plot.drop(plot[plot['instability_time'] <1e9].index)
two = plot.drop(plot[plot['instability_time'] >1e5].index)
two = plot.drop(plot[plot['instability_time'] <5e3].index)
plot = pd.concat([one.sample(n=10),two.sample(n=10)])


# plot = plot.drop(plot[plot['Stable']==False].index)
# plot = plot.drop(plot[plot['index']==113762].index)


#%%
#plot

#%%
import plotFunctions
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
def get_plot(num,Nout=5000,Nint=100000):
    sim = simsetup.get_simList(initial.iloc[num,2:])
    simsetup.init_sim_parameters(sim)
    figure = plt.figure(figsize=[20,35])
    gs = GridSpec(4, 2, figure=figure)
    #gs.update(wspace = .1, hspace = .1)
    
    data, res12, res23 = plotFunctions.get_data(sim,Nout,Nint)
    ax1 = plt.subplot(gs[0,0])
    ax1.set_title('Stable:' +str(dataset['Stable'][num]))
    data.plot.scatter(ax = ax1,x="p2/p1", y="p3/p2",s=2, c="time", colormap="copper", alpha=.35)
    ax2 = plt.subplot(gs[1,:2])
    ax2.set_title(str(res12[1])+':'+str(res12[0]))
    data.plot.scatter(ax=ax2,x="time", y="theta12",s=1)
    ax3 = plt.subplot(gs[2,:2])
    ax3.set_title(str(res23[1])+':'+str(res23[0]))
    data.plot.scatter(ax = ax3,x="time", y="theta23",s=1)
    ax4 = plt.subplot(gs[3,:2])
    data.plot(ax=ax4,x='time',y=['e1','e2','e3'])
    ax5 = plt.subplot(gs[0,1])
    ax5.set_title(str(num))
    ax5.set_aspect('equal')
    rebound.OrbitPlot(sim,fig=figure, ax=ax5,ylim=[-3,3],xlim=[-3,3])
    plt.savefig(f'imgs/'+str(dataset['threeBRfillfac'][num])+'.png')
    #plt.show(False)

    #return figure
from multiprocessing import Pool

#%%

systems = plot['index']




if __name__ == "__main__":  # confirms that the code is under main function

  

    
    #bound = test = np.linspace(0, 138543, num=138544, endpoint=True, retstep=False, dtype=int, axis=0)
    with Pool() as p:
        p.map(get_plot, systems)
    # test = list(map(runInt, bound))
    # print(test)


#plot_data(sim,5000,100000, str(2))
# %%
