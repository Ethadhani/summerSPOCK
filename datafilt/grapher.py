#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib_inline
if __name__ == "__main__":  # confirms that the code is under main function

    dataset = pd.read_csv('../modeldata/zfixed3brfill.csv')


    def hasnull(row):
        numnulls = row.isnull().sum()
        if numnulls == 0:
            return 0
        else:
            return 1

    def tmax(row):
        #sim = get_sim(row, csvfolder)
        tmax = 1e4 # replace with a calculation of tmax
        return tmax


    if 'hasnull' not in dataset.columns:
        dataset['hasnull'] = dataset.apply(hasnull, axis=1)
        #dataset['tmax'] = dataset.apply(tmax, axis=1)
        # dataset['tmax'] = dataset.apply(lambda x:1e4, axis=1) # this version would just set tmax=1e4 for all of them

        #dataset.to_csv(trainingdatafolder+"trainingdata.csv", encoding='ascii')

    mask = (dataset['hasnull'] == 0 )
    filtData = dataset[mask]

    import plotFunctions

    from multiprocessing import Pool


    def getOrder(Pratio):
        val = plotFunctions.getval(Pratio)
        return val[1]-val[0]


    temp = 1/filtData['p3/2']
    print('start')
    
    #bound = test = np.linspace(0, 138543, num=138544, endpoint=True, retstep=False, dtype=int, axis=0)
    with Pool() as p:
        new = list(p.map(getOrder, temp))

#%%