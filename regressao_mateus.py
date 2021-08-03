#import missingno as msno
#import pandas_profiling

from utils import *

import numpy as np
import pylab as plt
import seaborn as sns
import glob
import pandas as pd
pd.set_option('display.max_columns', 500)
import gc
import warnings
warnings.filterwarnings("ignore")

def ks(data=None,target=None, prob=None):
    data['target0'] = 1 - data[target]
    data['bucket'] = pd.qcut(data[prob], 10)
    grouped = data.groupby('bucket', as_index = False)
    kstable = pd.DataFrame()
    kstable['min_prob'] = grouped.min()[prob]
    kstable['max_prob'] = grouped.max()[prob]
    kstable['events']   = grouped.sum()[target]
    kstable['nonevents'] = grouped.sum()['target0']
    kstable = kstable.sort_values(by="min_prob", ascending=False).reset_index(drop = True)
    kstable['event_rate'] = (kstable.events / data[target].sum()).apply('{0:.2%}'.format)
    kstable['nonevent_rate'] = (kstable.nonevents / data['target0'].sum()).apply('{0:.2%}'.format)
    kstable['cum_eventrate']=(kstable.events / data[target].sum()).cumsum()
    kstable['cum_noneventrate']=(kstable.nonevents / data['target0'].sum()).cumsum()
    kstable['KS'] = np.round(kstable['cum_eventrate']-kstable['cum_noneventrate'], 3) #* 100

    #Formating
    kstable['cum_eventrate']= kstable['cum_eventrate']#.apply('{0:.2%}'.format)
    kstable['cum_noneventrate']= kstable['cum_noneventrate']#.apply('{0:.2%}'.format)
    kstable.index = range(1,11)
    kstable.index.rename('Decile', inplace=True)
    #pd.set_option('display.max_columns', 9)
    #print(kstable)
    
    #Display KS
    from colorama import Fore
    print(Fore.RED + "KS is " + str(max(kstable['KS']))+"%"+ " at decile " + str((kstable.index[kstable['KS']==max(kstable['KS'])][0])))
    return(kstable)

def adjust_colum(df,cols_to_move,new_index):
        """
        This method re-arranges the columns in a dataframe to place the desired columns at the desired index.
        ex Usage: df = move_columns(df, ['Rev'], 2)   
        :param df:
        :param cols_to_move: The names of the columns to move. They must be a list
        :param new_index: The 0-based location to place the columns.
        :return: Return a dataframe with the columns re-arranged
        """    
        other = [c for c in df if c not in cols_to_move]
        start = other[0:new_index]
        end = other[new_index:]
        return df[start + cols_to_move + end].copy()


#input_dir = '../raw_data_enade/input/'

data_list = np.sort(np.array(glob.glob('*.csv')))#
#data_list

data = pd.read_csv(data_list[0],sep=',')

print(data.head())

print('Não alvo', round(data['PERMANENCIA_PROLONGADA'].value_counts()[0]/len(data) * 100,2), '% of the dataset')
print('Alvo', round(data['PERMANENCIA_PROLONGADA'].value_counts()[1]/len(data) * 100,2), '% of the dataset')