import random
import numpy as np
import pandas as pd
import scipy
import scipy.stats
from datetime import datetime, timedelta



def compute_gini(y):

    '''
    Compute the Gini index for the throughput of a station. 

    :param y: the time series data for a station
    :return: Gini index for a station
    '''

    num = 0
    den = 0
    for i in range(len(y)):
        for j in range(len(y)):
            num += np.abs(y[i]-y[j])
            den += y[j]
    den *= 2
    gini_stn = num/den

    return gini_stn


def compute_gini_ci(y, t, n_ite):
    
    '''
    Compute the 95% confidence interval for the Gini index of a station based on bootstrap sampling. 

    :param y: the time series data for a station
    :param t: the time slots
    :param n_ite: number of iterations for random simulation
    :return: 95% confidence interval for the Gini index of a station
    '''
        
    gini_bootstrap = [0 for i in range(n_ite)]
    
    for i in range(len(gini_bootstrap)):
        indexes = random.choices(t, k=int(len(t)))
        sample_y = [y[int(index-1)] for index in indexes]
        gini_bootstrap[i] = compute_gini(sample_y)
    
    confidence=0.95
    
    return np.percentile(gini_bootstrap,[100*(1-confidence)/2,100*(1-(1-confidence)/2)])
    

def compute_gini_pvalue(gini_ci_stn, throughput_stn, t, n_ite):
    
    '''
    Compute the p-value for the Gini index of a station based on bootstrap sampling. 

    :param gini_ci_stn: the 95% confidence interval for the Gini index of the station
    :param throughput_stn: the throughput of the station
    :param t: the time slots
    :param n_ite: number of iterations for random simulation
    :return: p-value of the Gini index
    '''
    
    pvalue = 0
    
    for i in range(n_ite):

        # Initialize a list to hold the counts for each class
        y_random = [0 for i in range(len(t))]

        # Distribute the remaining counts randomly
        for j in range(throughput_stn):
            index = random.randint(0, len(t)-1)
            y_random[index] += 1

        gini_random_stn = compute_gini(y_random)
        
        if gini_random_stn >= gini_ci_stn[0] and gini_random_stn <= gini_ci_stn[1]:
            pvalue += 1
            
    pvalue /= n_ite

    return pvalue