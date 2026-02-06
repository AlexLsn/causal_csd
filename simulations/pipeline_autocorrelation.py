import os
import sys

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pickle 

import functions.significance_testing as sgtest
import functions.indicators_computation as ind
import functions.sliding_windows as sw


    
def calculate(model, X_function, method, detrend, realization):

    window_length = 1000
    t_len=10000
    time_windows, _ = sw.get_centered_sliding_windows(window_length, 1, t_len)

    # Path to data files (relative to this script)
    mypath = os.path.join(os.path.dirname(__file__), 'generated_data')
    
    data_file = os.path.join(mypath, '%s-%s.dat' % (model, X_function))

    with open(data_file, 'rb') as file:
        all_data = pickle.load(file)
    
    dataY_det_trunc = all_data['dataY_det_trunc'][realization]

    #############################################
    ##  Apply method extracted from config string
    #############################################

    indicator = ind.regular_autocorrelation(dataY_det_trunc, time_windows, lag=1, detrend=detrend)

    #############################################
    ##  Derive pvalues for each indicator
    #############################################

    slope, _ = np.polyfit([i for i in range(len(indicator))], indicator, 1)

    if method=='rosa':
        pvalue, _ = sgtest.p_value_from_fourier(indicator, ns=1000, tau=-slope)
    else:
        pvalue, _ = sgtest.p_value_from_fourier(indicator, ns=1000, tau=slope)


    return {
            'indicator': indicator,
            'pvalue': pvalue,
            }