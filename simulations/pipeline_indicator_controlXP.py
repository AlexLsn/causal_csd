import os, sys

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pickle 

from tigramite.causal_effects import CausalEffects
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler 

import functions.generate_data as gendata 
import functions.significance_testing as sgtest
import functions.indicators_computation as ind
import functions.sliding_windows as sw


data_transform = StandardScaler()

# This function contains the actual stuff that is computed on each single CPU
def calculate(model, X_function, estimator, detrend, realization):


    t_len=10000
    window_length=1000
    time_windows, _ = sw.get_centered_sliding_windows(window_length, 1, t_len)

    # Path to data files
    mypath = '/data/horse/ws/alla605g-causal_csd_ws/results_causalCSD/saddle-node/generated_data/controlXP/'
    
    data_file = mypath + '%s-%s-%s.dat' % (model, X_function)

    with open(data_file, 'rb') as file:
        all_data = pickle.load(file)

    
    # Extraire le dataset spécifique correspondant à la réalisation actuelle
    data_dict = all_data['data'][realization]

    dataY_det_trunc = all_data['dataY_det_trunc'][realization]

    bif_time = min(len(dataY_det_trunc), data_dict['shift_position'])
    truncated_data_dict = gendata.truncate_data_before_bif(data_dict, bif_time=bif_time)
    truncated_data_dict['truncated_data'][:, -1] = dataY_det_trunc[:bif_time]


    if estimator=='LinearRegression':
        estimator_class = LinearRegression()
    if estimator=='KNeighborsRegressor':
        estimator_class = KNeighborsRegressor(n_neighbors=5)



    graph =  np.array([[['', '-->'], ['', ''], ['', '']],
                [['', ''], ['', '-->'], ['', '-->']],
                [['', ''], ['', ''], ['', '-->']]], dtype='<U3')

    X = [(2,-1)]
    Y = [(2,0)]

    causal_effects = CausalEffects(graph, graph_type='stationary_dag', X=X, Y=Y, S=None, 
                                hidden_variables=None, verbosity=0)
    

    data = truncated_data_dict['truncated_data']


    indicator, residuals, _ = ind.pipeline_causalEE_weighted(causal_effects, data, estimator_class, data_transform, n_points=150, time_windows=time_windows, var_names=[r'$r$', r'$X$', r'$Y$'], detrend=detrend)



    slope, _ = np.polyfit([i for i in range(len(indicator))], indicator, 1)
    pvalue, _ = sgtest.p_value_from_fourier(indicator, ns=1000, tau=slope)



    return {
            'data': data_dict,                        
            'truncated_data': truncated_data_dict,
            'indicator': indicator,
            'residuals': residuals,
            'pvalue': pvalue,
            }