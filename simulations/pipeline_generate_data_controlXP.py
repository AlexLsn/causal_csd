import sys
import numpy as np


import functions.generate_data as gendata 
import functions.shift_detection as shiftdetect



t_len=10000
deterministic_ts = gendata.generate_time_series(t_len=t_len, model='base', sigma_Y=0, return_noise=True, bif_type='saddle-node', seed=0)
deterministic_bif_time = shiftdetect.find_shift_in_ts(deterministic_ts['data'][:, 1], lowwl=100, highwl = 3000, thresh=0.8, method='first')

sigma_Y = 0.01
    
# This function contains the actual stuff that is computed on each single CPU
def calculate(model, X_power, realization):


    for i in range(100):
        # We create a dataframe with 'confounderIncreasAC' but as sigma_X=0, X has no influence on Y. So we are in the base model but with the same data structure as when X is computed
        data_dict = gendata.generate_ts_with_detected_shift(t_len=t_len, model='confounderIncreasAC', sigma_Y=sigma_Y, sigma_X=0, threshold=0.8, bif_shift=200, return_noise=True, bif_type='saddle-node', X_power=X_power, seed=realization*100 + i)
        # Only X
        dataX = gendata.generate_time_series(t_len=t_len, sigma_Y = 0.0, sigma_X=0.01, seed = realization*50 + i, model = model, bif_type='saddle-node', return_noise=True, X_power = X_power)['data'][:, 1]
        # We replace the X in data_dict with the independent X
        data_dict['data'][:,1] = dataX

        if np.isnan(data_dict['shift_position']):
            pass
        else:
            det_data = data_dict['data'][:, -1] - deterministic_ts['data'][:, -1]
            det_trunc_data = det_data[:min(data_dict['shift_position'], deterministic_bif_time)]          
            break
    return {
            'data': data_dict,                        
            'dataY_det_trunc': det_trunc_data,
            }