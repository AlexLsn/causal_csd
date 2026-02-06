import os
import sys

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from scipy.ndimage import gaussian_filter1d

import functions.generate_data as gendata 
import functions.shift_detection as shiftdetect



t_len=10000
deterministic_ts = gendata.generate_time_series(t_len=t_len, model='base', sigma_X=0, sigma_Y=0, return_noise=True, bif_type='saddle-node', seed=0)
deterministic_bif_time = shiftdetect.find_shift_in_ts(deterministic_ts['data'][:, 1], lowwl=100, highwl = 3000, thresh=0.8, method='first')
deterministic_flatControl_ts = gendata.generate_time_series(t_len, sigma_Y = 0.0, seed = 10, model = 'flatControl', return_noise=True, bif_type='saddle-node')

sigma_Y = 0.01
    
def calculate(model, X_power, realization):

    for i in range(100):

        if model=='falseAlarm':
            sigma_X = 1
            gamma = 1
        elif model=='confounderDecreasAC' and X_power==2:
            sigma_X = 0.5
            gamma = 1
        else:
            sigma_X = 0.01
            gamma = 1
            
        data_dict = gendata.generate_ts_with_detected_shift(t_len=t_len, model=model, sigma_X=sigma_X, sigma_Y=sigma_Y, threshold=0.8, bif_shift=200, return_noise=True, bif_type='saddle-node', X_power=X_power, gamma=gamma, seed=realization*100 + i)

        if model in ('falseAlarm', 'flatControl'):
            det_data = data_dict['data'][:, -1] - deterministic_flatControl_ts['data'][:, -1]
            det_trunc_data = det_data[:deterministic_bif_time]
            det_trunc_data = det_trunc_data - gaussian_filter1d(det_trunc_data, sigma=30)
            return {
                'data': data_dict,                        
                'dataY_det_trunc': det_trunc_data,
                }
        
        else:
            if np.isnan(data_dict['shift_position']):
                pass
            else:
                det_data = data_dict['data'][:, -1] - deterministic_ts['data'][:, -1]
                det_trunc_data = det_data[:min(data_dict['shift_position'], deterministic_bif_time)]
                det_trunc_data = det_trunc_data - gaussian_filter1d(det_trunc_data, sigma=100)
                return {
                        'data': data_dict,                        
                        'dataY_det_trunc': det_trunc_data,
                        }