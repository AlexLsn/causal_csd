"""
This code is derived from the work of Boulton, C. A. and Lenton, T. M. in the paper:
A method for detecting abrupt shifts in time series, F1000Research, 2019.
https://doi.org/10.12688/f1000research.19310.1

Original code repository: https://github.com/caboulton/asdetect/
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools import add_constant
from statsmodels import robust
from itertools import groupby


def as_detect(ts, dt=None, lowwl=200, highwl='default'):
    """
    Detect gradient's anomalies in a time series using window-based linear regression.
    
    Parameters:
    - ts: array-like, time series data
    - dt: float, optional, time increment between data points
    - lowwl: int, minimum window length
    - highwl: int or 'default', maximum window length; if 'default', set to l // 3
    
    Returns:
    - numpy array or pandas DataFrame with anomaly scores
    """
    l = len(ts)  # Length of the time series

    # Set high window length if 'default' is used
    if highwl == 'default':
        highwl = l // 3

    tip_distrib = np.zeros(l)  # Array to store anomaly scores
    wls = range(lowwl, highwl + 1)  # Window length range

    for wl in wls:
        breaks = l / wl

        # Adjust for cases where the series length is not a perfect multiple of the window length
        if breaks != int(breaks):
            remainder = l - int(breaks) * wl
            ts_temp = ts[(remainder // 2):(remainder // 2 + int(breaks) * wl)]
        else:
            remainder = None
            ts_temp = ts

        breaks = int(breaks)
        lm_coeffs = np.empty((breaks, 2))  # Store linear model coefficients

        for i in range(breaks):
            y = ts_temp[(i * wl):(i * wl + wl)]
            X = add_constant(np.arange(1, wl + 1))  # Add constant term for intercept
            model = sm.OLS(y, X).fit()
            lm_coeffs[i, :] = model.params

        # Identify outliers based on linear trend slopes
        mad_slope = robust.mad(lm_coeffs[:, 1])
        outlier_ind_up = np.where((lm_coeffs[:, 1] - np.median(lm_coeffs[:, 1])) / mad_slope > 3)[0]
        outlier_ind_down = np.where((lm_coeffs[:, 1] - np.median(lm_coeffs[:, 1])) / mad_slope < -3)[0]

        # Update anomaly scores based on detected outliers
        if remainder is None:
            for k in outlier_ind_up:
                tip_distrib[(k * wl):(k * wl + wl)] += 1
            for k in outlier_ind_down:
                tip_distrib[(k * wl):(k * wl + wl)] -= 1
        else:
            for k in outlier_ind_up:
                tip_distrib[(remainder // 2 + k * wl):(remainder // 2 + (k + 1) * wl)] += 1
            for k in outlier_ind_down:
                tip_distrib[(remainder // 2 + k * wl):(remainder // 2 + (k + 1) * wl)] -= 1

    # Return results
    if dt is None:
        return tip_distrib / len(wls)
    else:
        t = np.arange(0, len(tip_distrib) * dt, dt)
        return pd.DataFrame({'t': t, 'detect': tip_distrib / len(wls)})


def where_as(ts, dt=None, thresh=0.5, method='first'):
    """
    Identify the position of abrupt shifts in the time series based on a threshold.
    
    Parameters:
    - ts: array-like, time series data
    - dt: float, optional, time increment between data points; used to convert indices to time values
    - thresh: float, threshold value to identify significant changes
    - method: str, 'first' to return the first detected shift above the threshold, 'max' to return the most significant shift
    
    Returns:
    - dict with 'as_pos': position of the detected shift (int) and 'dt_val': the corresponding value at that position (float)
      If no shift is detected (the threshold is not crossed), returns None.
    """
    # Find indices where the absolute value of the time series exceeds the threshold
    inds = np.where(abs(ts) > thresh)[0]

    if len(inds) > 0:

        if method=='first':
            return int(inds[0])
        
        if method=='max':
            # Calculate differences between consecutive indices
            diff_inds = np.diff(inds)
            
            # Group the differences and count the length of each group
            encoding = np.array([len(list(group)) for key, group in groupby(diff_inds)])
            
            # Count the number of groups that are not equal to 1
            numtip = sum(encoding != 1)
            
            # Identify which groups are not equal to 1
            whichtip = np.where(encoding != 1)[0]
            
            # Initialize arrays to store the positions and values of the shifts
            tip_pos = np.empty(numtip, dtype=int)
            tip_prob = np.empty(numtip)

            for k in range(numtip):
                if k == 0:
                    # For the first group, find the position and value of the maximum or minimum
                    tip_pos[k] = inds[np.argmax(ts[inds[:encoding[0] + 1]])] if ts[inds[0]] > 0 else inds[np.argmin(ts[inds[:encoding[0] + 1]])]
                    tip_prob[k] = ts[tip_pos[k]]
                else:
                    # For subsequent groups, find the position and value of the maximum or minimum within the group
                    temp_inds = inds[sum(encoding[:whichtip[k]]) + 1: sum(encoding[:whichtip[k] + 1]) + 1]
                    tip_pos[k] = temp_inds[np.argmax(ts[temp_inds])] if ts[temp_inds[0]] > 0 else temp_inds[np.argmin(ts[temp_inds])]
                    tip_prob[k] = ts[tip_pos[k]]

            if dt is not None:
                # If dt is provided, convert the positions to time values
                t = np.arange(0, len(ts) * dt, dt)
                tip_pos = np.round(tip_pos * dt).astype(int)

            return {'as_pos': int(tip_pos[0]), 'dt_val': tip_prob[0]}

    else:
        return None



def find_shift_in_ts(ts, lowwl=200, highwl='default', thresh=0.5, method='first'):
    """
    Detects shifts in a single time series.
    
    Parameters:
    - ts: The time series data (array/list)
    - lowwl: The lower wavelength for anomaly detection
    - highwl: The upper wavelength for anomaly detection ('default' if not specified)
    - thresh: The threshold for detecting shifts
    - method: defines if the shift is the first or the maximum value above the threshold
    
    Returns:
    - shift_pos: The position in the time series where the shift is detected
    """
    # Detect anomalies in the time series
    anomaly_scores = as_detect(ts, lowwl=lowwl, highwl=highwl)
    
    # Identify where the shift occurs based on anomaly scores
    shift_pos = where_as(anomaly_scores, thresh=thresh, method=method)

    return shift_pos


def find_shifts_multiple_ts(data_dict: dict, lowwl=200, highwl='default', thresh=0.5, method='first'):
    """
    Analyzes shifts in multiple time series from the provided data.
    
    Parameters:
    - data_dict: dict, where keys are identifiers for time series and values are the time series data (arrays/lists)
    - lowwl: The lower wavelength for anomaly detection
    - highwl: The upper wavelength for anomaly detection ('default' if not specified)
    - thresh: The threshold for detecting shifts
    
    Returns:
    - shift_positions: list of positions where shifts are detected
    """
    shift_positions = []  # List to store shift positions

    for key, ts in data_dict.items():
        # Detect shift in the current time series
        shift_pos = find_shift_in_ts(ts, lowwl=lowwl, highwl=highwl, thresh=thresh, method=method)
        
        # Store the results
        shift_positions.append(shift_pos)
    
    return shift_positions
