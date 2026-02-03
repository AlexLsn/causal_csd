import numpy as np

from . import shift_detection as shiftdetect



def generate_ts_with_detected_shift(t_len: int, model:str, sigma_Y, sigma_X, threshold:float, bif_shift:int, return_noise:bool=False, bif_type:str='saddle-node', X_power=1, gamma=1, seed=1):
    """
    Generate a time series with a detected shift based on the specified parameters.

    Parameters:
        t_len (int): Length of the time series to generate.
        model (str): The model type to use for generating the time series. 
                        If 'flatControl' or 'falseAlarm', no shift detection is performed.
        sigma_Y: Standard deviation for the Y variable in the time series.
        sigma_X: Standard deviation for the X variable in the time series.
        threshold (float): Threshold value for detecting a shift in the time series.
        bif_shift (int): Adjustment value for the detected shift position.
        return_noise (bool, optional): Whether to return noise data in the generated time series. Defaults to False.
        bif_type (str, optional): Type of bifurcation to use ('saddle-node' by default). Defaults to 'saddle-node'.
        X_power (int, optional): Power parameter for the X variable. Defaults to 1.
        gamma (float, optional): Gamma parameter for the scaling of X influence in the time series. Defaults to 1.
        seed (int, optional): Random seed for reproducibility. Defaults to 1.

    Returns:
        dict: A dictionary containing the generated time series data and the detected shift position.
                - 'data': The generated time series data.
                - 'shift_position': The position of the detected shift, or NaN if no shift is detected.
    """

    data = generate_time_series(t_len=t_len, sigma_Y=sigma_Y, sigma_X=sigma_X, seed=seed, model=model, return_noise=return_noise, bif_type=bif_type, X_power=X_power, gamma=gamma)

    if model not in ('falseAlarm', 'flatControl'):
        data_Y = data['data'][:, -1]
        shift_position = shiftdetect.find_shift_in_ts(data_Y, lowwl=100, highwl=3000, thresh=threshold, method='first')

        if shift_position is not None and shift_position >= 4000:
            data['shift_position'] = shift_position - bif_shift
        else:
            data['shift_position'] = np.nan 

    else: 
        data['shift_position'] = t_len

    return data



def generate_time_series(t_len, sigma_Y, sigma_X=0, seed=0, model='base', return_noise=True, bif_type='saddle-node', X_power=1, gamma=1):
    """
    Generate synthetic time series data based on specified parameters and models.
    Parameters:
    -----------
    t_len : int
        Length of the time series to generate.
    sigma_Y : float
        Standard deviation of the noise for the Y variable.
    sigma_X : float, optional, default=0
        Standard deviation of the noise for the X variable (used in certain models).
    seed : int, optional, default=0
        Seed for the random number generator to ensure reproducibility.
    model : str, optional, default='base'
        The model type to use for generating the time series. Options include:
        - 'base': Approaching a bifurcation without confounders.
        - 'confounderDecreasAC': Approaching a bifurcation with decreasing autocorrelation in the confounder.
        - 'confounderIncreasAC': Approaching a bifurcation with increasing autocorrelation in the confounder.
        - 'flatControl': No upcoming bifurcation and no confounder.
        - 'falseAlarm': No upcoming bifurcation but increasing autocorrelation in the confounder.
    return_noise : bool, optional, default=True
        Whether to return the noise component in the output.
    bif_type : str, optional, default='saddle-node'
        Type of bifurcation to simulate. Must be either 'pitchfork' or 'saddle-node'.
    X_power : int, optional, default=1
        Exponent applied to the confounder variable in certain models.
    gamma : float, optional, default=1
        Scaling factor for the confounder variable in certain models.
    Returns:
    --------
    dict
        A dictionary containing:
        - 'data': The generated time series data as a NumPy array.
        - 'noise': The noise component of the data (only included if `return_noise` is True).
    """

    # Validate bif_type and model type
    if bif_type not in ['pitchfork', 'saddle-node']:
        raise ValueError("Invalid bif_type. Must be either 'pitchfork' or 'saddle-node'.")
    
    if model not in ['base', 'confounderDecreasAC', 'confounderIncreasAC', 'flatControl', 'falseAlarm']:
        raise ValueError("Invalid model. Must be one of 'base', 'confounderDecreasAC', 'confounderIncreasAC', 'flatControl', or 'falseAlarm'.")

    # Initialize random generator
    random_state = np.random.default_rng(seed=seed)
    
    # Generate base random data
    dataR = random_state.normal(0, 0, size=(t_len, 1))
    dataY = random_state.normal(0, sigma_Y, size=(t_len, 1))

    # parameters
    h = 0.01  # time step
    initial_r = -1
    s = 0.02 if model not in ('falseAlarm', 'flatControl') else 0.0

    
    if model == 'base':
        data = np.hstack((dataR, dataY))

        # Initial conditions
        data[0, 0] = initial_r
        if bif_type == 'pitchfork':
            data[0, 1] = 0
        if bif_type=='saddle-node':
            data[0, 1] = 1.325

        for t in range(1, t_len):
            data[t, 0] += h * s + data[t-1, 0]

            if bif_type == 'pitchfork':
                data[t, 1] += data[t-1, 1] + h * (data[t-1, 0] * data[t-1, 1] - data[t-1, 1]**3)
            
            if bif_type == 'saddle-node':
                data[t, 1] += data[t-1, 1] + h * (data[t-1, 1] - data[t-1, 1]**3 - data[t-1, 0])

    elif model == 'flatControl':
        data = np.hstack((dataR, dataY))
        
        # Initial conditions
        data[0, 0] = initial_r
        if bif_type == 'pitchfork':
            data[0, 1] = 0
        if bif_type=='saddle-node':
            data[0, 1] = 1.325

        for t in range(1, t_len):
            data[t, 0] += data[t-1, 0]

            if bif_type == 'pitchfork':
                data[t, 1] += data[t-1, 1] + h * (data[t-1, 0] * data[t-1, 1] - data[t-1, 1]**3)
            
            if bif_type == 'saddle-node':
                data[t, 1] += data[t-1, 1] + h * (data[t-1, 1] - data[t-1, 1]**3 - data[t-1, 0])

    elif model == 'confounderDecreasAC' or model == 'confounderIncreasAC' or model == 'falseAlarm':
        if bif_type == 'pitchfork':
            dataX = random_state.normal(0, sigma_X, size=(t_len, 1))
        elif bif_type == 'saddle-node':
            dataX = random_state.normal(0, sigma_X, size=(t_len, 1))
        data = np.hstack((dataR, dataX, dataY))

        if model == 'confounderDecreasAC':
            beta_0 = 0.9
            beta_middle = 0.1
            k = (beta_middle - beta_0) / (t_len // 2)

        elif model == 'confounderIncreasAC' or model == 'falseAlarm':
            beta_0 = 0.1
            beta_final = 0.9
            k = (beta_final - beta_0) / t_len

        # Initial conditions
        data[0, 0] = initial_r
        data[0, 1] = 0
        if bif_type == 'pitchfork':
            data[0, 2] = 0
        if bif_type=='saddle-node':
            data[0, 2] = 1.325

        for t in range(1, t_len):
            data[t, 0] += h * s + data[t-1, 0]

            if model == 'confounderDecreasAC':
                beta_t = max(0, beta_0 + k * t)
                data[t, 1] += beta_t * data[t-1, 1]
    
            elif model == 'confounderIncreasAC' or model == 'falseAlarm':
                beta_t = beta_0 + k * t
                data[t, 1] += beta_t * data[t-1, 1]

            if bif_type == 'pitchfork':
                data[t, 2] += data[t-1, 2] + h * (data[t-1, 0] * data[t-1, 2] - data[t-1, 2]**3 + gamma * data[t-1, 1]**X_power)
                       
            if bif_type == 'saddle-node':
                data[t, 2] += data[t-1, 2] + h * (data[t-1, 2] - data[t-1, 2]**3 - data[t-1, 0] + gamma * data[t-1, 1]**X_power)

    if not return_noise:
        return {'data': data}
    else:
        if model == 'base':
            noise = dataY.flatten()
        else:
            noise = dataY.flatten() + data[:, 1]
        return {'data': data, 'noise': noise}




def truncate_data_before_bif(data_dict, bif_time=None):
    """
    Truncate each item in the 'data' dictionary up to the corresponding 'shift_positions' or a specified bifurcation time.

    Parameters:
    -----------
    data_dict : dict
        Dictionary containing the time series data. Expected keys are:
        - 'data': The generated time series data as a NumPy array.
        - 'noise': The noise component of the data (if available).
        - 'shift_position': The position of the detected shift in the time series.
    bif_time : int or None, optional
        If provided, truncates the data up to this specific time step. If None, truncates up to the detected shift position.

    Returns:
    --------
    dict
        A dictionary containing:
        - 'truncated_data': The truncated time series data.
        - 'truncated_noise': The truncated noise component of the data.
    """
    if bif_time is not None:
        # Truncate data and noise up to the specified bifurcation time
        truncated_data = data_dict['data'][:bif_time].copy()
        truncated_noise = data_dict['noise'][:bif_time].copy()
    else: 
        # Truncate data and noise up to the detected shift position
        shift_position = data_dict['shift_position']
        truncated_data = data_dict['data'][:shift_position].copy()
        truncated_noise = data_dict['noise'][:shift_position].copy()
    
    return {'truncated_data': truncated_data, 'truncated_noise': truncated_noise}