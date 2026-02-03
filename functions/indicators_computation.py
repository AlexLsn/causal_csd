
import numpy as np
import pandas as pd

from scipy.stats import gaussian_kde

from tigramite import data_processing as pp




##### Functions for deriving the DCE #####


def produce_intervention_data(ts, margin, n_points):
    """
    Generates intervention data (here, values taken from a given time series with a margin).

    Parameters:
    ts (np.ndarray): The time series data.
    margin (float): Margin to extend beyond the min and max of the time series.
    n_points (int): Number of points to generate in the intervention data.

    Returns:
    tuple: A tuple containing the intervention data and its reshaped version.
    """
    max_intervention = np.max(ts) + margin
    min_intervention = np.min(ts) - margin

    intervention_data = np.linspace(min_intervention, max_intervention, n_points)
    intervention_data_here = np.tile(intervention_data.reshape(n_points, 1), (1, 1))

    return intervention_data, intervention_data_here


def windows_density(data, time_windows):
    """
    Computes density functions for data in each time window.

    Parameters:
    data (np.ndarray): Time series data.
    time_windows (list): List of time window indices.

    Returns:
    list: List of density functions for each window.
    """
    data_length = data.shape[0]
    densities = []

    for window in time_windows:
        if window[-1] >= data_length:
            break
        density = gaussian_kde(data[window, -1])
        densities.append(density)
    return densities


def compute_causal_effect_on_windows(causal_effects, data, var_names, time_windows, n_points, estimator, data_transform=None, detrend=False):
    """
    Computes causal effects for data in each time window.

    Parameters:
    causal_effects: Initialized CausalEffects class.
    data (np.ndarray): Time series data.
    var_names (list): Variable names in the data.
    time_windows (list): List of time window indices.
    n_points (int): Number of intervention data points.
    estimator: Estimator for causal effect computation.
    data_transform (callable, optional): Transformation function for data.
    detrend (bool): Whether to detrend the data.

    Returns:
    tuple: Estimated causal effects and intervention data for each window.
    """
    estimated_causal_effects = []
    intervention_data_windows = []
    data_copy = data.copy()
    data_length = data_copy.shape[0]

    # compute causal effect for each window
    i=0
    for window in time_windows:
        if window[-1] >= data_length:
            break
        i=i+1
        window_slice = data_copy[window, :].copy() # get the data slice for the current window
        if detrend: # remove linear trend of each window if specified
            wlen = window_slice.shape[0]
            t = np.arange(wlen)
            for i in range(window_slice.shape[1]):
                xw_i = window_slice[:, i].astype(float).copy()
                xw_i = xw_i - xw_i.mean()
                slope, intercept = np.polyfit(t, xw_i, 1)
                detrended = xw_i - (slope * t + intercept)
                window_slice[:, i] = detrended

        dataframe = pp.DataFrame(window_slice, var_names=var_names) # create tigramite dataframe for the current window
        intervention_data_window, intervention_data_window_here = produce_intervention_data(window_slice[:, -1], 0.01, n_points)

        # fit and predict causal effect for the current window
        causal_effects.fit_total_effect(
            dataframe=dataframe, 
            estimator=estimator,
            adjustment_set='optimal',
            conditional_estimator=None,  
            data_transform=data_transform,
            mask_type=None,
        )
        
        if data_transform is not None:
            transform_interventions_and_prediction = True
        else:
            transform_interventions_and_prediction = False
        window_estimated_causal_effects = causal_effects.predict_total_effect(
            intervention_data=intervention_data_window_here,
            transform_interventions_and_prediction=transform_interventions_and_prediction)

        estimated_causal_effects.append(window_estimated_causal_effects) # store estimated causal effects for the current window (one effect per intervention point)
        intervention_data_windows.append(intervention_data_window) # store intervention data for the current window

    return estimated_causal_effects, intervention_data_windows



def compute_weighted_slope(x, y, weights):
    """
    Computes a weighted slope using weighted least squares.

    Parameters:
    -----------
    x : np.ndarray
        Independent variable (1D array).
    y : np.ndarray
        Dependent variable (1D array).
    weights : np.ndarray
        Weights for each data point (1D array).

    Returns:
    --------
    tuple
        (slope, mean_residual, std_dev_y)
        slope          : Estimated regression slope.
        mean_residual  : Mean absolute residual between observed and fitted y.
        std_dev_y      : Standard deviation of the observed y values.
    """

    # Each row of X corresponds to one observation [x_i, 1],
    # representing the linear model y = slope * x + intercept.
    X = np.vstack([x, np.ones(len(x))]).T  # shape (n_samples, 2)
    W = np.diag(weights)  # diagonal weight matrix W

    # (Xᵀ W X) β = Xᵀ W y
    XT_W_X = X.T @ W @ X       # (2x2 matrix)
    XT_W_y = X.T @ W @ y       # (2-element vector)
    # Solve for β = [slope, intercept]ᵀ
    beta = np.linalg.inv(XT_W_X) @ XT_W_y
    slope, intercept = beta[0], beta[1]

    residuals = y - (slope * x + intercept)
    mean_residual = np.mean(np.abs(residuals))
    std_dev_y = np.std(y)

    return slope, mean_residual, std_dev_y



def compute_effect_slopes_weighted(causal_autodependencies, densities, nb_edge_points, intervention_data_windows):
    """
    Computes weighted slopes of causal effects for multiple time windows.

    Parameters:
    causal_autodependencies (list): List of causal effect values for each window.
    densities (list): List of density functions for each window.
    nb_edge_points (int): Number of edge points to exclude from weighting.
    intervention_data_windows (list): List of intervention data for each window.

    Returns:
    tuple: Lists of slopes, residuals, and standard deviations for each window.
    """
    effect_slopes = []
    residuals = []
    std_devs = []

    # Compute weighted slope for each window
    for i in range(len(causal_autodependencies)):
        x = intervention_data_windows[i]
        y = causal_autodependencies[i]
        density = densities[i](x)

        # compute and normalize weights
        weights = np.copy(density).astype(float)
        if nb_edge_points > 0:
            weights[:nb_edge_points] = 0.0
            weights[-nb_edge_points:] = 0.0

        sumw = np.sum(weights)
        if sumw == 0.0:
            weights = np.ones_like(weights, dtype=float)
            sumw = np.sum(weights)

        weights = weights / sumw

        # compute weighted slope
        try:
            slope, residual, std_dev = compute_weighted_slope(x, y, weights)
            effect_slopes.append(slope)
            residuals.append(residual)
            std_devs.append(std_dev)
        except np.linalg.LinAlgError:
            raise ValueError(f"Error in weighted least squares for time window number: {i}")

    return effect_slopes, residuals, std_devs


def pipeline_causalEE_weighted(causal_effects, data, estimator, data_transform, n_points, time_windows, var_names, edge_points_proportion=0.1, detrend=False):
    """
    Pipeline for computing weighted causal effect slopes and related metrics.

    Parameters:
    causal_effects: Causal effect model.
    data (np.ndarray): Time series data.
    estimator: Estimator for causal effect computation.
    data_transform (callable): Transformation function for data.
    n_points (int): Number of points for intervention data.
    time_windows (list): List of time window indices.
    var_names (list): Variable names in the data.
    edge_points_proportion (float): Proportion of edge points to exclude from weighting.
    detrend (bool): Whether to detrend the data.

    Returns:
    tuple: Weighted effect slopes, residuals, and standard deviations for each window.
    """
    # compute causal effects on windows, one effect per intervention point
    causal_autodependencies, intervention_data_windows = compute_causal_effect_on_windows(
        causal_effects, data, var_names=var_names, 
        time_windows=time_windows, n_points=n_points, 
        estimator=estimator, data_transform=data_transform,
        detrend=detrend
    )
    densities = windows_density(data, time_windows=time_windows) # get the densities of Y for each window

    try:
        nb_edge_points = int(edge_points_proportion * n_points) # compute number of edge points to exclude from the slope estimation
        # compute weighted slopes of causal effects for each window
        effect_slopes, residuals, std_devs = compute_effect_slopes_weighted(causal_autodependencies, densities, nb_edge_points, intervention_data_windows)
    except ValueError as e:
        print(f"An error occurred: {e}")

    return effect_slopes, residuals, std_devs





##### Autocorrelation #####

def regular_autocorrelation(data, time_windows, lag, detrend: bool = False):
    """
    Computes the autocorrelation as given in the pandas package.

    Parameters:
    data (array-like or dict): The data to be analyzed. Can be a single array or a dictionary of arrays.
    time_windows (list of arrays): The time windows to be analyzed.
    lag (int): The lag for the autocorrelation.
    detrend (bool): Indicates whether the data should be detrended.

    Returns:
    list or dict: List of autocorrelation coefficients if input is a single array.
                  Dictionary of autocorrelation coefficients if input is a dictionary of arrays.
    """
    
    reg_autocorr = []

    for window in time_windows:
        if window[-1] >= len(data):
            break  # Stop if the entire window is not included in the data

        if detrend:
            xw = data[window]
            xw = xw - xw.mean()
            
            p0, p1 = np.polyfit(np.arange(xw.shape[0]), xw, 1)
            series = pd.Series(xw - p0 * np.arange(xw.shape[0]) - p1)
        else:
            series = pd.Series(data[window])
            
        reg_autocorr.append(series.autocorr(lag))
        
    return reg_autocorr