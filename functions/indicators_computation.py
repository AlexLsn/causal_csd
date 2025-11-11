
import numpy as np
import pandas as pd

import scipy.signal
from scipy.stats import gaussian_kde

from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm

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
    for window in time_windows:
        if window[-1] >= data_length:
            break

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





##### Other indicators #####

def regular_autocorrelation(data, time_windows, lag, detrend: bool = False):
    """
    Computes the autocorrelation as given in the pandas package.

    Parameters:
    data (array-like or dict): The data to be analyzed. Can be a single array or a dictionary of arrays.
    num_variable (int): The number of the target variable.
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


###
# def restoring_rate(data, time_windows, rho, detrend: bool = False):
#     """
#     This code is derived from the work of Boers, N. in the paper:
#     Observation-based early-warning signals for a collapse of the Atlantic Meridional Overturning Circulation, Nature Climate Change, 2021.
#     https://doi.org/10.1038/s41558-021-01184-6

#     Original code repository: https://github.com/niklasboers/AMOC_EWS

#     Fits an AR(1) model to the differenced data using the window data as predictors.

#     Parameters:
#     data (array-like or dict): The data series to be analyzed. Can be a single array or a dictionary of arrays.
#     time_windows (list of arrays): The time windows to be analyzed.
#     rho (int): The lag parameter for the AR model.
#     detrend (bool): Indicates whether the data should be detrended.

#     Returns:
#     list or dict: List of AR(1) coefficients if input is a single array.
#                   Dictionary of AR(1) coefficients if input is a dictionary of arrays.
#     """
    
#     xs = []

#     for window in time_windows:
#         if window[-1] >= len(data):
#             break  # Stop if the entire window is not included in the data

#         xw = data[window]

#         if detrend:
#             xw = xw - xw.mean()  # Subtract mean
#             p0, p1 = np.polyfit(np.arange(xw.shape[0]), xw, 1)  # Linear fit for the demeaned data
#             xw = xw - p0 * np.arange(xw.shape[0]) - p1  # Remove linear trend
        
#         dxw = np.diff(xw)  # Compute first-order difference of the detrended data
        
#         xw = sm.add_constant(xw[:-1])  # Ensure the lengths match
#         model = sm.GLSAR(dxw, xw, rho)  # Fit an AR(1) to the differenced data dxw using the window data xw as predictors
#         results = model.iterative_fit(maxiter=10)  # The model is iteratively fitted with a maximum of 10 iterations

#         # Check for convergence
#         if not results.converged:
#             #print(f"{i}: Model did not converge.")
#             xs.append(np.nan)
#             continue

#         a = results.params[1]
#         xs.append(a)
    
#     return xs
    

# def restoring_rate_and_filter(data, time_windows, center_points, rho, detrend=False):
#     """
#     Computes the restoring rate and filters out invalid results.

#     Parameters:
#     data (array-like): The data to be analyzed.
#     num_variable (int): The number of the target variable.
#     time_windows (list of arrays): The time windows to be analyzed.
#     rho (int): The lag parameter for the AR model.
#     detrend (bool): Indicates whether the data should be detrended.

#     Returns:
#     tuple: Valid center points and their corresponding restoring rates.
#     """
#     rrate = restoring_rate(data, time_windows, rho, detrend)
#     valid_indices = ~np.isnan(rrate)
#     valid_center_points, valid_restoring_rate = center_points[valid_indices], np.array(rrate)[valid_indices]
#     return valid_center_points, valid_restoring_rate


def restoring_rate(data, time_windows, rho=1, detrend=True):
    """
    This code is derived from the work of Boers, N. in the paper:
    Observation-based early-warning signals for a collapse of the Atlantic Meridional Overturning Circulation, Nature Climate Change, 2021.
    https://doi.org/10.1038/s41558-021-01184-6

    Original code repository: https://github.com/niklasboers/AMOC_EWS

    Fits an AR(1) model to the differenced data using the window data as predictors.

    Parameters:
    data (array-like): Time series data.
    time_windows (list of arrays): List of index arrays defining windows.
    rho (int): Lag parameter for GLSAR.
    detrend (bool): Whether to detrend linearly after mean subtraction.

    Returns:
    np.ndarray: Array of AR(1) coefficients aligned with data length.
    """
    n = len(data)
    xs = np.full(n, np.nan)  # initialize full-length array with NaNs

    for window in time_windows:
        center = window[len(window)//2]  # center index of current window
        if window[-1] >= n:
            break  # skip windows extending beyond data length

        xw = data[window]

        # Subtract mean first
        xw = xw - np.mean(xw)

        if detrend:
            # Linear detrend after mean subtraction
            t = np.arange(len(xw))
            p0, p1 = np.polyfit(t, xw, 1)
            xw = xw - (p0 * t + p1)

        dxw = np.diff(xw)

        # GLSAR expects predictors to be one row shorter than dependent
        xw_const = sm.add_constant(xw[:-1])
        model = sm.GLSAR(dxw, xw_const, rho=rho)
        results = model.iterative_fit(maxiter=1000)

        if not results.converged:
            #xs[center] = np.nan
            print(f"Warning: AR(1) model did not converge for window centered at index {center}.")
            continue

        a = results.params[1]  # AR(1) coefficient
        xs[center] = a

    xs = xs[~np.isnan(xs)]  # Remove NaNs for plotting

    return xs



# def BB_method(data, time_windows, detrend: bool = True, y_ac_greaterthan_x_ac: bool = True):
#     """
#     This code is derived from the work of Boettner&Boers, 2022 in the paper:
#     Critical slowing down in dynamical systems driven by nonstationary correlated noise, Physical Review Research, 2022.
#     https://doi.org/10.1103/PhysRevResearch.4.013230

#     Original code repository: https://github.com/ChrisBoettner/CSD_autocorrelated_noise/tree/main

#     Computes the unbiased phi estimate using the analytical correction method
#     on predefined windows of the time series data.

#     Parameters:
#     data (array-like or dict): The data to be analyzed. Can be a single array or a dictionary of arrays.
#     num_variable (int): The index of the variable to analyze in the data. Used if data is multi-dimensional.
#     time_windows (list of tuples): List of tuples where each tuple contains the start and end indices for a window.
#     detrend (bool): Indicates whether the data should be detrended.

#     Returns:
#     list or dict: List of unbiased phi estimates if input is a single array.
#                   Dictionary of unbiased phi estimates if input is a dictionary of arrays.
#     """
    
#     def estimate_phi_a_on_window(data):
#         """
#         Compute the unbiased phi estimate for a single predefined window.

#         Parameters:
#         y (array-like): The time series data for the current window.

#         Returns:
#         float: The unbiased phi estimate.
#         """
#         # Prepare data for linear regression
#         data1 = data[1:]    # y_t
#         data2 = data[:-1]   # y_{t-1}
        
#         # Create the design matrix for OLS (including intercept and lag term)
#         X = np.column_stack([np.ones(len(data2)), data2])
        
#         # Fit OLS model
#         model = LinearRegression(fit_intercept=False)
#         model.fit(X, data1)
#         params = model.coef_
        
#         # Calculate biased estimate for phi
#         phi_b = params[1]
        
#         # Calculate residuals and autocorrelation of residuals
#         residuals = data1 - (model.intercept_ + phi_b * data2) 

#         residuals_t = residuals[1:]
#         residuals_t_minus_1 = residuals[:-1]
#         rho_b = np.sum(residuals_t * residuals_t_minus_1) / np.sum(residuals_t_minus_1 ** 2)
        
#         # Calculate unbiased phi using the analytical correction
#         a = phi_b + rho_b
#         b = rho_b / phi_b
#         c = a**2 - 4*b

#         if c < 0:
#             phi = np.nan  # If c is negative, the result is not valid
        
#         else:
#             # Calculate the two potential roots
#             phi = np.nan * np.ones_like(a)
#             phi1 = (a + np.sqrt(c)) / 2
#             phi2 = (a - np.sqrt(c)) / 2
        
#             # If y_ac is greater than x_ac, select phi1; otherwise select phi2
#             if y_ac_greaterthan_x_ac==True:
#                 phi = phi1
#             else:
#                 phi = phi2
        
#         return phi
    

#     phi_estimates = []
    
#     for window in time_windows:
#         if window[-1] >= len(data):
#             break  # Stop if the entire window is not included in the data

#         # Extract the data for the current window
#         xw = data[window]
        
#         if detrend:
#             # Detrend the window data
#             xw = xw - np.mean(xw)
#             p0, p1 = np.polyfit(np.arange(xw.size), xw, 1)
#             xw -= (p0 * np.arange(xw.size) + p1)
        
#         # Calculate the unbiased phi for the current window
#         phi_estimates.append(estimate_phi_a_on_window(xw))
    
#     return phi_estimates



def BB_method(data, window_size):
    """
    This code is derived from the work of Boettner&Boers, 2022 in the paper:
    Critical slowing down in dynamical systems driven by nonstationary correlated noise, Physical Review Research, 2022.
    https://doi.org/10.1103/PhysRevResearch.4.013230

    Original code repository: https://github.com/ChrisBoettner/CSD_autocorrelated_noise/tree/main
    Computes bias-corrected AR(1) coefficient estimates using a rolling window approach,
    based on the method described by Boettner & Baur (BB method).
    
    Parameters:
    data : np.ndarray
        Time series data of shape (n_time,) or (n_time, n_series).
    window_size : int
        Size of the rolling window used for local regression and statistics.
    
    Returns:
    phi_star : np.ndarray
        Biased AR(1) coefficient estimates, shape (n_time, n_series), padded with NaNs at the start.
    phi : np.ndarray
        Bias-corrected AR(1) coefficient estimates, shape (n_time, n_series), padded with NaNs.
    rho : np.ndarray
        Autocorrelation of residuals estimates, shape (n_time, n_series), padded with NaNs.
    """
    if data.ndim == 1:
        data = data[:, np.newaxis]

    data1 = data[1:, :]
    data2 = data[:-1, :]

    params_list = []
    for i in range(data1.shape[1]):
        model = RollingOLS(data1[:, i], np.vander(data2[:, i], 2, increasing=True),
                           window=window_size, missing="skip")
        params = model.fit(params_only=True).params  # shape (n_time, 2)
        params_list.append(params)

    params_star = np.stack(params_list, axis=1)  # shape (n_time, n_series, 2)

    e = data1 - params_star[:, :, 0] - params_star[:, :, 1] * data2
    e1 = e[1:, :]
    e2 = e[:-1, :]

    e1e2 = pd.DataFrame(e1 * e2).rolling(window_size).sum().values
    e22 = pd.DataFrame(e2 ** 2).rolling(window_size).sum().values
    rho_star = e1e2 / e22

    a = params_star[1:, :, 1] + rho_star
    b = rho_star / params_star[1:, :, 1]
    c = a ** 2 - 4 * b

    c[c < 0] = np.nan

    phi = (a + np.sqrt(c)) / 2
    rho = (a - np.sqrt(c)) / 2

    phi_star = np.vstack([np.full((1, data1.shape[1]), np.nan), params_star[:, :, 1]])
    phi = np.vstack([np.full((2, data1.shape[1]), np.nan), phi])
    rho = np.vstack([np.full((2, data1.shape[1]), np.nan), rho])

    phi_star = phi_star[~np.isnan(phi_star)]
    
    return phi_star, phi, rho





###
def rosa(data_y, data_eta, time_windows, dt, rosa_method="ratio", detrend: bool = False):
    """
    This code is derived from the work of Clarke et al., 2023 in the paper:
    Seeking more robust early warning signals for climate tipping points: the ratio of spectra method (ROSA), Environmental Researcher Letters, 2023.
    https://doi.org/10.1088/1748-9326/acbc8d

    Original code repository: https://github.com/josephjclarke/BeyondWhiteNoiseEWS

    Performs windowed spectral analysis on a time series.

    Parameters:
    data_x (array-like or dict): The primary time series data (x) to be analyzed.
    data_eta (array-like or dict): The secondary time series data (eta) to be analyzed.
    time_windows (list of arrays): The time windows to be analyzed.
    dt (float): The time step between consecutive data points in the time series.
    method (str): The method used for fitting ('ratio' or 'least_squares').
    detrend (bool): Indicates whether the data should be detrended.

    Returns:
    list or dict: List of ls estimates if input is a single array.
                  Dictionary of ls estimates if input is a dictionary of arrays.
    """
    
    def fitfunction(f, ls):
        return (1 / dt)**2 / ((2 * np.pi * f)**2 + ls**2)

    ls = []

    for window in time_windows:
        if window[-1] >= len(data_y) or window[-1] >= len(data_eta):
            break  # Stop if the entire window is not included in the data

        # Extract windowed data
        xw = data_y[window]
        etaw = data_eta[window]

        # Detrend if required
        if detrend:
            xw = xw - xw.mean()
            etaw = etaw - etaw.mean()

            p0, p1 = np.polyfit(np.arange(xw.shape[0]), xw, 1)
            xw = xw - p0 * np.arange(xw.shape[0]) - p1

            p0, p1 = np.polyfit(np.arange(etaw.shape[0]), etaw, 1)
            etaw = etaw - p0 * np.arange(etaw.shape[0]) - p1

        # Compute power spectral densities
        f, Sxx = scipy.signal.welch(xw, fs=1/dt)
        f, Sff = scipy.signal.welch(etaw, fs=1/dt)

        if rosa_method == "ratio":
            popt, _ = scipy.optimize.curve_fit(fitfunction, f[1:], Sxx[1:] / Sff[1:], p0=[1.0], bounds=(0.0, np.inf))
            ls.append(popt[0])
        else:
            def f2m(L):
                return Sff[1:] * (1 / dt)**2 / ((2 * np.pi * f[1:])**2 + L**2) - Sxx[1:]

            opt = scipy.optimize.least_squares(f2m, 1.0, bounds=(0.0, np.inf))
            ls.append(opt.x.item())

    return ls




def block_bootstrap_sliding_windows(
    time_series,
    time_windows,
    boot_blocklength,
    n_bootstrap,
    seed=0
):
    """
    Block bootstrap sur des fenêtres glissantes d'une série temporelle univariée,
    avec recentrage de chaque réplique bootstrap pour conserver la moyenne locale.

    Parameters
    ----------
    time_series : np.ndarray
        Tableau 1D de forme (T,) représentant la série temporelle.
    time_windows : list of np.ndarray
        Liste de tableaux d’indices, chaque tableau spécifie les indices temporels d’une fenêtre.
    boot_blocklength : int
        Longueur des blocs bootstrap.
    n_bootstrap : int
        Nombre de réplicats bootstrap par fenêtre.
    seed : int, optional
        Graine aléatoire pour la reproductibilité.

    Returns
    -------
    np.ndarray
        Réplicats bootstrap de forme (n_bootstrap, n_windows, window_length).
    """
    seed_seq = np.random.SeedSequence(seed)
    window_seeds = seed_seq.spawn(len(time_windows))

    n_windows = len(time_windows)
    window_length = len(time_windows[0])
    output = np.empty((n_bootstrap, n_windows, window_length))

    for w_idx, (window_indices, window_seed) in enumerate(zip(time_windows, window_seeds)):
        rng = np.random.default_rng(window_seed)
        window_data = time_series[window_indices]
        n_blocks = int(np.ceil(window_length / boot_blocklength))

        for b in range(n_bootstrap):
            block_starts = rng.integers(0, window_length - boot_blocklength + 1, size=n_blocks)
            bootstrapped = np.concatenate([
                window_data[start:start + boot_blocklength]
                for start in block_starts
            ])[:window_length]

            output[b, w_idx] = bootstrapped

    return output










# def compute_slope_with_positive_density(x, y, density, min_density):
#     """
#     Computes the slope of a linear regression for points with density above a threshold.

#     Parameters:
#     x (np.ndarray): Independent variable values.
#     y (np.ndarray): Dependent variable values.
#     density (np.ndarray): Density values corresponding to x.
#     min_density (float): Minimum density threshold for filtering points.

#     Returns:
#     tuple: Slope, mean residual, and standard deviation of y.
#     """
#     valid_indices = np.where(density > min_density)[0]
#     valid_indices = valid_indices[10:-10]  # Remove edge points to avoid boundary effects

#     if len(valid_indices) < 2:
#         raise ValueError("Not enough points: Need at least two points to perform linear regression")
    
#     x_valid = x[valid_indices]
#     y_valid = y[valid_indices]

#     # Perform linear regression
#     slope, intercept, r_value, p_value, std_err = st.linregress(x_valid, y_valid)
#     residuals = y_valid - (slope * x_valid + intercept)
#     mean_residual = np.mean(np.abs(residuals))
#     std_dev_y_valid = np.std(y_valid)
    
#     return slope, mean_residual, std_dev_y_valid


# def compute_effect_slopes(causal_autodependencies, densities, intervention_data_windows, min_density):
#     """
#     Computes the slopes of causal effects for multiple time windows.

#     Parameters:
#     causal_autodependencies (list): List of causal effect values for each window.
#     densities (list): List of density functions for each window.
#     intervention_data_windows (list): List of intervention data for each window.
#     min_density (float): Minimum density threshold for filtering points.

#     Returns:
#     tuple: Lists of slopes, residuals, and standard deviations for each window.
#     """
#     min_slope_threshold = 0.05
#     max_residual_threshold = 1

#     effect_slopes = []
#     residuals = []
#     std_devs = []

#     for i in range(len(causal_autodependencies)):
#         x = intervention_data_windows[i]
#         y = causal_autodependencies[i]
#         density = densities[i](x)
#         try:
#             result = compute_slope_with_positive_density(x, y, density, min_density, min_slope_threshold, max_residual_threshold)
#             slope, residual, std_dev = result
#             effect_slopes.append(slope)
#             residuals.append(residual)
#             std_devs.append(std_dev)
#         except ValueError:
#             raise ValueError(f"Error comes from time window number: {i}")
        
#     return effect_slopes, residuals, std_devs



# def pipeline_causalEE(causal_effects, data, estimator, data_transform, n_points, time_windows, var_names, min_density):
#     """
#     Pipeline for computing causal effect slopes and related metrics.

#     Parameters:
#     causal_effects: Causal effect model.
#     data (np.ndarray): Time series data.
#     estimator: Estimator for causal effect computation.
#     data_transform (callable): Transformation function for data.
#     n_points (int): Number of points for intervention data.
#     time_windows (list): List of time window indices.
#     var_names (list): Variable names in the data.
#     min_density (float): Minimum density threshold for filtering points.

#     Returns:
#     tuple: Effect slopes, residuals, and standard deviations for each window.
#     """
#     causal_autodependencies, intervention_data_windows = compute_causal_effect_on_windows(
#         causal_effects, data, var_names=var_names, 
#         time_windows=time_windows, n_points=n_points, 
#         estimator=estimator, data_transform=data_transform,
#     )
#     densities = windows_density(data, time_windows=time_windows)

#     try:
#         effect_slopes, residuals, std_devs = compute_effect_slopes(causal_autodependencies, densities, intervention_data_windows, min_density=min_density)
#     except ValueError as e:
#         print(f"An error occurred: {e}")

#     return effect_slopes, residuals, std_devs