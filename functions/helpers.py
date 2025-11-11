import numpy as np
import pandas as pd

import plotly.graph_objects as go

from scipy.ndimage import gaussian_filter1d



def linearized_rr(h, r_ts):
    """
    Computes the linearized return rate time series.

    This function takes a parameter `h` and a list of return rates `r_ts`, and computes the linearized return rate time series based on a specific transformation function.

    Parameters:
    h (float): A parameter used in the transformation function.
    r_ts (list of float): A list of return rates.

    Returns:
    list of float: A list containing the linearized return rate time series.
    """
    linearized_rr_ts = []

    def f(y):
        return 1 + h * (1 - 3 * y**2)
    for r in r_ts:
        if r<2/(3 * np.sqrt(3)):
            linearized_rr_ts.append(f(find_fixed_point(r)))
    return linearized_rr_ts


def find_fixed_point(r):
    """
    Finds the positive fixed point of the cubic equation y^3 - y + r = 0.

    Parameters:
    r (float): The constant term in the cubic equation.

    Returns:
    float or None: The positive fixed point if it exists, otherwise None.
    """
    coefficients = [1, 0, -1, r]  # y^3 - y + r = 0
    roots = np.roots(coefficients)  # Solve cubic equation
    real_roots = [root.real for root in roots if np.isclose(root.imag, 0)]  # Filter real roots and select the positive one

    positive_root = [y for y in real_roots if y > 0]
    return positive_root[0] if positive_root else None


def divide_trend_residuals(ts, method='running_mean', bandwidth=100, sigma=20):
    if method == 'running_mean':
        ts = pd.Series(ts)
        trend = ts.rolling(window=bandwidth, center=True).mean()
        residuals = ts - trend
    elif method == 'gaussian_smoothing':
        trend = gaussian_filter1d(ts, sigma=sigma)
        residuals = ts - trend
    return trend, residuals



def plot_smoothing(ts, trend, residuals):
    fig = go.Figure()

    # Left y-axis: Original and Trend
    fig.add_trace(go.Scatter(y=ts, mode='lines', name='Original Time Series',
                             line=dict(color='orange'), yaxis='y1'))
    fig.add_trace(go.Scatter(y=trend, mode='lines', name='Trend (smoothing)',
                             line=dict(color='red'), yaxis='y1'))

    # Right y-axis: Residuals
    fig.add_trace(go.Scatter(y=residuals, mode='lines', name='Residuals',
                             line=dict(color='#00AAD4'), yaxis='y2'))

    fig.update_layout(
        title='Time Series, Trend, and Residuals',
        xaxis=dict(title='Time'),
        yaxis=dict(title='Value (Original + Trend)', side='left'),
        yaxis2=dict(title='Residuals', overlaying='y', side='right'),
        legend=dict(x=0.025, y=0.96, traceorder='normal', bgcolor='rgba(255, 255, 255, 0.5)'),
        width=800,
        height=400,
        plot_bgcolor='rgba(215, 238, 244, 0.3)',
    )

    fig.show()



def plot_r_ac1(r_increasing_alpha, residuals, trend, window_size, lag=1):
    ts_r_increasing_alpha = pd.Series(r_increasing_alpha)
    ts_residuals = pd.Series(residuals)
    ts_trend = pd.Series(trend)

    fig = go.Figure()

    rolling_autocorr = ts_r_increasing_alpha.rolling(window=window_size, center=True).apply(lambda x: x.autocorr(lag=lag), raw=False)
    fig.add_trace(go.Scatter(x=rolling_autocorr.index, y=rolling_autocorr, mode='lines', name='AC1 of the original time series', showlegend=True, line=dict(color='orange')))

    rolling_autocorr = ts_residuals.rolling(window=window_size, center=True).apply(lambda x: x.autocorr(lag=lag), raw=False)
    fig.add_trace(go.Scatter(x=rolling_autocorr.index, y=rolling_autocorr, mode='lines', name='AC1 of the residuals', showlegend=True, line=dict(color='#00AAD4')))

    rolling_autocorr = ts_trend.rolling(window=window_size, center=True).apply(lambda x: x.autocorr(lag=lag), raw=False)
    fig.add_trace(go.Scatter(x=rolling_autocorr.index, y=rolling_autocorr, mode='lines', name='AC1 of the trend', showlegend=True, line=dict(color='red')))

    fig.update_layout(
        title='Rolling Autocorrelation',
        xaxis_title='Time',
        yaxis_title='Autocorrelation',
        width=800, height=400,
        legend=dict(x=0.025, y=0.92, traceorder='normal', bgcolor='rgba(255, 255, 255, 0.5)'),
        plot_bgcolor='rgba(215, 238, 244, 0.3)',
    )

    fig.show()