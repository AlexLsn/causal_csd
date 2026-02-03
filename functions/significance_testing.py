import numpy as np
import scipy.stats as st



def fourier_surrogates(ts, ns, seed=0):
    """
    Generate Fourier surrogates for a time series.

    Parameters:
    ts (array-like): The original time series.
    ns (int): The number of surrogates to generate.
    seed (int, optional): Seed for the random number generator.

    Returns:
    array-like: Array of surrogate time series.
    """
    if seed is not None:
        np.random.seed(seed)
    
    ts_fourier = np.fft.rfft(ts)
    random_phases = np.exp(np.random.uniform(0, 2 * np.pi, (ns, len(ts_fourier))) * 1.0j)
    ts_fourier_new = ts_fourier * random_phases
    new_ts = np.real(np.fft.irfft(ts_fourier_new, n=len(ts)))
    return new_ts



def p_value_from_fourier(ts, ns, tau, seed=0):
    """
    Calculate the p-value using Fourier surrogates.

    Parameters:
    ts (array-like): The original time series.
    ns (int): The number of surrogates to generate.
    tau (float): The observed test statistic.
    seed (int, optional): Seed for the random number generator.

    Returns:
    float: The calculated p-value.
    array-like: The generated surrogate time series.
    """
    # Convert the input time series to a NumPy array
    ts = np.array(ts)
    tlen = ts.shape[0]

    # Remove the mean from the time series to center it
    tsf = ts - ts.mean()

    # Generate Fourier surrogates
    nts = fourier_surrogates(tsf, ns, seed)

    # Initialize an array to store the test statistic for each surrogate
    stat = np.zeros(ns)
    tlen = nts.shape[1]

    # Compute the test statistic (slope of linear regression) for each surrogate
    for i in range(ns):
        stat[i] = st.linregress(np.arange(tlen), nts[i])[0]
            
    # Calculate the p-value by comparing the observed statistic to the surrogate distribution
    p = 1 - st.percentileofscore(stat, tau) / 100

    return p, nts