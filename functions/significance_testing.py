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
    tau (float): The threshold value for the test statistic.
    seed (int, optional): Seed for the random number generator.


    Returns:
    float: The calculated p-value.
    """
    ts = np.array(ts)
    tlen = ts.shape[0]

    tsf = ts - ts.mean()
    nts = fourier_surrogates(tsf, ns, seed)

    stat = np.zeros(ns)
    tlen = nts.shape[1]

    for i in range(ns):
        stat[i] = st.linregress(np.arange(tlen), nts[i])[0]
            
    p = 1 - st.percentileofscore(stat, tau) / 100
    return p, nts


def pvalues_fourier_multiple_data(data_dict, nb_surrogates, seed=0):
    """
    Calculate p-values for multiple data sets using Fourier surrogates.

    Parameters:
    data_dict (dict): Dictionary of data sets.
    nb_surrogates (int): The number of surrogates to generate.
    seed (int, optional): Seed for the random number generator.

    Returns:
    list: List of calculated p-values.
    """
    pvalues = []
    
    for i in range(len(data_dict)):        
        slope, intercept = np.polyfit([i for i in range(len(data_dict[i]))], data_dict[i], 1)
        pv = p_value_from_fourier(data_dict[i], nb_surrogates, slope, seed)
        
        pvalues.append(pv)

    return pvalues



##########################################################################################################################
def nb_pvalues_threshold(pvalues, threshold):
    inf = len([x for x in pvalues if x <= threshold])
    sup = len([x for x in pvalues if x > threshold])

    return inf, sup



def fisher_combined_pvalue(pvalues):
    """
    Perform Fisher's Combined Probability Test.
    
    Parameters:
    pvalues (list or array): List of p-values to combine.
    
    Returns:
    float: Combined p-value.
    """
    chi2_statistic = -2 * np.sum(np.log(pvalues))
    df = 2 * len(pvalues) # Degrees of freedom
    combined_pvalue = st.chi2.sf(chi2_statistic, df)
    
    return float(f"{combined_pvalue:.4e}") 


def display_nb_pvalues_threshold(pvalues_causal_auto, pvalues_reg_auto, pvalues_restoring_rate, threshold, display=True):

    inf_causal, sup_causal = nb_pvalues_threshold(pvalues_causal_auto, threshold)
    inf_reg, sup_reg = nb_pvalues_threshold(pvalues_reg_auto, threshold)
    inf_restoring, sup_restoring = nb_pvalues_threshold(pvalues_restoring_rate, threshold)

    if display==True:
        print('Causal autodependency: # pvalues < =', threshold, ':', inf_causal, '# pvalues >', threshold, ':', sup_causal)
        print('Regular AC1: # pvalues < =', threshold, ':', inf_reg, '# pvalues >', threshold, ':', sup_reg)
        print('Restoring rate: # pvalues < =', threshold, ':', inf_restoring, '# pvalues >', threshold, ':', sup_restoring)

    return [[inf_causal, sup_causal], [inf_reg, sup_reg], [inf_restoring, sup_restoring]]


def display_combined_pvalues(causal_auto_dict, reg_auto_dict, restoring_rate_dict, nb_surrogates, display=True):
    pvalues_causal_auto = pvalues_fourier_multiple_data(causal_auto_dict, nb_surrogates)
    pvalues_reg_auto = pvalues_fourier_multiple_data(reg_auto_dict, nb_surrogates)
    pvalues_restoring_rate = pvalues_fourier_multiple_data(restoring_rate_dict, nb_surrogates)

    if display==True:
        print('combined pvalue causal:', fisher_combined_pvalue(pvalues_causal_auto))
        print('combined pvalue regular:', fisher_combined_pvalue(pvalues_reg_auto))
        print('combined pvalue restoring rate:', fisher_combined_pvalue(pvalues_restoring_rate))

    return pvalues_causal_auto, pvalues_reg_auto, pvalues_restoring_rate