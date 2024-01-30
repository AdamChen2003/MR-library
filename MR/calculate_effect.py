import polars as pl
from MR.mr_methods.egger_regression import mr_egger_regression
from MR.mr_methods.inverse_variance_weighted import mr_inverse_variance_weighted
from MR.mr_methods.maximum_likelihood import mr_maximum_likelihood
from MR.mr_methods.median import mr_penalised_weighted_median, mr_simple_median, mr_weighted_median
from MR.mr_methods.mode import mr_penalised_weighted_mode, mr_simple_mode, mr_weighted_mode
from MR.mr_methods.wald_ratio import mr_wald_ratio


methods = [
    'inverse_variance_weighted',
    'wald_ratio',
    'maximum_likelihood',
    'simple_median',
    'weighted_median',
    'penalised_weighted_median',
    'simple_mode',
    'weighted_mode',
    'penalised_weighted_mode',
    'egger_regression',
    'presso'
]


def calculate_effect(data: pl.DataFrame, method: str):
    """
    Calculates causal effect using the specified method.
    List of methods can be accessed through the methods array.

    Arguments:

    data -- polars dataframe which contains the following columns:
        beta_exp: Vector of genetic effects on exposure
        beta_out: Vector of genetic effects on outcome
        se_exp: Standard errors of genetic effects on exposure
        se_out: Standard errors of genetic effects on outcome

    method -- str which specifies which method to use. 
        The list of available methods are accessible through the 'methods' list

    Returns:

    {
        'effect: causal effect estimation,
        'se' : standard error of effect estimation
    }
    """
    if method == 'inverse_variance_weighted':
        return mr_inverse_variance_weighted(data['beta_exp'], data['beta_out'], data['se_out'])

    elif method == 'wald_ratio':
        return mr_wald_ratio(data['beta_exp'], data['beta_out'], data['se_out'])

    elif method == 'maximum_likelihood':
        return mr_maximum_likelihood(data['beta_exp'], data['beta_out'], data['se_exp'], data['se_out'])

    elif method == 'simple_median':
        return mr_simple_median(data['beta_exp'], data['beta_out'], data['se_exp'], data['se_out'])

    elif method == 'weighted_median':
        return mr_weighted_median(data['beta_exp'], data['beta_out'], data['se_exp'], data['se_out'])

    elif method == 'penalised_weighted_median':
        return mr_penalised_weighted_median(data['beta_exp'], data['beta_out'], data['se_exp'], data['se_out'])

    elif method == 'simple_mode':
        return mr_simple_mode(data['beta_exp'], data['beta_out'], data['se_exp'], data['se_out'])

    elif method == 'weighted_mode':
        return mr_weighted_mode(data['beta_exp'], data['beta_out'], data['se_exp'], data['se_out'])

    elif method == 'penalised_weighted_mode':
        return mr_penalised_weighted_mode(data['beta_exp'], data['beta_out'], data['se_exp'], data['se_out'])

    elif method == 'egger_regression':
        return mr_egger_regression(data['beta_exp'], data['beta_out'], data['se_out'])
