import polars as pl
from MR.mr_methods.mr_egger_regression import mr_egger_regression
from MR.mr_methods.mr_inverse_variance_weighted import mr_inverse_variance_weighted
from MR.mr_methods.mr_maximum_likelihood import mr_maximum_likelihood
from MR.mr_methods.mr_median import mr_penalised_weighted_median, mr_simple_median, mr_weighted_median
from MR.mr_methods.mr_mode import mr_mode
from MR.mr_methods.mr_wald_ratio import mr_wald_ratio


methods = [
    'inverse_variance_weighted',
    'wald_ratio',
    'maximum_likelihood',
    'simple_median',
    'weighted_median',
    'penalised_weighted_median',
    'weighted_mode',
    'egger_regression',
    'presso'
]


def calculate_effect(data: pl.DataFrame, method: str):
    """
    Calculates causal effect using the specified method.
    List of methods can be accessed through the methods array.
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

    elif method == 'weighted_mode':
        return mr_mode(data['beta_exp'], data['beta_out'], data['se_exp'], data['se_out'])

    elif method == 'egger_regression':
        return mr_egger_regression(data['beta_exp'], data['beta_out'], data['se_out'])
