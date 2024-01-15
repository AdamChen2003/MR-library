import polars as pl
from scipy.optimize import minimize
import numpy as np


# iwv: Inverse Weighted Variance
# wr: Wald Ratio
# ml: Maximum Likelihood
# wm: Weighed Median
# sm: Simple Median
methods = ['iwv', 'wr', 'ml', 'wm']


def calculate_effect(data: pl.DataFrame, method: str):
    """
    Calculates causal effect using the specified method
    """
    if method == 'ivw':
        effect = (data['beta_exp'] * data['beta_out'] * data['se_out']
                  ** -2).sum() / (data['beta_exp'] ** 2 * data['se_out'] ** -2).sum()

        se = ((data['beta_exp'] ** 2 *
              data['se_out'] ** -2).sum()) ** -0.5

    elif method == 'wr':
        effect = (data['beta_out'] / data['beta_exp']).mean()
        se = (data['se_out'] / abs(data['beta_exp'])).mean()

    elif method == 'ml':
        n = data.shape[0]

        def log_likelihood(param):
            return 1/2 * ((data['beta_exp'] - param[0:n])**2/data['se_exp']**2).sum() + 1/2 * ((data['beta_out'] - param[n] * param[0:n])**2 / data['se_out']**2).sum()

        initial = np.append((data['beta_exp'].to_numpy()), (data['beta_exp']*data['beta_out'] /
                                                            data['se_out']**2).sum()/(data['beta_exp']**2/data['se_out']**2).sum())

        res = minimize(log_likelihood, initial)
        effect = res.x[n]
        se = res.hess_inv[n, n] ** 1/2

    return {
        'effect': effect, 'se': se
    }
