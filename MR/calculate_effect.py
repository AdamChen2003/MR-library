import polars as pl
from scipy.optimize import minimize
import numpy as np
from statistics import stdev


# iwv: Inverse Weighted Variance
# wr: Wald Ratio
# ml: Maximum Likelihood
# sm: Simple Median
# wm: Weighted Median
# pwm: Penalised Weighted Median
methods = ['iwv', 'wr', 'ml', 'sm', 'wm', 'pwm']


def calculate_effect(data: pl.DataFrame, method: str):
    """
    Calculates causal effect using the specified method.
    List of methods can be accessed through the methods array.
    """
    if method == 'ivw':
        return mr_inverse_variance_weighted(data['beta_exp'], data['beta_out'], data['se_out'])

    elif method == 'wr':
        return mr_wald_ratio(data['beta_exp'], data['beta_out'], data['se_out'])

    elif method == 'ml':
        return mr_maximum_likelihood(data['beta_exp'], data['beta_out'], data['se_exp'], data['se_out'])

    elif method == 'sm':
        return mr_simple_median(data['beta_exp'], data['beta_out'], data['se_exp'], data['se_out'])

    elif method == 'wm':
        return mr_weighted_median(data['beta_exp'], data['beta_out'], data['se_exp'], data['se_out'])

    elif method == 'pwm':
        return mr_penalised_weighted_median(data)


def mr_inverse_variance_weighted(beta_exp, beta_out, se_out):
    effect = (beta_exp * beta_out * se_out
              ** -2).sum() / (beta_exp ** 2 * se_out ** -2).sum()
    se = ((beta_exp ** 2 *
           se_out ** -2).sum()) ** -0.5

    return {
        'effect': effect, 'se': se
    }


def mr_wald_ratio(beta_exp, beta_out, se_out):
    effect = (beta_out / beta_exp).mean()
    se = (se_out / abs(beta_exp)).mean()

    return {
        'effect': effect, 'se': se
    }


def mr_maximum_likelihood(beta_exp, beta_out, se_exp, se_out):
    n = len(beta_exp)

    def log_likelihood(param):
        return 1/2 * ((beta_exp - param[0:n])**2/se_exp**2).sum() + 1/2 * ((beta_out - param[n] * param[0:n])**2 / se_out**2).sum()

    initial = np.append((beta_exp.to_numpy()), (beta_exp*beta_out /
                                                se_out**2).sum()/(beta_exp**2/se_out**2).sum())

    res = minimize(log_likelihood, initial)
    effect = res.x[n]
    se = res.hess_inv[n, n] ** 1/2

    return {
        'effect': effect, 'se': se
    }


def weighted_median(b_iv, weights):
    beta_IV_sorted = np.sort(b_iv)
    weights_sorted = np.sort(weights)
    weights_sum = np.cumsum(weights_sorted) - 1/2 * weights_sorted
    weights_sum = weights_sum/np.sum(weights_sorted)
    below = np.max(np.where(weights_sum < 1/2))
    return beta_IV_sorted[below-1] + (beta_IV_sorted[below] - beta_IV_sorted[below-1]) * (
        1/2-weights_sum[below-1])/(weights_sum[below]-weights_sum[below-1])


def weighted_median_se(beta_exp, beta_out, se_exp, se_out, weights, nboot=1000):
    med = []
    for i in range(0, nboot):
        beta_exp_boot = np.random.normal(
            loc=beta_exp, scale=se_exp, size=len(beta_exp))
        beta_out_boot = np.random.normal(
            loc=beta_out, scale=se_out, size=len(beta_out))
        betaIV_boot = beta_out_boot/beta_exp_boot
        med.append(weighted_median(betaIV_boot, weights))

    return stdev(med)


def mr_simple_median(beta_exp, beta_out, se_exp, se_out):
    n = len(beta_exp)
    b_iv = beta_out / beta_exp
    effect = weighted_median(b_iv, np.repeat(1/n, n))
    se = weighted_median_se(beta_exp, beta_out, se_exp,
                            se_out, np.repeat(1/n, n))

    return {
        'effect': effect, 'se': se
    }


def mr_weighted_median(beta_exp, beta_out, se_exp, se_out):
    b_iv = beta_out / beta_exp
    VBj = se_out**2/beta_exp**2 + \
        beta_out**2 * ((se_exp**2))/beta_exp**4
    effect = weighted_median(b_iv, 1/VBj)
    se = weighted_median_se(beta_exp, beta_out, se_exp, se_out, 1/VBj)

    return {
        'effect': effect, 'se': se
    }


def mr_penalised_weighted_median(data):
    return
