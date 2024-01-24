import numpy as np
from statistics import stdev
from scipy.stats import chi2


def weighted_median(beta_iv, weights):
    """
    Weighted median method.

    Arguments:

    beta_iv -- Wald Ratios

    weights -- Weights for each SNP
    """
    beta_iv_sorted = np.sort(beta_iv)
    weights_sorted = np.sort(weights)
    weights_sum = np.cumsum(weights_sorted) - 1/2 * weights_sorted
    weights_sum = weights_sum/np.sum(weights_sorted)
    below = np.max(np.where(weights_sum < 1/2))
    return beta_iv_sorted[below-1] + (beta_iv_sorted[below] - beta_iv_sorted[below-1]) * (
        1/2-weights_sum[below-1])/(weights_sum[below]-weights_sum[below-1])


def weighted_median_bootstrap(beta_exp, beta_out, se_exp, se_out, weights, nboot=1000):
    """
    Computes the SE for median methods.

    Arguments:

    beta_exp -- Vector of genetic effects on exposure

    beta_out -- Vector of genetic effects on outcome

    se_exp -- Standard errors of genetic effects on exposure

    se_out -- Standard errors of genetic effects on outcome

    weights -- Weights of each SNP

    nboot -- Number of bootstraps to calculate SE
    """
    med = []
    for i in range(1, nboot + 1):
        beta_exp_boot = np.random.normal(
            loc=beta_exp, scale=se_exp, size=len(beta_exp))
        beta_out_boot = np.random.normal(
            loc=beta_out, scale=se_out, size=len(beta_out))
        beta_iv_boot = beta_out_boot/beta_exp_boot
        med.append(weighted_median(beta_iv_boot, weights))

    return stdev(med)


def mr_simple_median(beta_exp, beta_out, se_exp, se_out, nboot=1000):
    """
    Computes the causal effect using simple median.

    Arguments:

    beta_exp -- Vector of genetic effects on exposure

    beta_out -- Vector of genetic effects on outcome

    se_exp -- Standard errors of genetic effects on exposure

    se_out -- Standard errors of genetic effects on outcome

    nboot -- Number of bootstraps to calculate SE

    Returns:

    {
        'effect: causal effect estimation,
        'se' : standard error of effect estimation
    }
    """
    n = len(beta_exp)
    beta_iv = beta_out/beta_exp
    effect = weighted_median(beta_iv, np.repeat(1/n, n))
    se = weighted_median_bootstrap(beta_exp, beta_out, se_exp,
                                   se_out, np.repeat(1/n, n), nboot)

    return {
        'effect': effect, 'se': se
    }


def mr_weighted_median(beta_exp, beta_out, se_exp, se_out, nboot=1000):
    """
    Computes the causal effect using weighted median.

    Arguments:

    beta_exp -- Vector of genetic effects on exposure

    beta_out -- Vector of genetic effects on outcome

    se_exp -- Standard errors of genetic effects on exposure

    se_out -- Standard errors of genetic effects on outcome

    nboot -- Number of bootstraps to calculate SE

    Returns:

    {
        'effect: causal effect estimation,
        'se' : standard error of effect estimation
    }
    """
    beta_iv = beta_out/beta_exp
    VBj = se_out**2/beta_exp**2 + beta_out**2 * se_exp**2/beta_exp**4
    effect = weighted_median(beta_iv, 1/VBj)
    se = weighted_median_bootstrap(
        beta_exp, beta_out, se_exp, se_out, 1/VBj, nboot)

    return {
        'effect': effect, 'se': se
    }


def mr_penalised_weighted_median(beta_exp, beta_out, se_exp, se_out, nboot=1000, penk=20):
    """
    Computes the causal effect using penalised weighted median.

    Arguments:

    beta_exp -- Vector of genetic effects on exposure

    beta_out -- Vector of genetic effects on outcome

    se_exp -- Standard errors of genetic effects on exposure

    se_out -- Standard errors of genetic effects on outcome

    nboot -- Number of bootstraps to calculate SE

    Returns:

    {
        'effect: causal effect estimation,
        'se' : standard error of effect estimation
    }
    """
    beta_iv = beta_out/beta_exp
    VBj = se_out**2/beta_exp**2 + beta_out**2 * se_exp**2/beta_exp**4
    weights = 1/VBj
    bwm = mr_weighted_median(beta_exp, beta_out, se_exp, se_out)
    penalty = chi2.cdf(weights*(beta_iv-bwm['effect'])**2, df=1)

    def pmin(x1, x2):
        arr = np.array([])
        for i in range(0, len(x1)):
            arr = np.append(arr, min(x1[i], x2[i]))

        return arr

    penalty_weights = weights*pmin(np.repeat(1, len(penalty)), penalty*penk)
    effect = weighted_median(beta_iv, penalty_weights)
    se = weighted_median_bootstrap(beta_exp, beta_out, se_exp,
                                   se_out, penalty_weights, nboot)

    return {
        'effect': effect, 'se': se
    }
