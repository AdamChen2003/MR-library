import numpy as np
from scipy import stats
from scipy.optimize import minimize


def mr_maximum_likelihood(beta_exp, beta_out, se_exp, se_out):
    """
    Computes the causal effect using inverse weighted variance.

    Arguments:

    beta_exp -- Vector of genetic effects on exposure

    beta_out -- Vector of genetic effects on outcome

    se_exp -- Standard errors of genetic effects on exposure

    se_out -- Standard errors of genetic effects on outcome

    Returns:

    {
        'effect: MR estimate,
        'se': standard error of MR estimate,
        'pval': pval of MR estimation
    }
    """
    n = len(beta_exp)

    def log_likelihood(param):
        return 1/2 * ((beta_exp - param[0:n])**2/se_exp**2).sum() + 1/2 * ((beta_out - param[n] * param[0:n])**2 / se_out**2).sum()

    initial = np.append((beta_exp.to_numpy()), (beta_exp*beta_out /
                                                se_out**2).sum()/(beta_exp**2/se_out**2).sum())

    res = minimize(log_likelihood, initial)
    effect = res.x[n]
    se = np.sqrt(res.hess_inv[n, n])
    pval = 2 * (1-stats.norm.cdf(abs(effect)/se))

    return {
        'effect': effect, 'se': se, 'pval': pval
    }
