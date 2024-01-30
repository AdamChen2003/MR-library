from statistics import stdev
import numpy as np
from scipy.stats import chi2
from sklearn.neighbors import KernelDensity
from MR.mr_methods.utils import pmin


def mr_mode(beta_exp, beta_out, se_exp, se_out, method, phi=1, nboot=1000, penk=20):
    """
    Performs simple or weighted mode.

    Arguments:

    beta_exp -- Vector of genetic effects on exposure

    beta_out -- Vector of genetic effects on outcome

    se_exp -- Standard errors of genetic effects on exposure

    se_out -- Standard errors of genetic effects on outcome

    phi -- Bandwidth parameter for density estimation

    nboot -- Number of bootstraps to calculate SE

    Returns:

    {
        'effect: causal effect estimation,
        'se' : standard error of effect estimation
    }
    """
    def mad(data):
        """
        Computes median absolute deivation for provided data
        """
        return (abs(data-data.mean())).sum()/len(data)

    def beta(beta_iv_in, se_beta_iv_in, phi=1):
        s = 0.9*min(stdev(beta_iv_in), mad(beta_iv_in))*len(beta_iv_in)**(-1/5)
        weights = 1/(se_beta_iv_in**(2)*(1/se_beta_iv_in**(2)).sum())
        # for cur_phi in phi:
        h = max(0.00000001, s*phi)
        X = beta_iv_in.reshape((-1, 1))
        kde = KernelDensity(kernel='gaussian',
                            bandwidth=h).fit(X, sample_weight=weights)

        X = X.copy()
        density_scores = kde.score_samples(X)
        return X[density_scores.argmax()][0]

    def boot(beta_iv_in, se_beta_iv_in):
        beta_boot = []
        for _ in range(1, nboot + 1):
            beta_iv_boot = np.random.normal(
                loc=beta_iv_in, scale=se_beta_iv_in, size=len(beta_iv_in))
            if method == 'simple':
                beta_boot.append(
                    beta(beta_iv_boot, np.repeat(1, len(beta_iv)), phi))
            elif method == 'weighted':
                beta_boot.append(beta(beta_iv_boot, se_beta_iv_in, phi))
            elif method == 'penalised':
                weights = 1/se_beta_iv**2
                penalty = chi2.cdf(weights*(beta_iv - beta_weighted)**2, df=1)
                penalty_weights = weights * \
                    pmin(np.repeat(1, len(penalty)), penalty*penk)
                beta_boot.append(
                    beta(beta_iv_boot, 1/penalty_weights**0.5, phi))
        return np.array(beta_boot)

    beta_iv = beta_out/beta_exp
    se_beta_iv = ((se_out**2/beta_exp**2) +
                  ((beta_out**2)*(se_exp**2))/beta_exp**4)**0.5
    if method == 'simple':
        effect = beta(beta_iv.to_numpy(), np.repeat(1, len(beta_iv)), phi)
    elif method == 'weighted':
        effect = beta(beta_iv.to_numpy(), se_beta_iv, phi)
    elif method == 'penalised':
        beta_weighted = beta(beta_iv.to_numpy(), se_beta_iv, phi)
        weights = 1/se_beta_iv**2
        penalty = chi2.cdf(weights*(beta_iv - beta_weighted)**2, df=1)
        penalty_weights = weights * \
            pmin(np.repeat(1, len(penalty)), penalty*penk)
        effect = beta(beta_iv.to_numpy(), 1/penalty_weights**0.5, phi)

    return {
        'effect': effect,
        # 'se': 0
        'se': mad(boot(beta_iv, se_beta_iv))
    }


def mr_simple_mode(beta_exp, beta_out, se_exp, se_out, phi=1, nboot=1000):
    """
    Computes the causal effect using simple mode.

    Arguments:

    beta_exp -- Vector of genetic effects on exposure

    beta_out -- Vector of genetic effects on outcome

    se_exp -- Standard errors of genetic effects on exposure

    se_out -- Standard errors of genetic effects on outcome

    phi -- Bandwidth parameter for density estimation

    nboot -- Number of bootstraps to calculate SE

    Returns:

    {
        'effect: causal effect estimation,
        'se' : standard error of effect estimation
    }
    """
    return mr_mode(beta_exp, beta_out, se_exp, se_out, 'simple', phi, nboot)


def mr_weighted_mode(beta_exp, beta_out, se_exp, se_out, phi=1, nboot=1000):
    """
    Computes the causal effect using weighted mode.

    Arguments:

    beta_exp -- Vector of genetic effects on exposure

    beta_out -- Vector of genetic effects on outcome

    se_exp -- Standard errors of genetic effects on exposure

    se_out -- Standard errors of genetic effects on outcome

    phi -- Bandwidth parameter for density estimation

    nboot -- Number of bootstraps to calculate SE

    Returns:

    {
        'effect: causal effect estimation,
        'se' : standard error of effect estimation
    }
    """
    return mr_mode(beta_exp, beta_out, se_exp, se_out, 'weighted', phi, nboot)


def mr_penalised_weighted_mode(beta_exp, beta_out, se_exp, se_out, phi=1, nboot=1000, penk=20):
    """
    Computes the causal effect using weighted mode.

    Arguments:

    beta_exp -- Vector of genetic effects on exposure

    beta_out -- Vector of genetic effects on outcome

    se_exp -- Standard errors of genetic effects on exposure

    se_out -- Standard errors of genetic effects on outcome

    phi -- Bandwidth parameter for density estimation

    nboot -- Number of bootstraps to calculate SE

    penk -- Constant term in penalisation

    Returns:

    {
        'effect: causal effect estimation,
        'se' : standard error of effect estimation
    }
    """
    return mr_mode(beta_exp, beta_out, se_exp, se_out, 'penalised', phi, nboot, penk)
