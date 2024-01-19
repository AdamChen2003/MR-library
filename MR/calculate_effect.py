import polars as pl
from scipy.optimize import minimize
import numpy as np
from statistics import stdev
from scipy.stats import chi2
from sklearn.linear_model import LinearRegression


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


def mr_inverse_variance_weighted(beta_exp, beta_out, se_out):
    """
    Computes the causal effect using inverse weighted variance.

    Arguments:

    beta_exp -- Vector of genetic effects on exposure

    beta_out -- Vector of genetic effects on outcome

    se_out -- Standard errors of genetic effects on outcome
    """
    effect = (beta_exp * beta_out * se_out ** -2).sum() / \
        (beta_exp ** 2 * se_out ** -2).sum()
    se = ((beta_exp ** 2 * se_out ** -2).sum()) ** -0.5

    return {
        'effect': effect, 'se': se
    }


def mr_wald_ratio(beta_exp, beta_out, se_out):
    """
    Computes the causal effect using the Wald ratio method.

    Arguments:

    beta_exp -- Vector of genetic effects on exposure

    beta_out -- Vector of genetic effects on outcome

    se_out -- Standard errors of genetic effects on outcome
    """
    effect = (beta_out/beta_exp).mean()
    se = (se_out/abs(beta_exp)).mean()

    return {
        'effect': effect, 'se': se
    }


def mr_maximum_likelihood(beta_exp, beta_out, se_exp, se_out):
    """
    Computes the causal effect using inverse weighted variance.

    Arguments:

    beta_exp -- Vector of genetic effects on exposure

    beta_out -- Vector of genetic effects on outcome

    se_exp -- Standard errors of genetic effects on exposure

    se_out -- Standard errors of genetic effects on outcome
    """
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
    """
    Computes the causal effect using simple median.

    Arguments:

    beta_exp -- Vector of genetic effects on exposure

    beta_out -- Vector of genetic effects on outcome

    se_exp -- Standard errors of genetic effects on exposure

    se_out -- Standard errors of genetic effects on outcome
    """
    n = len(beta_exp)
    b_iv = beta_out/beta_exp
    effect = weighted_median(b_iv, np.repeat(1/n, n))
    se = weighted_median_se(beta_exp, beta_out, se_exp,
                            se_out, np.repeat(1/n, n))

    return {
        'effect': effect, 'se': se
    }


def mr_weighted_median(beta_exp, beta_out, se_exp, se_out):
    """
    Computes the causal effect using weighted median.

    Arguments:

    beta_exp -- Vector of genetic effects on exposure

    beta_out -- Vector of genetic effects on outcome

    se_exp -- Standard errors of genetic effects on exposure

    se_out -- Standard errors of genetic effects on outcome
    """
    b_iv = beta_out/beta_exp
    VBj = se_out**2/beta_exp**2 + \
        beta_out**2 * se_exp**2/beta_exp**4
    effect = weighted_median(b_iv, 1/VBj)
    se = weighted_median_se(beta_exp, beta_out, se_exp, se_out, 1/VBj)

    return {
        'effect': effect, 'se': se
    }


def mr_penalised_weighted_median(beta_exp, beta_out, se_exp, se_out):
    """
    Computes the causal effect using penalised weighted median.

    Arguments:

    beta_exp -- Vector of genetic effects on exposure

    beta_out -- Vector of genetic effects on outcome

    se_exp -- Standard errors of genetic effects on exposure

    se_out -- Standard errors of genetic effects on outcome
    """
    beta_iv = beta_out/beta_exp
    beta_ivw = (beta_out*beta_exp*se_out**(-2)).sum() / \
        (beta_exp**2*se_out**(-2)).sum()
    VBj = se_out**2/beta_exp**2 + \
        beta_out**2 * se_exp**2/beta_exp**4
    weights = 1/VBj
    bwm = mr_weighted_median(beta_exp, beta_out, se_exp, se_out)
    penalty = chi2.cdf(weights*beta_iv-bwm['effect']**2, df=1)
    penalty_weights = penalty*weights
    effect = weighted_median(beta_iv, penalty_weights)
    se = weighted_median_se(beta_exp, beta_out, se_exp,
                            se_out, penalty_weights)

    return {
        'effect': effect, 'se': se
    }


def mr_mode(beta_exp, beta_out, se_exp, se_out):
    """
    Arguments:

    beta_exp -- Vector of genetic effects on exposure

    beta_out -- Vector of genetic effects on outcome

    se_exp -- Standard errors of genetic effects on exposure

    se_out -- Standard errors of genetic effects on outcome
    """
    def mad(data):
        """
        Computes median absolute deivation for provided data
        """
        return (data-data.mean()).sum()/len(data)

    def beta(beta_iv_in, se_beta_inv_in, phi):
        s = 0.9 * min(stdev(beta_iv_in), mad(beta_iv_in)) * \
            len(beta_iv_in)**(-1/5)
        weights = se_beta_inv_in**(-2)/(se_beta_inv_in**(-2)).sum()
        beta = []
        for cur_phi in range(0, phi):
            h = max(0.00000001, s*cur_phi)
            from rpy2 import robjects
            from rpy2.robjects.packages import importr
            from rpy2.robjects import vectors
            import numpy as np

            stats = importr("stats")

            column = vectors.IntVector([63, 45, 47, 28, 59, 28, 59])

            output = stats.density(column, adjust=1)

            x = np.array(output[0])
            y = np.array(output[1])
            beta.append(x[y == max(y)])

        print(beta)
        return beta

    beta(beta_exp, beta_out, se_exp)


def mr_egger_regression(beta_exp, beta_out, se_out):
    """
    Computes the causal effect using egger regression.

    Arguments:

    beta_exp -- Vector of genetic effects on exposure

    beta_out -- Vector of genetic effects on outcome

    se_out -- Standard errors of genetic effects on outcome
    """
    def sign0(x):
        x[x == 0] = -1
        return np.sign(x)

    # to_flip = sign0(beta_exp) == -1
    beta_out = (beta_out * sign0(beta_exp)).to_numpy().reshape((-1, 1))
    beta_exp = abs(beta_exp).to_numpy().reshape((-1, 1))
    model = LinearRegression().fit(beta_exp,
                                   beta_out,
                                   sample_weight=se_out**(-2))

    # Code for the following is drawn from
    # https://gist.github.com/grisaitis/cf481034bb413a14d3ea851dab201d31
    def get_se():
        """
        Calculates the standard error of the coefficients of a fitted linear model
        """
        N = len(beta_exp)
        p = 2
        X_with_intercept = np.empty(shape=(N, p), dtype=float)
        X_with_intercept[:, 0] = 1
        X_with_intercept[:, 1:p] = beta_exp
        predictions = model.predict(beta_exp)
        residuals = beta_out - predictions
        residual_sum_of_squares = residuals.T @ residuals
        sigma_squared_hat = residual_sum_of_squares[0, 0] / (N - p)
        var_beta_hat = np.linalg.inv(
            X_with_intercept.T @ X_with_intercept) * sigma_squared_hat
        return [var_beta_hat[p_, p_] ** 0.5 for p_ in range(p)]

    effect = model.coef_[0][0]
    # se = get_se()[1] / min(1, model.sigma)
    se = get_se()[1]

    return {
        'effect': effect, 'se': se
    }
