import numpy as np
from sklearn.linear_model import LinearRegression


def mr_egger_regression(beta_exp, beta_out, se_out):
    """
    Computes the causal effect using egger regression.

    Arguments:

    beta_exp -- Vector of genetic effects on exposure

    beta_out -- Vector of genetic effects on outcome

    se_out -- Standard errors of genetic effects on outcome

    Returns:

    {
        'effect: causal effect estimation,
        'se' : standard error of effect estimation
    }
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
