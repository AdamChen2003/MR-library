def mr_inverse_variance_weighted(beta_exp, beta_out, se_out):
    """
    Computes the causal effect using inverse weighted variance.

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
    effect = (beta_exp*beta_out * se_out**(-2)).sum() / \
        (beta_exp**2*se_out**(-2)).sum()
    se = ((beta_exp**2*se_out**(-2)).sum())**(-0.5)

    return {
        'effect': effect, 'se': se
    }
