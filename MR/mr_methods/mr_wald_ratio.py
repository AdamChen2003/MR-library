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
