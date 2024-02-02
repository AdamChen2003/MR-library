import scipy.stats as stats


def mr_wald_ratio(beta_exp, beta_out, se_out):
    """
    Computes the causal effect using the Wald ratio method.

    Arguments:

    beta_exp -- Vector of genetic effects on exposure

    beta_out -- Vector of genetic effects on outcome

    se_out -- Standard errors of genetic effects on outcome

    Returns:

    {
        'effect: MR estimate,
        'se': standard error of MR estimate,
        'pval': pval of MR estimation
    }
    """
    effect = (beta_out/beta_exp)
    se = (se_out/abs(beta_exp))
    pval = 2*stats.norm.sf(abs(effect)/se)

    return {
        'effect': effect, 'se': se, 'pval': pval
    }
