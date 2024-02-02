from scipy import stats
import statsmodels.api as sm


def mr_inverse_variance_weighted(beta_exp, beta_out, se_out):
    """
    Computes the causal effect using inverse weighted variance.

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
    mod = sm.WLS(beta_out.to_numpy(), beta_exp.to_numpy(),
                 weights=1 / (se_out.to_numpy() ** 2)).fit()

    effect = mod.params[0]
    se = mod.bse[0]/min(1, mod.scale)
    pval = 2*(1-stats.norm.cdf(abs(effect/se)))

    return {
        'effect': effect, 'se': se, 'pval': pval
    }
    # effect = (beta_exp*beta_out * se_out**(-2)).sum() / \
    #     (beta_exp**2*se_out**(-2)).sum()
    # se = ((beta_exp**2*se_out**(-2)).sum())**(-0.5)

    # return {
    #     'effect': effect, 'se': se
    # }
