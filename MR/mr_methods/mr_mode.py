from statistics import stdev


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

    def beta(beta_iv_in, se_beta_iv_in, phi):
        s = 0.9 * min(stdev(beta_iv_in), mad(beta_iv_in)) * \
            len(beta_iv_in)**(-1/5)
        weights = se_beta_iv_in**(-2)/(se_beta_iv_in**(-2)).sum()
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

    phi = 1000

    beta(beta_exp, se_exp, phi)
