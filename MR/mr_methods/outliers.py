import numpy as np
import statsmodels.api as sm
import polars as pl
from MR.mr_methods.inverse_variance_weighted import mr_inverse_variance_weighted
from statistics import stdev


def cooks_distance(x, y):
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    influence = model.get_influence()
    cooks = influence.cooks_distance
    return {
        'cooks distance': cooks[0],
        'pvals': cooks[1]
    }


def mr_remove_outliers(data, method):
    beta_exp = data.select('beta_exp')
    beta_out = data.select('beta_out')
    if method == 'cooks distance':
        threshold = 4/len(data)
        result = cooks_distance(beta_exp.to_numpy(), beta_out.to_numpy())
        data_copy = data.clone()
        data_copy.insert_column(data_copy.shape[1], pl.Series(
            'cooks distance', result['cooks distance']))
        return data_copy.filter(pl.col('cooks distance') < threshold).drop('cooks distance')


def mr_presso(data, k=1000):
    # Global test
    beta_minus_j_list = []
    RSS_obs_list = []
    for j in range(len(data)):
        data_no_j = data.slice(0, j).vstack(data.slice(j + 1, len(data)))
        beta_minus_j = mr_inverse_variance_weighted(data_no_j.select(
            'beta_exp'), data_no_j.select('beta_out'), data_no_j.select('se_out'))['effect']
        beta_minus_j_list.append(beta_minus_j)
        RSS_obs_list.append((data.select('beta_out').row(j)[0] -
                            beta_minus_j * data.select('beta_exp').row(j)[0])**2)
    RSS_obs = sum(RSS_obs_list)
    # beta_exp_stdev = data.select('beta_exp').std().row(0)[0]
    # beta_out_stdev = data.select('beta_out').std().row(0)[0]
    RSS_exp_list = []
    for i in range(k):
        RSS_exp_k_list = []
        for j in range(0, len(data)):
            beta_exp_random = np.random.normal(
                loc=data.select('beta_exp').row(j)[0], scale=data.select('se_exp').row(j)[0], size=1)[0]
            beta_out_random = np.random.normal(
                loc=data.select('beta_exp').row(j)[0] * beta_minus_j_list[j], scale=data.select('se_out').row(j)[0], size=1)[0]
            # RSS_exp += (beta_out_random -
            #             beta_minus_j_list[j] * beta_exp_random)**2
            RSS_exp_k_list.append((beta_out_random -
                                   beta_minus_j_list[j] * beta_exp_random)**2)
        RSS_exp_list.append(RSS_exp_k_list)
    # print(RSS_obs)
    # print([sum(i) for i in RSS_exp_list])
    RSS_exp_df = pl.DataFrame({'RSS_exp': [sum(i) for i in RSS_exp_list]})
    print(f'Global Test: {len(RSS_exp_df.filter(
        pl.col('RSS_exp') > RSS_obs))/k}')

    # Outlier test
    pval_list = []
    for i in range(len(data)):
        pval = 0
        for j in range(k):
            if RSS_exp_list[j][i] > RSS_obs_list[i]:
                pval += 1
        pval_list.append(pval*len(data)/k)

    print(pval_list)
