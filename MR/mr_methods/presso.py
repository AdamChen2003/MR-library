import statsmodels.api as sm
import polars as pl


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

# import polars as pl
# import numpy as np


# def getRSS_LOO(beta_exp, beta_out, data, returnIV):
#     dataW = data.select([beta_out, beta_exp]) * np.sqrt(data['Weights'])
#     X = dataW.select(beta_exp)
#     Y = dataW.select(beta_out)

#     n = len(dataW)
#     CausalEstimate_LOO = []
#     for i in range(n):
#         X_minus_i = X[:i].vstack(X[i+1:]).to_numpy()
#         Y_minus_i = Y[:i].vstack(Y[i+1:]).to_numpy()
#         inv_term = np.linalg.inv(X_minus_i.transpose() @ X_minus_i)
#         CausalEstimate_LOO.append((
#             inv_term @ X_minus_i.transpose() @ Y_minus_i))

#     CausalEstimate_LOO = np.array(CausalEstimate_LOO)

#     if len(beta_exp) == 1:
#         RSS = ((beta_out - CausalEstimate_LOO * beta_exp)**2).sum()
#         # RSS = ((Y - CausalEstimate_LOO * X) ** 2).sum().sum()
#     else:
#         RSS = ((beta_out - (beta_exp * CausalEstimate_LOO.sum())**2))
#         # RSS = ((Y - (CausalEstimate_LOO * X).sum()) ** 2).sum().sum()

#     if returnIV:
#         return RSS, CausalEstimate_LOO
#     else:
#         return RSS


# def getRandomData(beta_exp, beta_out, se_exp, se_out, data):
#     mod_IVW = []
#     for i in range(len(data)):
#         formula = f"{beta_out} ~ {' + '.join(beta_exp)}"
#         weights = data['Weights'].to_numpy()
#         mod = pl.df(data.drop(i)).with_column(
#             pl.col("weights", pl.Series("f64", weights[:-1]))).with_column(
#             pl.col("response", pl.Series("f64", data[beta_out].values))).fit(formula, weight="weights")
#         mod_IVW.append(mod)

#     dataRandom = np.zeros((len(data), 3))
#     for i in range(len(data)):
#         exposure_mean = data.iloc[i][beta_exp].values
#         exposure_sd = data.iloc[i][se_exp]
#         outcome_sd = data.iloc[i][se_out]
#         predicted_outcome = mod_IVW[i].predict(data.iloc[i])[0]

#         exposure_values = np.random.normal(
#             exposure_mean, exposure_sd, size=len(data))
#         outcome_values = np.random.normal(predicted_outcome, outcome_sd)

#         dataRandom[:, 0] = exposure_values
#         dataRandom[:, 1] = outcome_values
#         dataRandom[:, 2] = data['Weights'].to_numpy()

#     dataRandom = pl.DataFrame(
#         dataRandom, columns=[beta_exp, beta_out, "Weights"])
#     return dataRandom


# def mr_presso(beta_exp, beta_out, se_exp, se_out, OUTLIERtest=False, DISTORTIONtest=False, SignifThreshold=0.05, NbDistribution=1000):
#     data = pl.DataFrame([beta_exp, beta_out, se_exp, se_out])

#     # Multiply beta_out and beta_exp columns by the sign of the first value in beta_exp

#     data.with_columns(
#         pl.col('beta_out').mul(np.sign(beta_exp[0])),
#         pl.col('beta_exp').mul(np.sign(beta_exp[0]))
#     )

#     # Calculate Weights
#     data = data.with_columns((1/pl.col('se_out')**2).alias('Weights'))

#     print(data.head)
#     RSSobs = getRSS_LOO(beta_exp, beta_out, data, OUTLIERtest)

#     print(RSSobs)

#     random_data = []
#     for _ in range(NbDistribution):
#         random_data.append(getRandomData(
#             beta_exp, beta_out, se_exp, se_out, data))

#     RSSexp = []
#     for rd in random_data:
#         RSSexp.append(getRSS_LOO(beta_exp, beta_out, se_exp,
#                       se_out, rd, returnIV=OUTLIERtest))

#     if OUTLIERtest:
#         GlobalTest = {'RSSobs': RSSobs[0], 'Pvalue': np.sum(
#             RSSexp[0] > RSSobs[0]) / NbDistribution}
#     else:
#         GlobalTest = {'RSSobs': RSSobs[0], 'Pvalue': np.sum(
#             RSSexp > RSSobs[0]) / NbDistribution}

#     print(GlobalTest)
