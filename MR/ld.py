import ieugwaspy as igd
import polars as pl

population = ['EUR', 'SAS', 'EAS', 'AFR', 'AMR', 'legacy']


def ld_matrix(rsids, pop_option=0):
    pop = population[pop_option]
    return igd.api_query(
        path='/ld/matrix',
        query={
            'rsid': rsids,
            'pop': pop
        },
        method='POST'
    )


def ld_clump(rsids, pvals, pthresh=5*10**(-8), r2=0.001, kb=5000, pop_option=0):
    pop = population[pop_option]
    pruned = igd.api_query(
        path='/ld/clump',
        query={
            'rsid': rsids,
            'pval': pvals,
            'pthresh': pthresh,
            'r2': r2,
            'kb': kb,
            'pop': pop
        },
        method='POST')
    pruned_rsids = pl.DataFrame({'rsid': pruned})
    return pruned_rsids
