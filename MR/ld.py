import ieugwaspy as igd
import polars as pl

population = ['EUR', 'SAS', 'EAS', 'AFR', 'AMR', 'legacy']


def ld_matrix(rsids, pop='EUR'):
    """
    Returns linkage disequilibirum matrix between pairs of rsids using the ieugwaspy library which 
    acesses the IEUGWAS api.

    Arguments:

    rsids -- list of rsids to be examined

    pop -- Choice of demographic from the population list (default EUR)
    """
    return igd.api_query(
        path='/ld/matrix',
        query={
            'rsid': rsids,
            'pop': pop
        },
        method='POST'
    )


def ld_clump(rsids, pvals, pthresh=5*10**(-8), r2=0.001, kb=5000, pop='EUR'):
    """
    Returns a dataframe of pruned rsids based on linkage disequilibrium using the ieugwaspy library which 
    acesses the IEUGWAS api.

    Arguments:

    rsids -- A list of rsids to be examined

    pvals -- A list of p-values which are associated with the rsids

    pop -- A choice of demographic from the population list (default EUR)
    """
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
