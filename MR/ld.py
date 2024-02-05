import ieugwaspy as igd
import polars as pl

population = ['EUR', 'SAS', 'EAS', 'AFR', 'AMR', 'legacy']


def ld_matrix(rsids, pop='EUR'):
    """
    For a list of SNPs get the LD R values. These are presented relative to a specified reference allele.
    Uses 1000 genomes reference data filtered to within-population MAF > 0.01 and only retaining SNPs.

    Accesses the IEUGWAS api documented at http://gwasapi.mrcieu.ac.uk/docs/

    Arguments:

    rsids -- list of rsids to be examined

    pop -- choice of demographic from the population list (default = 'EUR')
        possible options: 'EUR', 'SAS', 'EAS', 'AFR', 'AMR', 'legacy'
    """
    return igd.api_query(
        path='/ld/matrix',
        query={
            'rsid': rsids,
            'pop': pop
        },
        method='POST'
    )


def ld_clump(rsids, pvals, pthresh=5*10**(-8), r2=0.001, kb=10000, pop='EUR'):
    """
    Perform clumping a specified set of rs IDs.
    Uses 1000 genomes reference data filtered to within-population MAF > 0.01 and only retaining SNPs.

    Accesses the IEUGWAS api documented at http://gwasapi.mrcieu.ac.uk/docs/

    Arguments:

    rsids -- list of rsids to be examined

    pvals -- list of p-values which are associated with the rsids

    pthresh -- p-value threshold used to discard SNPs based on p-values (default = 5e-8)

    pop -- choice of demographic from the population list (default = 'EUR')
        possible options: 'EUR', 'SAS', 'EAS', 'AFR', 'AMR', 'legacy'
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
