# MR-library

Things to do:
* Proxying SNPs
* Provide better visualization of data
* Implement MR Presso
* Write software structure report


This package is intended to provide a python alternative to the already existing R counterpart here 
https://github.com/MRCIEU/TwoSampleMR/blob/master/R/mr.R.

All functionalities are located within the MR directory with the notebook providing a brief overview of usage.

## MR/harmonize.py
Data taken from GWAS studies may vary on the strand being reported as well as the alleles being used in reference to the effects of SNPs on the exposure or outcome. Hence, prior to performing any subsequent operations, it is vital to harmonize the data before proceeding.

More info can be found here https://mrcieu.github.io/TwoSampleMR/articles/harmonise.html#palindromic-snp-inferrable

### def harmonize(data, palindromic_action=1, palindromic_threshold=0.08)

The harmonize function essentially ensures each SNP is being viewed from the same strand and allele so that the reported effects are correct. Furthermore, there are available options to deal with palindromic SNPs which are problematic due to being unable to determine whether SNPs are using the incorrect strand or incorrect allele.

Arguments:

    data -- polars dataframe which contains the following columns:
        ea_exp: Effect allele for exposure
        oa_exp: Other allele for exposure
        ea_out: Effect allele for outcome
        oa_out: Other allele for outcome
        beta_out: Vector of genetic effects on outcome
        eaf_exp: Effect allele frequency for exposure
        eaf_out: Effect allele frequency for outcome

    palindromic_action -- int type which determines how to deal with palindromic SNPs (default = 1)
        1: Uses the palindromic_threshold to filter out palindromic SNPs
        2: Discards all palindromic SNPs. Ignores the provided 
            palindromic_threshold
        3: Does nothing in relation to palindromic SNPs

    palindromic_threshold -- float type specifying threshold to filter out palindromic SNPs. 
        A higher value results in stricter filtering (default = 0.08)

Returns: A harmonised polars dataframe with same columns as input data


## MR/ld.py
Typically, genetic associations are clustered together in loci that consist of many associated variants in one genomic region. In many cases, a locus only includes one independently associated, or causal, variant, while the associations of the other variants are due to the fact that they are genetically linked to the causal variant. This genetic linkage is expressed as linkage disequilibrium (LD), a measure of the extent of correlation between any two alleles. [1] Ideally, in the context of MR, independent SNPs are incorporated into calculations in order to not ‘double count’ contributions of particular variants. [2]

[1] https://kp4cd.org/help/LD_clumping
[2] https://mr-dictionary.mrcieu.ac.uk/term/ld/

### def ld_matrix(rsids, pop='EUR')
For a list of SNPs get the LD R values. These are presented relative to a specified reference allele.
Uses 1000 genomes reference data filtered to within-population MAF > 0.01 and only retaining SNPs.
Accesses the IEUGWAS api documented at http://gwasapi.mrcieu.ac.uk/docs/

Arguments:

    rsids -- list of rsids to be examined

    pop -- choice of demographic from the population list (default = 'EUR')
        possible options: 'EUR', 'SAS', 'EAS', 'AFR', 'AMR', 'legacy'


### def ld_clump(rsids, pvals, pthresh=5*10**(-8), r2=0.001, kb=10000, pop='EUR')
Perform clumping a specified set of rs IDs. Uses 1000 genomes reference data filtered to within-population MAF > 0.01 and only retaining SNPs. Accesses the IEUGWAS api documented at http://gwasapi.mrcieu.ac.uk/docs/

Arguments:

    rsids -- list of rsids to be examined

    pvals -- list of p-values which are associated with the rsids

    pthresh -- p-value threshold used to discard SNPs based on p-values (default = 5e-8)

    pop -- choice of demographic from the population list (default = 'EUR')
        possible options: 'EUR', 'SAS', 'EAS', 'AFR', 'AMR', 'legacy'
