import polars as pl


def harmonize(data: pl.DataFrame, palindromic_threshold=0.05):
    """
    Harmonizes the data based on this article
    https://mrcieu.github.io/TwoSampleMR/articles/harmonise.html#palindromic-snp-inferrable

    Arguments:

    data -- A polars dataframe which contains the following columns:
        ea_exp: Effect allele for exposure
        oa_exp: Other allele for exposure
        ea_out: Effect allele for outcome
        oa_out: Other allele for outcome
        beta_out: Vector of genetic effects on outcome
        eaf_exp: Effect allele frequency for exposure
        eaf_out: Effect allele frequency for outcome

    palindromic_threshold -- A threshold to filter out palindromic SNPs. 
        A higher value results in stricter filtering. Default value is 0.05
    """
    forwards_same = data.filter(((pl.col('ea_exp') == pl.col(
        'ea_out')) & (pl.col('oa_exp') == pl.col('oa_out'))))

    forwards_flipped = (
        data.filter(((pl.col('ea_exp') == pl.col('oa_out')) &
                     (pl.col('oa_exp') == pl.col('ea_out'))))
        # Flip the signs of the outcome effects
        .with_columns(
            pl.col('beta_out').mul(-1),
            pl.col('eaf_out').mul(-1).add(1)
        )
    )

    # Find cases where alleles don't match
    reverse = (data.filter(~(((pl.col('ea_exp') == pl.col('ea_out')) & (pl.col('oa_exp') == pl.col('oa_out'))) |
                             (((pl.col('ea_exp') == pl.col('oa_out')) & (pl.col('oa_exp') == pl.col('ea_out'))))))
               # Flipping the alleles
               .with_columns(pl.col('ea_out').str.replace('a', 't'))
               .with_columns(pl.col('ea_out').str.replace('t', 'a'))
               .with_columns(pl.col('ea_out').str.replace('g', 'c'))
               .with_columns(pl.col('ea_out').str.replace('c', 'g'))
               .with_columns(pl.col('oa_out').str.replace('a', 't'))
               .with_columns(pl.col('oa_out').str.replace('t', 'a'))
               .with_columns(pl.col('oa_out').str.replace('g', 'c'))
               .with_columns(pl.col('oa_out').str.replace('c', 'g'))
               )

    reverse_same = (
        reverse.filter(((pl.col('ea_exp') == pl.col('ea_out')) &
                       (pl.col('oa_exp') == pl.col('oa_out'))))
    )

    reverse_flipped = (
        # Find all reversed cases
        reverse.filter(((pl.col('ea_exp') == pl.col('oa_out')) &
                       (pl.col('oa_exp') == pl.col('ea_out'))))
        # Flip the signs of the outcome effects
        .with_columns(
            pl.col('beta_out').mul(-1),
            pl.col('eaf_out').mul(-1).add(1)
        )
    )

    # Combining all SNPs
    total = pl.concat([forwards_same, forwards_flipped,
                      reverse_same, reverse_flipped])

    # Dealing with palindromic SNPs
    palindromic = total.filter(
        ((pl.col('ea_exp') == 'a') & (pl.col('oa_exp') == 't')) |
        ((pl.col('ea_exp') == 't') & (pl.col('oa_exp') == 'a')) |
        ((pl.col('ea_exp') == 'g') & (pl.col('oa_exp') == 'c')) |
        ((pl.col('ea_exp') == 'c') & (pl.col('oa_exp') == 'g')))

    total = total.filter(
        ~(((pl.col('ea_exp') == 'a') & (pl.col('oa_exp') == 't')) |
          ((pl.col('ea_exp') == 't') & (pl.col('oa_exp') == 'a')) |
          ((pl.col('ea_exp') == 'g') & (pl.col('oa_exp') == 'c')) |
          ((pl.col('ea_exp') == 'c') & (pl.col('oa_exp') == 'g'))))

    # For all SNPs where effect allele freq is greater than threshold + 0.5 and minor allele freq is less than 0.5 - threshold, flip the beta effect.
    # For all SNPs where both effect ellele freq and minor allele freq are greater than 0.5 + threshold or less than 0.5 - threshold, change nothing.
    # Otherwise, discard the SNP.

    correct_palindromic = palindromic.filter((((pl.col('eaf_exp') > 0.5 + palindromic_threshold) & (pl.col('eaf_out') > 0.5 + palindromic_threshold)) |
                                              ((pl.col('eaf_exp') < 0.5 - palindromic_threshold) & (pl.col('eaf_out') < 0.5 - palindromic_threshold))))

    flipped_palindromic = (palindromic.filter(((pl.col('eaf_exp') > 0.5 + palindromic_threshold) & (pl.col('eaf_out') < 0.5 - palindromic_threshold)))
                           .with_columns(
        # Flip the signs of the outcome effects
        pl.col('beta_out').mul(-1))
    )

    return pl.concat([total, correct_palindromic, flipped_palindromic])
