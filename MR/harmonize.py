import polars as pl


def harmonize(data: pl.DataFrame, palindromic_action=1, palindromic_threshold=0.08):
    """
    Harmonizes the data based on this article
    https://mrcieu.github.io/TwoSampleMR/articles/harmonise.html#palindromic-snp-inferrable

    Arguments:

    data -- polars dataframe which contains the following columns:
        ea_exp: Effect allele for exposure
        oa_exp: Other allele for exposure
        ea_out: Effect allele for outcome
        oa_out: Other allele for outcome
        beta_out: Vector of genetic effects on outcome
        eaf_exp: Effect allele frequency for exposure
        eaf_out: Effect allele frequency for outcome

    palindromic_threshold -- threshold to filter out palindromic SNPs. 
        A higher value results in stricter filtering. (Default 0.05)


    Returns: A harmonised polars dataframe with same columns as input data
    """

    # Gathering all SNPs using fowards strand with matching effect and alternate alleles between exposure and outcome.
    forwards_same = data.filter(((pl.col('ea_exp') == pl.col(
        'ea_out')) & (pl.col('oa_exp') == pl.col('oa_out'))))

    # Gathering all SNPs using forwards strand with flipped effect and alternate alleles between exposure and outcome. The effect is then multiplied by -1.
    forwards_flipped = (
        data.filter(((pl.col('ea_exp') == pl.col('oa_out')) &
                     (pl.col('oa_exp') == pl.col('ea_out'))))
        # Flip the signs of the outcome effects
        .with_columns(
            pl.col('beta_out').mul(-1),
            pl.col('eaf_out').mul(-1).add(1)
        )
    ).rename({
        'oa_out': 'ea_out',
        'ea_out': 'oa_out',
    })

    # Flipping the outcome alleles of the remaining SNPs since the remaining valid SNPs must use the reverse strand.
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

    # Gathering SNPs from reverse strand which use the same alleles for exposure and outcome.
    reverse_same = (
        reverse.filter(((pl.col('ea_exp') == pl.col('ea_out')) &
                       (pl.col('oa_exp') == pl.col('oa_out'))))
    )

    # Gathering SNPs from reverse strand which flipped the effect and alternate alleles. We then multiply the effect by -1.
    reverse_flipped = (
        # Find all reversed cases
        reverse.filter(((pl.col('ea_exp') == pl.col('oa_out')) &
                       (pl.col('oa_exp') == pl.col('ea_out'))))
        # Flip the signs of the outcome effects
        .with_columns(
            pl.col('beta_out').mul(-1),
            pl.col('eaf_out').mul(-1).add(1)
        )
    ).rename({
        'oa_out': 'ea_out',
        'ea_out': 'oa_out',
    })

    # Combining all the different cases into one dataframe.
    total = pl.concat([forwards_same[sorted(forwards_same.columns)],
                       forwards_flipped[sorted(forwards_flipped.columns)],
                       reverse_same[sorted(reverse_same.columns)],
                       reverse_flipped[sorted(reverse_flipped.columns)]])

    # Dealing with palindromic SNPs. These are troublesome because we are unable to determine whether they
    # are reverse strand or using the different effect and alternate alleles. Thus, we have to infer from
    # the effect allele frequency.
    if palindromic_action == 1:
        # Keep the SNPs which meet the thresholds and discard the rest
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

        total = pl.concat([total, correct_palindromic, flipped_palindromic])

    elif palindromic_action == 2:
        # Get rid of all palindromic SNPs
        total = total.filter(
            ~(((pl.col('ea_exp') == 'a') & (pl.col('oa_exp') == 't')) |
              ((pl.col('ea_exp') == 't') & (pl.col('oa_exp') == 'a')) |
              ((pl.col('ea_exp') == 'g') & (pl.col('oa_exp') == 'c')) |
              ((pl.col('ea_exp') == 'c') & (pl.col('oa_exp') == 'g'))))

    elif palindromic_action != 3:
        raise ValueError('no such value for palindromic_action')

    # Getting rid of duplicate SNPs
    return total.unique(subset=['rsid'])
