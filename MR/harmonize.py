import polars as pl


def harmonize(data: pl.DataFrame):
    """
    Harmonizes the data based on this article
    https://mrcieu.github.io/TwoSampleMR/articles/harmonise.html#palindromic-snp-inferrable
    """
    forwards_same = data.filter(((pl.col('ea_exp') == pl.col(
        'ea_out')) & (pl.col('oa_exp') == pl.col('oa_out'))))

    forwards_flipped = (
        data.filter(((pl.col('ea_exp') == pl.col('oa_out')) &
                     (pl.col('oa_exp') == pl.col('ea_out'))))
        # Flip the signs of the outcome effects
        .with_columns(
            pl.col('beta_out').mul(-1),
            pl.col('ea_freq').mul(-1).add(1)
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
            pl.col('ea_freq').mul(-1).add(1)
        )
    )

    # Combining all SNPs
    total = pl.concat([forwards_same, forwards_flipped,
                       reverse_same, reverse_flipped])

    # Removing all palindromic SNPs
    total = total.filter(
        ~(((pl.col('ea_exp') == 'a') & (pl.col('oa_exp') == 't')) |
          ((pl.col('ea_exp') == 't') & (pl.col('oa_exp') == 'a')) |
            ((pl.col('ea_exp') == 'g') & (pl.col('oa_exp') == 'c')) |
            ((pl.col('ea_exp') == 'c') & (pl.col('oa_exp') == 'g')))
    )

    return total
