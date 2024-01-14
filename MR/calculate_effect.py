methods = ['iwv', 'wald ratio']


def calculate_effect(data, method):
    if method == 'ivw':
        effect = (data['beta_exp'] * data['beta_out'] * data['se_out']
                  ** -2).sum() / (data['beta_exp'] ** 2 * data['se_out'] ** -2).sum()

        se = ((data['beta_exp'] ** 2 *
              data['se_out'] ** -2).sum()) ** -0.5

    elif method == 'wald ratio':
        effect = (data['beta_out'] / data['beta_exp']).mean()
        se = (data['se_out'] / abs(data['beta_exp'])).mean()

    return {
        'effect': effect, 'se': se
    }
