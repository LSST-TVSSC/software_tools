from os import path
import config_utils

def filter_balance_by_science_case_group():
    # Fetch configuration:
    config_file = './footprint_maps/footprint_map_config.json'
    config = config_utils.read_config(config_file)

    # Time-domain science cases:
    time_domain_science = ['Magellenic Clouds', 'Galactic Bulge',
                            'Resolved Stellar Populations',
                            'Bonito Star Forming Regions', 'Galactic Pencilbeams',
                            'X-Ray Binaries']
    time_domain_map_codes = ["M","GB","C","B","P","X"]

    # Wide-area science cases:
    wide_area_science = ['Galactic Plane', 'Globular Clusters', 'Open Clusters',
                        'Star Forming Regions', "K2 Fields"]
    wide_area_map_codes = ["GP","G","O","Z","K2"]

    science_groups = {'Time Domain Science': time_domain_map_codes,
                      'Wide Area Science': wide_area_map_codes}

    # Calculate the average weighting per filter for the different science groups
    for science_group, map_codes in science_groups.items():
        filter_weights = calc_science_group_filter_balance(config, map_codes)

        print('\nFilter balance for '+science_group)
        print(filter_weights)

def calc_science_group_filter_balance(config, map_codes):
    """Function to calculate the average weight given
    to each filter, over a number of science cases in a group.
    Note that the map_weight is NOT used here - that is used only
    to weight regions for selection in the sky maps"""

    weights = {'u_weight': 0.0,
                'g_weight': 0.0,
                'r_weight': 0.0,
                'i_weight': 0.0,
                'z_weight': 0.0,
                'y_weight': 0.0}

    # Average over all science cases
    for code in map_codes:
        sci_config = config[code]

        for key in weights.keys():
            weights[key] += sci_config[key]

    # Normalize by the number of science cases.
    # Note that the sum over all bandpasses can still exceed 1 at this
    # point, if there are superpositions of filter weights for a given
    # filter between multiple science cases
    for key in weights.keys():
        weights[key] /= len(map_codes)

    # The scheduler's normalization schema is relative to r-band:
    r_weight = weights['r_weight']
    for key in weights.keys():
        weights[key] /= r_weight

    return weights


if __name__ == '__main__':
    filter_balance_by_science_case_group()
