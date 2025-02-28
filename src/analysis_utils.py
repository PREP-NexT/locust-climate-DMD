from plot_utils import convert_mode_vector2XY_lo, convert_mode_XY2vectors, convert_mode_vector2XY_c, compute_phase
# for moving average/uniform filter
from scipy.ndimage import uniform_filter
from data_utils import *


####################### correlation between locust and climatic modes (phase/mag) for a specific country ###################
#### moving average filter
def spatially_filter(magXY, filter_size=3, filter_mode='nearest'):
    """
    [V 2021.11.03] Moving average by size "filter_size".

    :param magXY: 2d array
    :param filter_size: default is 3
    :param filter_mode: default is "nearest"
    :return:
    """
    if ma.is_masked(magXY):
        magXY = np.asarray(magXY.filled(0))
    magXY[np.isnan(magXY)] = 0
    magXY = uniform_filter(magXY, size=filter_size, mode=filter_mode)
    magXYm = np.ma.masked_where(magXY == 0, magXY)
    return magXYm


def locust_cond_climatic_phase_sign_country_specific(mode_locust, period_locust, mode_climatic, period_climatic,
                                                     nosea_indices_lo, nosea_indices_c, locust_type, var_name,
                                                     cty_code=None, corr=None, locust_cond=None, filter_on=False,
                                                     verbose=True, binning_phase=False):
    """
    Complicated function for locust mag/phase on condition of signs of climatic phase, correlation between locust
    mag/phase and climatic mag/phase.

    [V 2022.11.30] Add conditional analysis on binning phase plot. (splitting: locust_cond='binning_phase')
    [V 2022.07.07] Add another input type without climatic mode & period.
    [V 2022.07.06] Add optional parameter verbose. If verbose=0, turn off print.
    [V 2021.11.03] Add optional filter_on method. If filter_on, cannot compute phase correlation
    [V 2021.09.01] Changed the phase KDE pdf to default bw_adjust
    [V 2021.07.07] Created the function.

    :param mode_locust: a mode vector for locust
    :param period_locust: period for locust
    :param mode_climatic: a mode vector for climatic variable
    :param period_climatic: period for climatic variable
    :param nosea_indices_lo: nosea_indices_lo from load_data_locust
    :param nosea_indices_c: nosea_indices_c from load_data_climatic
    :param locust_type: locust 5 types
    :param var_name: climatic 7 vars
    :param cty_code: This should be a list instead of a number. If None, do not do country specific analysis
                    (i.e., whole region will be considered).
    :param corr: 'Pearson', 'Spearman', or None, choose which correlation coefficient to use. If 'Spearman', also output
                the p-value; if locust_cond = 'complete', output the whole dataframe instead of the correlation coefficient
    :param locust_cond: 'mag', 'phase' or 'complete', deciding which to do for locust in conditional analysis.
                        If locust_cond = 'complete', output the whole dataframe
    :param filter_on: True or False. Choose whether to filter both the locust and climatic dynamic map before
                        calculating the correlation/do the analysis.
    :param binning_phase: If 'binning_phase'=True, return df with climate phase splitted in 12 bins.
    :return: print and figures
    """
    if mode_climatic is None:   # need to be emphasized while using
        # conditional case and correlation could not be computed, can only return df (without climatic info), i.e., locust_cond = 'complete'
        # no use for filter_on
        modeXY_lo, _ = convert_mode_vector2XY_lo(mode_locust, nosea_indices_lo)
        ### load country mask
        cowcode_XY, _, _, _ = load_mask_country(0.4, locust_type)
        ### make locust mode, climatic vars, cowcode_z the same dimensions
        mode_z = convert_mode_XY2vectors(modeXY_lo, modeXY_c=None, var_name=None, cowcode_XY=cowcode_XY, ctrlXY=None)
        mode_z_lo = mode_z[0]  # flattened w/o mask
        cowcode_z = mode_z[1]
        ## compute mag and phase
        mag_lo = np.absolute(mode_z_lo)
        phase_lo_yr = compute_phase(mode_z_lo, normalized=True)      # [0,1]

        ## summary
        df = pd.DataFrame(np.stack((mag_lo, phase_lo_yr, cowcode_z), axis=-1),
                          columns=[f'magnitude({locust_type} locust)', f'phase({locust_type} locust)[yr]', 'cowcode'])
        # make sure the column elements are numbers to be plotted
        df[f'magnitude({locust_type} locust)'] = df[f'magnitude({locust_type} locust)'].astype(float)
        df[f'phase({locust_type} locust)[yr]'] = df[f'phase({locust_type} locust)[yr]'].astype(float)
        df['cowcode'] = df['cowcode'].astype(float)

        if cty_code is not None:  # cty_code can be a list or a number
            df = df.loc[df['cowcode'].isin(cty_code)]
        if locust_cond == 'complete':
            return df

    else:
        #### modes with sea needs to be computed to mask by country mask
        modeXY_lo, _ = convert_mode_vector2XY_lo(mode_locust, nosea_indices_lo)
        modeXY_c = convert_mode_vector2XY_c(mode_climatic, nosea_indices_c, var_name, locust_type)

        ### load country mask
        cowcode_XY, _, _, _ = load_mask_country(0.4, locust_type)

        if filter_on:
            # filter magXY before converting it to 1d array,
            # filtering phase is meaningless, it will be all 0 to make the size the same.
            magXY_lo = np.absolute(modeXY_lo)
            magXY_c = np.absolute(modeXY_c)
            magXY_lo = spatially_filter(magXY_lo)
            magXY_c = spatially_filter(magXY_c)

            ### make locust mag, climatic var mag, cowcode_z the same dimensions
            mag_z = convert_mode_XY2vectors(magXY_lo, magXY_c, var_name, cowcode_XY, ctrlXY=None)
            mag_lo = mag_z[0]  # flattened w/o mask
            mag_c = mag_z[1]
            cowcode_z = mag_z[2]

            ## if mag is filtered, then no phase info will be used (i.e., phase cannot be filtered)
            phase_lo_yr = np.zeros(mag_lo.shape)
            phase_c_yr = np.zeros(mag_lo.shape)

        else:
            ### make locust mode, climatic vars, cowcode_z the same dimensions
            mode_z = convert_mode_XY2vectors(modeXY_lo, modeXY_c, var_name, cowcode_XY, ctrlXY=None)
            mode_z_lo = mode_z[0]  # flattened w/o mask
            mode_z_c = mode_z[1]
            cowcode_z = mode_z[2]

            ## compute mag and phase
            mag_lo = np.absolute(mode_z_lo)
            mag_c = np.absolute(mode_z_c)
            # phase_lo = np.angle(mode_z_lo)
            # phase_c = np.angle(mode_z_c)
            phase_lo_yr = compute_phase(mode_z_lo, normalized=True)  # [0,1]
            phase_c_yr = compute_phase(mode_z_c, normalized=True)  # [0,1]

        ## classify locust phase conditioned on climate phase
        phase_label = np.empty((len(phase_c_yr),), dtype='<U8')  # maximum string length is 8
        if binning_phase is False:
            phase_label[phase_c_yr >= 0] = 'Positive'
            phase_label[phase_c_yr < 0] = 'Negative'
        else:
            labels = np.arange(1, 13).astype(str)
            phase_label[(phase_c_yr >= 0) & (phase_c_yr < 1 / 12)] = labels[0]
            phase_label[(phase_c_yr >= 1 / 12) & (phase_c_yr < 2 / 12)] = labels[1]
            phase_label[(phase_c_yr >= 2 / 12) & (phase_c_yr < 3 / 12)] = labels[2]
            phase_label[(phase_c_yr >= 3 / 12) & (phase_c_yr < 4 / 12)] = labels[3]
            phase_label[(phase_c_yr >= 4 / 12) & (phase_c_yr < 5 / 12)] = labels[4]
            phase_label[(phase_c_yr >= 5 / 12) & (phase_c_yr < 6 / 12)] = labels[5]
            phase_label[(phase_c_yr >= 6 / 12) & (phase_c_yr < 7 / 12)] = labels[6]
            phase_label[(phase_c_yr >= 7 / 12) & (phase_c_yr < 8 / 12)] = labels[7]
            phase_label[(phase_c_yr >= 8 / 12) & (phase_c_yr < 9 / 12)] = labels[8]
            phase_label[(phase_c_yr >= 9 / 12) & (phase_c_yr <= 10 / 12)] = labels[9]
            phase_label[(phase_c_yr >= 10 / 12) & (phase_c_yr < 11 / 12)] = labels[10]
            phase_label[(phase_c_yr >= 11 / 12) & (phase_c_yr <= 1)] = labels[11]

        unique = np.unique(phase_label)
        palette = dict(zip(unique, sns.color_palette(n_colors=len(unique))))

        ## summary
        df = pd.DataFrame(np.stack((mag_lo, phase_lo_yr, mag_c, phase_c_yr, phase_label, cowcode_z), axis=-1),
                          columns=[f'magnitude({locust_type} locust)', f'phase({locust_type} locust)[yr]',
                                   f'magnitude({var_name})', f'phase({var_name})[yr]', f'{var_name} phase label',
                                   'cowcode'])
        # make sure the column elements are numbers to be plotted
        df[f'magnitude({locust_type} locust)'] = df[f'magnitude({locust_type} locust)'].astype(float)
        df[f'phase({locust_type} locust)[yr]'] = df[f'phase({locust_type} locust)[yr]'].astype(float)
        df[f'magnitude({var_name})'] = df[f'magnitude({var_name})'].astype(float)
        df[f'phase({var_name})[yr]'] = df[f'phase({var_name})[yr]'].astype(float)
        df['cowcode'] = df['cowcode'].astype(float)

        if cty_code is not None:  # cty_code can be a list or a number
            df = df.loc[df['cowcode'].isin(cty_code)]

        if locust_cond == 'mag':
            # ecdf
            plt.figure()
            sns.ecdfplot(data=df, x=f'magnitude({locust_type} locust)', hue=f'{var_name} phase label', palette=palette)
            plt.title("Cowcode: {} period: {} locust = {:.3f}, {:} = {:.3f}".format(cty_code, locust_type, period_locust,
                                                                                    var_name, period_climatic), size=15)
        elif locust_cond == 'phase':
            # density plot
            plt.figure()
            sns.kdeplot(data=df, x=f'phase({locust_type} locust)[yr]', hue=f'{var_name} phase label', common_norm=False,
                        linewidth=2, palette=palette)  # bw_adjust make kde estimation less smooth, bw_adjust=.2
            plt.title("Cowcode: {} period: {} locust = {:.3f}, {:} = {:.3f}".format(cty_code, locust_type, period_locust,
                                                                                    var_name, period_climatic), size=15)
            # histogram
            plt.figure()
            sns.histplot(data=df, x=f'phase({locust_type} locust)[yr]', hue=f'{var_name} phase label', stat="density",
                         bins=100, common_norm=False, element="step", linewidth=2, palette=palette)
            plt.title("Cowcode: {} period: {} locust = {:.3f}, {:} = {:.3f}".format(cty_code, locust_type, period_locust,
                                                                                    var_name, period_climatic), size=15)
        elif (locust_cond == 'complete') | (binning_phase is True):
            return df

        if corr == 'Pearson':
            mag_cor = np.corrcoef(df[f'magnitude({locust_type} locust)'], df[f'magnitude({var_name})'])[0, 1]
            phase_cor = np.corrcoef(df[f'phase({locust_type} locust)[yr]'], df[f'phase({var_name})[yr]'])[0, 1]
            if verbose:
                print('-----------------------------------------------------------------------------')
                print('\nCowcode {}:\nCorrelation between {} (period {:.3f}) and {} (period {:.3f}):\nMagnitude: {:.3f}'
                      '\nPhase:{:.3f}'
                      .format(cty_code, locust_type, period_locust, var_name, period_climatic, mag_cor, phase_cor))
            return mag_cor, phase_cor
        elif corr == 'Spearman':
            mag_cor, pvalue1 = stats.spearmanr(df[f'magnitude({locust_type} locust)'], df[f'magnitude({var_name})'])
            phase_cor, pvalue2 = stats.spearmanr(df[f'phase({locust_type} locust)[yr]'], df[f'phase({var_name})[yr]'])
            if verbose:
                print('-----------------------------------------------------------------------------')
                print('\nCowcode {}:\nCorrelation between {} (period {:.3f}) and {} (period {:.3f}):'
                      '\nMagnitude: {:.3f}, p-value {:.3f}\nPhase:{:.3f}, p-value {:.3f}'
                      .format(cty_code, locust_type, period_locust, var_name, period_climatic, mag_cor, pvalue1, phase_cor,
                              pvalue2))
            return mag_cor, pvalue1, phase_cor, pvalue2

