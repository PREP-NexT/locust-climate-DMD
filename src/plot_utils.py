import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
import matplotlib.ticker as mticker
import matplotlib.patheffects as PathEffects

from numpy import pi
from scipy import io
import os
import cartopy.crs as ccrs
import cartopy
import cartopy.io.shapereader as shpreader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def plot_settings():
    """
    Change font name to 'Myriad Pro' by running plot_settings() in the main content.
    """
    font = {'family': 'Myriad Pro'}
    mpl.rc('font', **font)
    return


def highlight_country(country_name: "a list", ax: "the axis to be plotted",
                      show_hl: "whether to print highlighted country names", linewidth=1):
    """
        Highlight the top 10 hotspot countries when plotting the map.
    """
    shpfilename = shpreader.natural_earth(resolution='110m',
                                          category='cultural',
                                          name='admin_0_countries')
    reader = shpreader.Reader(shpfilename)
    countries = reader.records()

    for country in countries:
        for name in country_name:
            if country.attributes['ADM0_A3'] == name:
                ax.add_geometries(country.geometry, ccrs.PlateCarree(), zorder=3,
                                  facecolor='none', edgecolor='#595959',  # '#484719' dark green in Fig.1
                                  label=country.attributes['ADM0_A3'], linewidth=linewidth)
                x = country.geometry.centroid.x
                y = country.geometry.centroid.y
                if show_hl:
                    ax.text(x, y, country.attributes['NAME'].upper(), color='w', size=10, ha='center', va='center',
                            transform=ccrs.PlateCarree(),
                            path_effects=[PathEffects.withStroke(linewidth=2, foreground="k", alpha=.8)], zorder=4)


### convert locust mode vector to mode_XY
def convert_mode_vector2XY_lo(mode_locust, nosea_indices_lo):
    """
    Convert mode vector to mode_XY, where sea pixels and locust_mag=0 is masked

    [V 2021.09.02] Fixed the function to make it applicable to both 'mag' and 'mode'.
    [V 2021.07.06] Created the function to use in plot_mag, plot_phase, and conditional analysis.

    :param mode_locust: mode vector
    :param nosea_indices_lo: from load_data_locust
    :return: mode_XYm_wo0_lo: masked, XY scale
             mask_locust
    """
    mode_wsea = np.empty(len(nosea_indices_lo), dtype=type(mode_locust[0]))
    mode_wsea[:] = np.NaN

    ## add the sea pixels as nan
    k = 0
    for i in range(len(mode_wsea)):
        if nosea_indices_lo[i]:
            mode_wsea[i] = mode_locust[k]
            k += 1

    # reshape to a map
    mode_XY = mode_wsea.reshape((416, 220)).T  # reshape to (416,220) will first fill in 220 dim
    # do not plot where mag = 0
    mode_XY_wo0 = np.array(mode_XY)
    mode_XY_wo0[mode_XY_wo0 == 0] = np.nan
    # mask sea and mag=0 pixels
    mode_XYm_wo0 = np.ma.masked_invalid(mode_XY_wo0)

    ## save mask_locust for future
    mask_locust = np.isnan(mode_XY_wo0)

    return mode_XYm_wo0, mask_locust


def plot_mag_lo(Xlon, Ylat, nosea_indices, locust_type, mag=None, mode=None,
                save_lo_mask=False, normalization=True, mag_diff=False):
    """
        Plot the magnitude derived from dynamic patterns.

    [V 2022.03.08] Change the input associated with the title.
    [V 2022.01.18] Minor changes to the font size and title, add country borders, deactivate the gridlines
    [V 2021.12.15]  Added input "normalization" and "mag_diff" for plotting El Nino and La Nina mag difference.
    [V 2021.07.06]  Used convert_mode_vector2XY_lo() function
    [V 2021.07.03]  deleted log colormap, changed to colormap with fixed bounds (normalization after masking off mag=0)
    [V 2021.07.01]  Normalized the magnitude to [0,1] before plotting (normalization before masking off mag=0 for log scale)
    [V 2021.06.25]  Changed to log colormap, deleted extend 'max'

    :param locust_type: locust type.
    :param save_lo_mask: whether to save a mask for locust area.
    :param normalization: whether to normalize the magnitude before plotting.
    :param mag_diff: whether to plot El Nino and La Nina mag difference
    :param mode: mode vector, e.g. Phi_sub[:,0] is the first mode, provide either "mode" or "mag".
    :param mag: computed mag (column vector), provide either "mode" or "mag".
    :param Ylat: shape(220,416) the same lat in each row, available from load_data_locust().
    :param Xlon: shape(220,416) the same lon in each column, available from load_data_locust().
    :param nosea_indices: Available from load_data_locust().
    :return: save mask_locust for climatic var visualization, show the figure of mode magnitude
    """
    # convert the long array to 2d array with nan nosea data
    if mag is None:
        mode_XYm, mask_locust = convert_mode_vector2XY_lo(mode, nosea_indices)
        magXYm = np.absolute(mode_XYm)
    else:
        magXYm, mask_locust = convert_mode_vector2XY_lo(mag, nosea_indices)

    # normalization
    if normalization:
        magXYmn = (magXYm - np.min(magXYm)) / (np.max(magXYm) - np.min(magXYm))  # min-max normalization
    else:
        magXYmn = magXYm

    ## preparation for color map and color bar
    # bounds = [0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 1e-1, 1]
    bounds = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    cmap = plt.get_cmap('OrRd', len(bounds) - 1)  # original
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    if mag_diff:
        # bounds = [-1e-2, -1e-3, -1e-4, -1e-5, -5e-6, -1e-6, 0, 1e-6, 5e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        bounds = [-1e-2, -1e-3, -1e-4, -1e-5, -1e-6, -1e-7, 0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        cmap = plt.get_cmap('RdBu_r', len(bounds) + 1)  # 'seismic'
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')

    # ## log norm
    # norm = mpl.colors.LogNorm(vmin=magXYm_wo0.min(), vmax=magXYm_wo0.max())

    ## plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    extent = [-18, 86, -4, 51]  # lon_min, lon_max, lat_min, lat_max
    # central_lon, central_lat = np.mean(extent[:2]), np.mean(extent[2:]) # not used
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.coastlines()
    # country borders
    ax.add_feature(cartopy.feature.BORDERS, alpha=1, zorder=1, edgecolor='#D0D0D0')
    top10_a3 = ['MRT', 'SAU', 'IND', 'SDN', 'PAK', 'KEN', 'YEM', 'NER', 'DZA']  # Method 2, , 'MAR'
    highlight_country(top10_a3, ax, show_hl=False)

    # # adjust the gridlines
    # gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    # gl.xlabels_top = False
    # gl.ylabels_right = False
    # gl.xformatter = LONGITUDE_FORMATTER
    # gl.yformatter = LATITUDE_FORMATTER
    # gl.xlabel_style = {'size': 10, 'color': 'gray'}
    # gl.ylabel_style = {'size': 10, 'color': 'gray'}

    # title
    # ax.set_title("Magnitude of {}: period = {:.3f} yr".format(locust_type, period), fontsize=12)
    fontsize = 20
    if locust_type == 'all':
        add_on = "1985-2020"
    elif locust_type == 'all_1st18y':
        add_on = "1985-2002"
    elif locust_type == 'all_2nd18y':
        add_on = "2003-2020"
    elif locust_type == "El":
        add_on = "El Nino"
    elif locust_type == "La":
        add_on = "La Nina"
    elif mag_diff:
        add_on = "El Nino vs La Nina"
    else:
        add_on = locust_type
    ax.set_title("Magnitude, {}".format(add_on), fontsize=fontsize)

    # pcolor: do not need to change the z variable every time changing the mode to plot
    cs = ax.pcolor(Xlon, Ylat, magXYmn, shading='auto', cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), zorder=3)
    # binary; Reds; bwr;'OrRd'
    # set up the colorbar
    cbar = fig.colorbar(cs, orientation="horizontal", format='%.0e', ticks=bounds)    # orientation="horizontal", format='%.1e'
    cbar.ax.tick_params(labelsize=15)
    # cbar.ax.set_xlabel("Magnitude", fontsize=15)

    if mag_diff:
        # hide some tick labels for colorbar
        for label in cbar.ax.xaxis.get_ticklabels()[1::2]:  # use [1::2] to hide very other label
            label.set_visible(False)

    plt.show()

    # save mask_locust
    # The mask_locust has been tested the same for every period, and saved successfully.
    # Locust all and locust adults have different mask, need to redo
    if save_lo_mask:
        save_path = os.path.join('.', 'RESULTS', f'mask_locust_{locust_type}.mat')
        mdic = {"mask_locust": mask_locust}
        io.savemat(save_path, mdic)


def plot_phase_lo(mode: "mode vector", period: "this is for the title", Xlon, Ylat, nosea_indices, locust_type):
    """
    Phase plot with discrete cyclic colorbar

    [V 2022.01.19] Minor changes to the font size and title, add country borders, deactivate the gridlines
    [V 2021.07.06] Used convert_mode_vector2XY_lo() function
    [V 2021.07.01] Normalize phase to [0,1] by dividing 2pi and plus 1/2;
                   change the cyclic colorbar and the xtick accordingly
    [V 2021.06.25] Add explanations to inputs

    :param locust_type: locust type.
    :param nosea_indices: This must be nosea_indices_lo from load_data_locust
    :param Ylat: shape(220,416) the same lat in each row. This is from load_data_locust.
    :param Xlon: shape(220,416) the same lon in each column. This is from load_data_locust.
    :param mode: e.g. Phi_sub[:,0] is the first mode
    :param period: this is for the title, and also for vmin and vmax in the color map
    :return: phase figure
    """
    ## preparation for phase plot data
    mode_XYm, _ = convert_mode_vector2XY_lo(mode, nosea_indices)
    phase_XYm = np.angle(mode_XYm)  # compute the angle [radians], -pi ~ pi;
    phaseXYm_yr = phase_XYm / 2 / pi * period  # phase in [yr]
    phaseXYmn = phase_XYm / 2 / pi + 1 / 2  # phase in [0,1]

    ## preparation for color map and color bar
    # get the colors from 'jet' colormap
    cmap = plt.get_cmap('RdBu')  # 'jet'
    # define the bins and norm
    npiece = 10
    bounds = np.linspace(0, 1, npiece + 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    ## plot phase
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])

    fig.subplots_adjust(wspace=0)

    ax = plt.subplot(gs[0], projection=ccrs.PlateCarree())
    extent = [-18, 86, -4, 51]  # lon_min, lon_max, lat_min, lat_max
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.coastlines()
    # country borders
    ax.add_feature(cartopy.feature.BORDERS, alpha=1, zorder=1, edgecolor='#D0D0D0')
    top10_a3 = ['MRT', 'SAU', 'IND', 'SDN', 'PAK', 'KEN', 'YEM', 'NER', 'DZA']  # Method 2, , 'MAR'
    highlight_country(top10_a3, ax, show_hl=False, linewidth=1)

    # adjust the gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlines = False
    gl.ylines = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 10, 'color': 'gray'}
    gl.ylabel_style = {'size': 10, 'color': 'gray'}

    # title
    # ax.set_title("Phase of {}: period = {:.3f} yr".format(locust_type, period))
    fontsize = 20
    if locust_type == 'all':
        add_on = "1985-2020"
    elif locust_type == 'all_1st18y':
        add_on = "1985-2002"
    elif locust_type == 'all_2nd18y':
        add_on = "2003-2020"
    else:
        add_on = locust_type
    ax.set_title("Phase, {}".format(add_on), fontsize=fontsize)

    # use cyclic colorbar (phase colorbar)
    cs = ax.pcolor(Xlon, Ylat, phaseXYmn, shading='auto', cmap=cmap, norm=norm, transform=ccrs.PlateCarree(),
                   zorder=3)

    # set up colorbar
    cbar = fig.colorbar(cs, orientation="horizontal")
    cbar.ax.set_xlabel('phase [yr]')

    ################
    ## plot the cyclic color bar
    # ax2 = fig.add_axes([0.95, 0.1, 0.03, 0.8], projection='polar')  # [left, bottom, width, height]
    # get the colors from 'jet' colormap
    cmap2 = plt.get_cmap('RdBu', npiece)    #'jet'
    # define the bins and normalize
    bounds2 = np.linspace(0, 360, npiece + 1)
    norm2 = mpl.colors.BoundaryNorm(bounds2, cmap2.N)

    # plot
    azimuths = np.arange(0, 361, 1)
    zeniths = np.linspace(0.5, 1., 30)
    # values = np.flip(azimuths) * np.ones((30, 361))   # phase have both negative and positive values
    values = azimuths * np.ones((30, 361))  # phase only have positive values
    ax2 = plt.subplot(gs[1], projection='polar')
    ax2.pcolormesh(azimuths * np.pi / 180.0, zeniths, values, cmap=cmap2, norm=norm2, shading='auto')
    ax2.set_yticks([])
    xticks = [(i * 360 / npiece) * np.pi / 180 for i in range(npiece)]
    ax2.set_xticks(xticks)
    # negative and positive
    # xtick_label_half1 = [i * 1/npiece for i in range(int(npiece/2)+1)][::-1]
    # xtick_label_half2 = [-i * 1/npiece for i in range(1, int(npiece/2))]
    # reverse the order, cannot use list.reverse(), because a function cannot be added with a list
    # xtick_label = xtick_label_half1 + xtick_label_half2

    # positive only
    xtick_label = [i * 1 / npiece for i in range(npiece)]

    ax2.set_xticklabels(map(str, xtick_label), fontsize=15)
    ax2.set_title('Phase [yr]')
    ax2.set_ylim(0, 1)  # assure to show the blank

