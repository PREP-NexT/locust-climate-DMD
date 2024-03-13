############# Visualize the frequency map ###################
### This code file creates the locust frequency map in Fig. 1 in the paper.

import cartopy.crs as ccrs
import cartopy
import cartopy.io.shapereader as shpreader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patheffects as PathEffects
from pylab import rcParams

import netCDF4 as nc

import os
import pandas as pd
import numpy as np
from scipy import io, stats
from scipy.stats import zscore
from statsmodels.distributions.empirical_distribution import ECDF
import seaborn as sns


###################### Modify the plot ########################
def plot_settings():
    font = {'family': 'Myriad Pro'}
    mpl.rc('font', **font)

    params = {'backend': 'ps',
              'axes.linewidth': 1,
              'text.usetex': False}
    rcParams.update(params)
    return


############### Put ai as background #########################
# prepare ai as background
def load_ai():
    # load the netcdf data
    fn = os.path.join('..', 'data', 'ai_et0_remap.nc')
    ds = nc.Dataset(fn)
    # Read data according to the variable names
    lon = ds.variables['lon'][:]
    lat = ds.variables['lat'][:]
    ai = ds.variables['ai'][:]  # ai is an masked array
    ai = ai / 10000  # ai data in the file was multiplied by 10, 000
    return ai, lat, lon


ai_XY, lat, lon = load_ai()


############### Prepare control data as filling patterns #####################
def load_control_acm():
    mat = io.loadmat(os.path.join('..', 'data', 'acm_ControlOperations.mat'))
    ctrl_XY = mat['XY_acm']  # nan = sea
    ctrl_z = mat['z_acm_Nosea'].squeeze()  # (59173,)
    nosea_indices = mat['nosea_indices']
    # z-score normalization
    # after normalization, it is easier to set up the classification bounds
    ctrl_XY_n = np.array(ctrl_XY)
    ctrl_XY_n[~np.isnan(ctrl_XY)] = zscore(ctrl_XY[~np.isnan(ctrl_XY)])  # only apply zscore to where is not nan
    ctrl_z_n = zscore(ctrl_z)
    return ctrl_XY_n, ctrl_z_n, nosea_indices


def categorize_ctrl_acm(ctrl_z, bound1, bound2):
    """
    Estimate the bounds for categorization

    :param ctrl_z: returned from load_control_acm() function
    :param bound1: estimated x1 value in ecdf P(x<x1) for categorization bounds
    :param bound2: estimated x2 value in ecdf P(x<x2) for categorization bounds
    :return: no return, will print and show figures
    """
    ctrl_z_wo0 = ctrl_z[ctrl_z != 0]
    # categorize as high control, low control and medium control
    ecdf = ECDF(ctrl_z_wo0)  # need to squeeze to shape (x,)
    # get cumulative probability for values
    print('P(x<{:}): {:.3f}'.format(bound1, ecdf(bound1)))
    print('P(x<{:}): {:.3f}'.format(bound2, ecdf(bound2)))
    # plt.figure()
    # plt.plot(ecdf.x, ecdf.y)
    # plt.xlabel('control')
    # plt.ylabel('proportion')
    # plt.title('ecdf(control)')
    return bound1, bound2


ctrl_XY, ctrl_z, nosea_indices = load_control_acm()
bound1, bound2 = categorize_ctrl_acm(ctrl_z[ctrl_z>0], bound1=0.276887, bound2=1.00679)
lon_2D, lat_2D = np.meshgrid(lon, np.flipud(lat))


################### Highlight specific countries ##################
def highlight_country(country_name: "a list", show_hl=False, ax=plt.axes(), linewidth=1, zorder=3):
    shpfilename = shpreader.natural_earth(resolution='110m',
                                          category='cultural',
                                          name='admin_0_countries')
    reader = shpreader.Reader(shpfilename)
    countries = reader.records()

    for country in countries:
        for name in country_name:
            if country.attributes['ADM0_A3'] == name:
                ax.add_geometries(country.geometry, ccrs.PlateCarree(), zorder=zorder,
                                  facecolor='none', edgecolor='#484719',
                                  label=country.attributes['ADM0_A3'], linewidth=linewidth)
                x = country.geometry.centroid.x
                y = country.geometry.centroid.y
                if show_hl:
                    ax.text(x, y, country.attributes['NAME'].upper(), color='w', size=10, ha='center', va='center', transform=ccrs.PlateCarree(),
                            path_effects=[PathEffects.withStroke(linewidth=2, foreground="k", alpha=.8)], zorder=5)


top10_a3 = ['MRT', 'SAU', 'IND', 'SDN', 'PAK', 'KEN', 'YEM', 'NER', 'DZA', 'MAR']     


############### Read the accumulated frequency ########################

## Get the lon, lat, and frequency data
## use time snapshot data instead
def load_data_locust(filename: "The file contains time snapshot matrix (without sea) created by MATLAB",
                     var_name: "'zz' or 'zz_all'"):
    timess_mat = io.loadmat(os.path.join('..', 'data', filename))
    X = timess_mat[var_name]  # var_name = 'zz' or 'zz_all'
    lat = timess_mat['lat_nosea']
    lon = timess_mat['lon_nosea']
    nosea_indices = timess_mat['nosea_indices']
    xlon = timess_mat['Xlon']
    ylat = timess_mat['Ylat']
    Xlon, Ylat = np.meshgrid(xlon, ylat)
    Ylat = np.flipud(Ylat)  # Flip array in the up/down direction
    return X, Ylat, Xlon, nosea_indices


def convert_mode_vector2XY_lo(mode_locust, nosea_indices_lo):
    """
    Convert mode vector to mode_XY, where sea pixels and locust_mag=0 is masked

    :param mode_locust: mode vector
    :param nosea_indices_lo: from load_data_locust
    :return: mode_XYm_wo0_lo: masked, XY scale
             mask_locust
    """
    mode_wsea = np.empty(len(nosea_indices_lo), dtype=type(mode_locust[0]))
    mode_wsea[:] = np.nan

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


timess_lo, Ylat, Xlon, nosea_indices_lo = load_data_locust('timesnapshots_allNosea.mat', 'zz_all')  ## 434
acm_freq_z = timess_lo.sum(axis=1)  # (59173,)
## convert to XY
acm_freq_XYm, _ = convert_mode_vector2XY_lo(np.array(acm_freq_z, dtype=float), nosea_indices_lo)


################## Plot the frequency map #########################
plot_settings()
save = True


## preparation for color map and color bar (aridity index)
bounds = [0, 0.03, 0.2, 0.5, 0.65, 5]
cmap = ListedColormap(['#ECF2E6', '#D6E1C9', '#B2C799', '#89A963', '#6C874B'], 'white2green')
cmap_none = ListedColormap(['none', 'none', 'none', 'none', 'none'], name='transparent')
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

## plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
extent = [-18, 86, -4, 51]  # lon_min, lon_max, lat_min, lat_max
ax.set_extent(extent, crs=ccrs.PlateCarree())

# adjust the gridlines
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='none', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = False
gl.xlines = False
# gl.xlocator = mticker.FixedLocator([-180, -45, 0, 45, 180])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 10, 'color': 'black'}
gl.ylabel_style = {'size': 10, 'color': 'black'}
# gl.xlabel_style = {'color': 'red', 'weight': 'bold'}

# add coastlines
# ax.coastlines()
ax.add_feature(cartopy.feature.COASTLINE, edgecolor='#D0D0D0')

# add the colorful background
ax.stock_img()

# add aridity index as hatch
hatches_list = ['/', '\\', '-', '+', 'x']  ## 'X', '\\\\'
cs = ax.contourf(lon, lat, ai_XY, levels=bounds, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), zorder=1)

## hatch colors
hatch_colors = [i for i in np.repeat("gray", len(bounds) - 1)]
# For each level, we set the color of its hatch
for i, collection in enumerate(cs.collections):
    collection.set_edgecolor(hatch_colors[i % len(hatch_colors)])
# Doing this also colors in the box around each level
# We can remove the colored line around the levels by setting the linewidth to 0
for collection in cs.collections:
    collection.set_linewidth(0.)

cbar = fig.colorbar(cs, orientation="horizontal", ticks=bounds)
cbar.ax.set_xlabel("Aridity Index")  ## ['hyper-arid', 'arid', 'semi-arid', 'dry sub-humid', 'humid']

# border
ax.add_feature(cartopy.feature.BORDERS, alpha=1, edgecolor='#D0D0D0')
highlight_country(top10_a3, ax=ax, show_hl=True, zorder=3)

# locust frequency
## preparation for color map and color bar (locust frequency)
bounds2 = [0, 10, 20, 50, 100, 200, 500]
cmap2 = ListedColormap(['#FCDE9C', '#FAA476', '#F0746E', '#E34F6F', '#DC3977', '#B9257A', '#7C1D6F'], 'SunsetDark')
norm2 = mpl.colors.BoundaryNorm(bounds2, cmap2.N, extend='max')

s1 = ax.pcolormesh(Xlon, Ylat, acm_freq_XYm, shading='auto', zorder=2, cmap=cmap2, norm=norm2, transform=ccrs.PlateCarree())
cbar2 = fig.colorbar(s1, orientation="horizontal", ticks=bounds2)
cbar2.ax.set_xlabel("Locust frequency")

# add good control areas
ctrl_XY[ctrl_XY < bound2] = np.nan
ctrl_XY = np.ma.masked_invalid(ctrl_XY)
s2 = ax.pcolor(Xlon, Ylat, ctrl_XY, cmap=cmap_none, edgecolor='gray', lw=0.25, transform=ccrs.PlateCarree(), zorder=3.1)  ## hatch='..',

ax.set_title('Map of accumulated Locust frequency', fontsize=15)

# save
# plt.savefig('../results/locust_frequency_map.pdf', bbox_inches='tight', format='pdf')
if save:
    # plt.savefig(f'../results/locust_frequency_map.png', bbox_inches='tight', format='png', dpi=1000)
    plt.savefig(f'../results/locust_frequency_map.svg', bbox_inches='tight', format='svg')

# show
# plt.show()
