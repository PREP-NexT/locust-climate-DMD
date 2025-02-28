import numpy as np
import os
from scipy import io
import netCDF4 as nc


def load_data_locust(filepath: "The file contains time snapshot matrix (without sea) created by MATLAB"):
    timess_mat = io.loadmat(filepath)
    X = timess_mat['zz']  # var_name = 'zz' or 'zz_all'
    lat = timess_mat['lat_nosea']
    lon = timess_mat['lon_nosea']
    nosea_indices = timess_mat['nosea_indices']
    xlon = timess_mat['Xlon']
    ylat = timess_mat['Ylat']
    Xlon, Ylat = np.meshgrid(xlon, ylat)
    Ylat = np.flipud(Ylat)  # Flip array in the up/down direction
    return X, Ylat, Xlon, nosea_indices


## country mask
def load_mask_country(r=0.4, locust_type='all'):
    """

    :param r: the area ratio affected by locust is larger than r
    :param locust_type: used for load_mask_locust(locust_type)
    :return:    mask_country   sea and mag=0 is true (masked), mag>0 is false
                cowcode_XY  sea is true (masked)
                country_sel: the sorted countries where area ratio affected by locust is larger than r

    """
    ## load the country mask for the whole world map
    fn = os.path.join('.', 'DATA', 'country_mask_0.25deg_2016-12-01.nc')
    # Displays the variable names, dimension, and units
    ds = nc.Dataset(fn)
    # print(ds)
    # ds gives us information about the variables contained in the file and their dimensions.
    # for var in ds.variables.values():
    #     print(var)

    # Read data according to the variable names
    lon = ds.variables['lon'][:]
    lat = ds.variables['lat'][:]
    cowcode_XY = ds.variables['COWCODE'][:]

    # crop
    index1 = np.asarray(np.where((lat > -4) & (lat < 51))).flatten()
    index2 = np.asarray(np.where((lon > -18) & (lon < 86))).flatten()
    cowcode_XY = cowcode_XY[index1, :]
    cowcode_XY = cowcode_XY[:, index2]

    # print('cowcode_XY shape = ', cowcode_XY.shape)  # check crop
    cowcode_z = np.array(cowcode_XY[~cowcode_XY.mask])  # flattened w/o mask array

    ## get the country mask for mag>0
    mask_locust = load_mask_locust(locust_type)
    cowcode_XY_mag = np.ma.masked_where(mask_locust, cowcode_XY)
    # count the unmasked elements in masked array
    np.ma.MaskedArray.count(cowcode_XY_mag)  # 7902, different from locust_mag=7949

    cowcode_z_mag = np.array(cowcode_XY_mag[~cowcode_XY_mag.mask])  # flattened w/o mask array
    # print('cowcode_z_mag shape = ', cowcode_z_mag.shape)    # for further computation

    # plot the country mask
    # plt.figure()
    # plt.imshow(cowcode_XY)
    # plt.figure()
    # plt.imshow(cowcode_XY_mag)

    ## find important country to analyze
    (unique_org, counts_org) = np.unique(cowcode_z, return_counts=True)
    (unique, counts) = np.unique(cowcode_z_mag, return_counts=True)

    # get the ratio of locust affected area compared with the country area
    ratio = np.array([counts[i] / counts_org[np.where(unique_org == unique[i])] for i in range(len(unique))]).squeeze()
    ratio = np.sort(ratio)
    country_sel = unique[ratio > r]

    return cowcode_XY, cowcode_XY_mag, cowcode_z_mag, country_sel


###### load the country mask
def load_mask_locust(locust_type):
    # period_locust = 'inf'    # should be a one digital period from DMD.py (for locust)
    fn = os.path.join('.', 'RESULTS', f"mask_locust_{locust_type}.mat")
    mat = io.loadmat(fn)
    mask = mat['mask_locust']
    return mask


################################################
## Added:
## Load climatic data

def load_data_climatic(filename, var_name):
    """
    Load climatic data, including spi1,spi3, fldfrac, smp, tsurf, u10 and v10

    [V 2021.07.05] Add var u10 and v10.
    [V 2021.06.19] Make it applicable to all variables

    :param filename: .nc file
    :param var_name: variable name can be seen in Panoply.
    :return: X: time snapshot matrix,
             lat: the same as the data array conventions,
             lon: the same as the data array conventions.
             nosea_indices: nosea_indices for climatic variables!!!
    """
    # load the netcdf data
    fn = os.path.join('.', 'DATA', filename)
    # Displays the content of your NetCFD file (.nc)
    # With this operation you can find the variable names, dimension, and units
    ds = nc.Dataset(fn)
    # print(ds)
    # ds gives us information about the variables contained in the file and their dimensions.
    # for var in ds.variables.values():
    #     print(var, '\n')

    # Read data according to the variable names
    lon = ds.variables['lon'][:]
    lat = ds.variables['lat'][:]
    spi3 = ds.variables[var_name][:]
    # time = ds.variables['t'][:]

    # create the timesnapshot
    # print(spi3.shape)
    # print(type(spi3))

    # Remember to make time snapshot data X a tall thin dataset, with long numbers in the first dim
    # and time steps in the second dim!!!
    if (var_name == 'spi1') | (var_name == 'spi3'):
        X = np.reshape(np.array(spi3), (np.shape(spi3)[0], 220 * 416)).T
        nosea_indices = False
    if (var_name == 'vcpct') | (var_name == 'fldfrc'):
        # reshape
        X = np.reshape(spi3, (np.shape(spi3)[0], 220 * 416)).T
        # get rid of the sea and the mask (causing filled missing value will influence DMD)
        nosea_indices = ~X.mask[:, 1]
        X = X[nosea_indices, :]
        X = np.array(X)
    if (var_name == 'stl1') | (var_name == 't2m') | (var_name == 'u10') | (var_name == 'v10'):
        # reshape
        X = np.reshape(spi3, (np.shape(spi3)[0], 220 * 416)).T
        # get rid of the sea and the mask (causing filled missing value will influence DMD)
        nosea_indices = ~X.mask[:, 1]
        X = X[nosea_indices, :]
        X = np.array(X)

    return X, lat, lon, nosea_indices