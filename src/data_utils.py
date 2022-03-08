import numpy as np
import os
from scipy import io


def load_data_locust(filename: "The file contains time snapshot matrix (without sea) created by MATLAB"):
    timess_mat = io.loadmat(os.path.join('..', 'data', filename))
    X = timess_mat['zz']  # var_name = 'zz' or 'zz_all'
    lat = timess_mat['lat_nosea']
    lon = timess_mat['lon_nosea']
    nosea_indices = timess_mat['nosea_indices']
    xlon = timess_mat['Xlon']
    ylat = timess_mat['Ylat']
    Xlon, Ylat = np.meshgrid(xlon, ylat)
    Ylat = np.flipud(Ylat)  # Flip array in the up/down direction
    return X, Ylat, Xlon, nosea_indices
