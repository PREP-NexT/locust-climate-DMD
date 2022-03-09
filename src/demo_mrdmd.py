import sys
sys.path.append("./src")        # This should be the path where the codes are

from data_utils import *
from mrdmd_utils import *
# from analysis_utils import *
from plot_utils import *

# # working directory
# os.getcwd()
# os.chdir('')    #This should be the path where the data folder is

#############################  retrieve spatio-temporal patterns by mrDMD ################################
"""
This demo consists of dynamic extraction and interpretation in the overall framework, which is shown in
part b-c in supplementary figure 1. Figure 2 and 4 could be reproduced by this script.
"""

################## plot settings ##################
# plot_settings()       # change the font
save_path = "../results"
if not os.path.exists(save_path):
    os.makedirs(save_path)
###################### load data ##########################
## The data is available from "https://github.com/PREP-NexT/locust-climate-DMD" and should be saved under "./data" folder.

locust_type = 'all'     # this could be changed!

file_name = ''.join(['timesnapshots_', locust_type, 'Nosea'])
file_path = os.path.join('..', 'data', file_name)       # this is where the .mat file is
timess_lo, Ylat, Xlon, nosea_indices_lo = load_data_locust(file_path)

"""
Note: Change "locust_type" to run mrDMD for different periods, or different locust types. 
You may choose from:
['Adults', 'Hoppers', 'Swarms', 'Bands', 'all', 'all_1st18y', 'all_2nd18y']. 
(the corresponding datasets will be available soon at https://github.com/PREP-NexT/locust-climate-DMD)
The first five are for different types of locusts or all locusts in total, 
the last two are for all locusts in period 1985-2002 or 2003-2020.
"""

####################### conduct mrDMD ##############################
max_level = 5       # do not change!
max_cycle = 38      # do not change!

if (locust_type == 'all_1st18y') | (locust_type == 'all_2nd18y'):
    max_level = 4   # do not change!
    max_cycle = 22  # do not change!

"""
Note: “max_cycle” affects the minimum frequency of your slow modes.
Putting it to a larger value assure that we find the period we want.
Here, we use max_cycle = 38 to find modes near period 1 year.
“max_level” does not affect modes in level 0. You only need to use it in finding El Nino and La Nina events
(max_level=5 meaning 6 levels in total, which is what we want).
For half period data (i.e., 1985-2002 and 2003-2020), we can use a lower max_cycle and max_level.
"""

# doing mrdmd, nodes contain all nodes with modes or not at different levels and different time bins.
nodes_lo = mrdmd(timess_lo, max_levels=max_level, max_cycles=max_cycle)

# For level 0,
Phi0_lo, _ = stitch(nodes_lo, 0)    # stitch can find all the modes at a given level (here level=0) --> "Phi0_lo".
nodes0_lo = [n for n in nodes_lo if n.level == 0]   # all the nodes at level 0  --> "nodes0_lo".
sLambda0_lo = nodes0_lo[0].sLambda      # corresponding eigenvalues
freq0_lo = np.imag(np.log(sLambda0_lo)) / 2 / np.pi * 12  # corresponding frequencies, unit:[1/yr]
period0_lo = 1 / freq0_lo  # corresponding period, unit: [yr]


################## mode selection for all modes in level 0 ###################
# Goal: find the mode with highest power near period=1 year. See details in Methods.

p = 10       # p is selected by iterative method following Proctor and Eckhoff
df = mode_selection(p, sLambda0_lo, Phi0_lo, dt=1, plot=None)

"""
Note: change "plot=None" to "plot='normal'" or "plot='log'" can visualize the mode selection.
"""
################### visualize 1-year periodic mode (dynamic pattern) ####################

mode_locust, period_locust = get_1yr_mode(locust_type, Phi0_lo, period0_lo)     # retrieve 1-year periodic mode from level 0 nodes
plot_mag_lo(mode=mode_locust, Xlon=Xlon, Ylat=Ylat, nosea_indices=nosea_indices_lo,
            save_lo_mask=False, locust_type=locust_type, savepath=save_path)        # plot magnitude in map
plot_phase_lo(mode_locust, period_locust, Xlon, Ylat, nosea_indices_lo, locust_type=locust_type, savepath=save_path)     # plot phase in map


#################### retrieve locust dynamic patterns influenced by El Nino/La Nina events ######################

# get the magnitude of locust mode influenced by El Nino/La Nina events
mag_El = mag_lo_ElLa(nodes_lo, ElLa='El', plot=True, Xlon=Xlon, Ylat=Ylat, nosea_indices_lo=nosea_indices_lo, savepath=save_path)
mag_La = mag_lo_ElLa(nodes_lo, ElLa='La', plot=True, Xlon=Xlon, Ylat=Ylat, nosea_indices_lo=nosea_indices_lo, savepath=save_path)

# plot El Nino vs La Nina
mag_diff = mag_El - mag_La
plot_mag_lo(Xlon=Xlon, Ylat=Ylat, nosea_indices=nosea_indices_lo, mag=mag_diff,
                normalization=False, mag_diff=True, savepath=save_path)

print("Finished!")
