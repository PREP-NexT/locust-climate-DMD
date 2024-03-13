############# Visualize the frequency time series ###################
### This code file creates the locust frequency time series in Fig. 1 in the paper.

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.transforms as transforms
from svgpathtools import svg2paths	## requires extra installation
from svgpath2mpl import parse_path	## requires extra installation
from pylab import rcParams


width = 2000
height = 1000


def plot_settings():
    font = {'family': 'Myriad Pro'}
    mpl.rc('font', **font)

    params = {'backend': 'ps',
              'axes.labelsize': 20,
              'axes.linewidth': 5,
              'grid.linewidth': 0.2,
              'font.size': 15,
              'legend.fontsize': 20,
              'legend.frameon': False,
              'xtick.labelsize': 20,
              'xtick.direction': 'out',
              'ytick.labelsize': 20,
              'ytick.direction': 'out',
              'savefig.bbox': 'tight',
              'text.usetex': False}
    rcParams.update(params)

    return

def normalize_hex(c):
    if c.startswith('#') and len(c) == 4:
        return '#{0}{0}{1}{1}{2}{2}'.format(c[1], c[2], c[3])
    return c

def get_colletion_shift_scale(dx, dy, lamda):
    """
    dx: shift PathCollection dx
    dy: shift PathCollection dy
    lamda: # scaling factor
    """
    
    trans = transforms.Affine2D().scale(lamda, lamda) + transforms.Affine2D().translate(dx, dy)

    paths = []
    facecolors = []
    #edgecolors = []
    #linewidths = []
    for elem in path_elems:
        g = elem
        p = parse_path(elem['d'])
        paths.append(trans.transform_path(p))
        facecolors.append(normalize_hex(g['fill']))
        # edgecolors.append(normalize_hex(g['stroke']))
        # linewidths.append(g['stroke-width'])
    
    collection = mpl.collections.PathCollection(
        paths, 
        #edgecolors='gray', 
        #linewidths=linewidths,
        facecolors=facecolors
    )

    return collection

def make_annual_locust_SVGs(lf, dx):
    """
    lf: locust freq 
    """
    
    nl = 500.0 # number of locust that each SVG symbol represents
    nSVG = round(lf/nl) # number of vertically stacked SVG

    locust_collections = []
    bloc = 0.972 # bottom location of SVG
    spbs = 0.012  # space between SVG
    yloc = [bloc-i*spbs for i in range(nSVG)]
    
    annual_locust_SVGs = [get_colletion_shift_scale(dx, yloc[i]*height, 0.05) for i in range(nSVG)]

    return annual_locust_SVGs
    

### Load the SVG file
#locust_path, attributes = svg2paths('locust.svg')
#attributes[8]['fill'] = 'none'
#attributes[9]['fill'] = 'none'
locust_path, attributes = svg2paths('../data/grasshopper.svg')
attributes[1]['fill'] = 'none'
attributes[2]['fill'] = 'none'
path_elems = attributes[:14]

### Load locust frequency data (annual)
nyear = 36
years = np.linspace(1985, 2020, 8).astype(int)
sX = -12.5
dX = 50
locust_data = np.loadtxt('../data/time_series_total.txt')
locust_freq_year = locust_data[:, 2].reshape(-1, 12).sum(-1)
X = [sX+dX*i for i in range(nyear)]
yticks = [965, 870, 750, 510, 33]
ylabels = [1000, 5000, 10000, 20000, 40000]
nylabels = np.shape(ylabels)[0]

locust_SVGs_all = [make_annual_locust_SVGs(locust_freq_year[iyear], X[iyear]) for iyear in range(nyear)] 
# locust_SVGs_all = [item for items in locust_SVGs_all for item in items]
# locust_legend = get_colletion_shift_scale(-50, 50, 0.15)
locust_legend = get_colletion_shift_scale(0, 50, 0.15)

### Figure
plot_settings()
fig = plt.figure(figsize=(20, 10))
ax = plt.subplot(111)
#locust.set_transform(ax.transData)
for iyear in range(nyear):
    [ax.add_artist(locust_SVGs_iyear) for locust_SVGs_iyear in locust_SVGs_all[iyear]]
ax.add_artist(locust_legend)
ax.set_xlim([-50, X[-1]+60])
ax.set_ylim([height, 0])
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_color('#4A4C4C')
ax.tick_params('x', length=10)
ax.tick_params('y', length=0)
[plt.axhline(y=yticks[i], c='#929292', ls='-', lw=1.0, zorder=-1) for i in range(nylabels)]
xticks = np.array(X[::5]) - sX
ax.set_xticks(xticks)
ax.set_xticklabels(years)
ax.set_yticks(yticks)
ax.set_yticklabels(ylabels)
ax.yaxis.tick_right()

ax.patch.set_facecolor('white')
ax.patch.set_alpha(0)
fig.patch.set_facecolor('white')
fig.patch.set_alpha(0)
plt.savefig('../results/locust.svg', facecolor=fig.get_facecolor(), edgecolor='none')
plt.show()
