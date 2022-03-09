# locust-climate-DMD

This is the official repository for the paper "Rising risks of synchronized locust outbreaks linked to a changing climate" (not published yet). This study develops a data-driven workflow to unravel the relationship between locust and climate dynamics across multiple spatial and temporal scales. The dominant spatio-temporal dynamic patterns are extracted by multi-resolution dynamic mode decomposition (mrDMD) and the derived magnitude and phase are analysed by comparison between different countries, climatic zones, control measures, and climate variability patterns.
This repository is currently under development.

### Structure:
-----------
 The source codes can be found in the `src` folder. The functions on mrDMD were adapted from [Robert Taylor's blog](http://www.pyrunner.com/weblog/2016/08/05/mrdmd-python/) and the book [Data driven science & engineering](http://www.databookuw.com/). We provide an example `demo_mrdmd.py` to show how to retrieve spatio-temporal locust patterns and quantify the influence of El Nino/La Nina events. It will be developed into an interactive  notebook via Google Colab soon. 
The data used to run the demo can be found in the `data` folder. The output could be found in `results` folder.


### Expected output:
-----------
Corresponding to the workflow as in Supplementary Figure 1, the example `demo_mrdmd.py` is divided into different sections, including data loading, mrDMD computation, mode selection, dynamic pattern visualization, and retrieving patterns associated with El Nino/La Nina events. The related variables and plots will be produced. The figures could be found in `results` folder.

#### Warnings:
-----------
- You may get a warning saying that dividing by zero occurred when computing the period. This is okay as some of the mode frequencies are zero, so the corresponding period will be infinity.
```
RuntimeWarning: divide by zero encountered in true_divide 
	period0_lo = 1 / freq0_lo  # corresponding period, unit: [yr]
```
- Another warning that might occur is inside the cartopy package. It could be ignored.
```
MatplotlibDeprecationWarning: 
The 'inframe' parameter of draw() was deprecated in Matplotlib 3.3 and will be removed two minor releases later. Use Axes.redraw_in_frame() instead. If any parameter follows 'inframe', they should be passed as keyword, not positionally.
```
- Downloading the required data from NaturalEarth for plotting the map may also generate warnings:
```
DownloadWarning: Downloading: http://naciscdn.org/naturalearth/110m/physical/ne_110m_coastline.zip
  warnings.warn('Downloading: {}'.format(url), DownloadWarning)
```
```
DownloadWarning: Downloading: http://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_boundary_lines_land.zip
  warnings.warn('Downloading: {}'.format(url), DownloadWarning)
```

### Requirements to run the demo:
-----------
 - You may set up a conda environment by running the following commands in the console (replace <environment_name> with the name of your virtual environment):
 
 ``` bash
 $ conda create --name <environment_name> cartopy=0.17.0 python=3.6 numpy=1.19.2 matplotlib=3.3.4 scipy=1.5.2 pandas=1.1.5
 ```
 - Or, if you decided to install the required packages (including numpy, pandas, scipy, cartopy, matplotlib) based on Python 3.x by yourself, remember to check if cartopy == 0.17.0. Using cartopy >= 0.18.0 may cause the following error: 
 ```
TypeError: 'Polygon' object is not iterable
```
 
