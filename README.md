# locust-climate-DMD

This is the official repository for the paper "Rising risks of synchronized locust outbreaks linked to a changing climate" (not published yet). This study develops a data-driven workflow to unravel the relationship between locust and climate dynamics across multiple spatial and temporal scales. The dominant spatio-temporal dynamic patterns are extracted by multi-resolution dynamic mode decomposition (mrDMD) and the derived magnitude and phase are analysed by comparison between different countries, climatic zones, control measures, and climate variability patterns.

### Structure:
-----------
 The source codes can be found in the `src` folder. The functions on mrDMD were adapted from [Robert Taylor's blog] (http://www.pyrunner.com/weblog/2016/08/05/mrdmd-python/) and the book [Data driven science & engineering] (http://www.databookuw.com/). We provide an example `demo_mrdmd.py` to show how to retrieve spatio-temporal locust patterns and quantify the influence of El Nino/La Nina events. It will be developed into an interactive  notebook via Google Colab soon.
The data used to run the demo can be found in the `data` folder.


### Expected output:
-----------
Corresponding to the workflow as in Supplementary Figure 1, the example `demo_mrdmd.py` is divided into different sections, including data loading,  mrDMD computation, mode selection, dynamic pattern visualization, retrieving patterns associated with El Nino/La Nina events. The related variables and plots are expected.



### Requirements to run the demo:
-----------
 - You may set up a conda environment by running the following commands in the console using the `environment.yml` file provided:
 
 ``` bash
 $ conda env create -f /path/to/environment.yml
 ```
 - Or, you may install the required packages based on Python 3.x, including numpy, pandas, scipy, cartopy, matplotlib (may need to be updated).
 
