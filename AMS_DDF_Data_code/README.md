# Precip-IDF

This repository contains code and resources for analyzing precipitation data and generating Intensity-Duration-Frequency (IDF) curves. The workflow processes raw precipitation data, performs statistical analysis, and outputs IDF results.

## Execution Order

0. `ddf_environment.yml` - The conda environment used to perform this analysis.
1. `helpers/align_obs_data.py` - Align PRISM and Huidobro datasets with CMIP6 data so bias correction can be applied.
2. `bias_correct_precip_QM.py` - Uses quantile mapping found in `QMAPP_extrap.py` to bias correct the downscaled CMIP6 data.
3. `calc_AMS.py` - Calculates Annual Maximum Series (AMS) values for user specified durations.
4. `helpers/combine_AMS.py` - Combines AMS files into a single file for each time period.
5. `calc_DDF.py` - Calculates Depth-Duration-Frequency values from AMS values with many user defined parameters.
