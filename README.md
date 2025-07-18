# IDF Curve Projections for the State of Minnesota Using Dynamically Downscaled CMIP6 Model Projections

This repository contains the necessary code for producing Intensity-Duration-Frequency (IDF) or Depth-Duration-Frequency (DDF) curves across Minnesota. The process leverages dynamically downscaled climate model projections from CMIP6 via WRF. 

The project is divided into two sections:  `AMS_DDF_Data_code` and  `DDF_analysis_figures_code`. 

---


## AMS_DDF_Data_code

This repository contains code and resources for analyzing precipitation data and generating Intensity-Duration-Frequency (IDF) curves. The workflow processes raw precipitation data, performs statistical analysis, and outputs DDF results. This process directly calls on downscaled climate simulations (via WRF) and outputs AMS and DDF rasters. 

### Execution Order

0. `ddf_environment.yml` - The conda environment used to perform this analysis.
1. `helpers/align_obs_data.py` - Align PRISM and Huidobro datasets with CMIP6 data so bias correction can be applied.
2. `bias_correct_precip_QM.py` - Uses quantile mapping found in `QMAPP_extrap.py` to bias correct the downscaled CMIP6 data.
3. `calc_AMS.py` - Calculates Annual Maximum Series (AMS) values for user specified durations.
4. `helpers/combine_AMS.py` - Combines AMS files into a single file for each time period.
5. `calc_DDF.py` - Calculates Depth-Duration-Frequency values from AMS values with many user defined parameters.

## DDF_analysis_figures_code

This repository contains code for analyzing the output from the AMS_DDF_Data_code. It calculates the analyses shown in Fernandez et al., 2025 and outputs figures, tables or console-printed results. 

### Contents

0. `environment.yml` - The conda environment used to perform this analysis.
1. `funcs.py` - Script of python functions necessary to run all other scripts in the repository.
2. `MK_test.py` - Performs Mann-Kendall trend tests on AMS series from NOAA datasets in order to assess statistically significant incresing or decreasing trends in the data (Figure 2 in article)
3. `DDF_change_distrib.py` - Produces box and whisker plots of the %-change in DDF values (Figure 3 in article)
4. `DDF_map_RC_bc.py` - Produces maps of the state of Minnesota which show projected %-change in IDF values for different return intervals and emissions scenarios (Figure 4 in article) (Sample figure below)
5. `DDF_bias_distribs_RI_mod.py` - Box-whisker plots of bias for each model, compared to NA14 data (Figure 5 in article)
6. `DDF_map_abs_proj.py` - Maps as in Figure 4, but for absolute IDF values projections (Figure 6 in article). Also, side-by-side comparisons of historical and future projection IDF values and their respective empirical distributions (Figure 7 in article). (Sample figure below)
7. `DDF_curve_proj_bc*.py`- IDF curves plotted either by model, period or SSP (Figure 8 in article).
8. `DDF_abs_proj_vals.py` - Outputs Multiplicative Median Increase (%) and Median Absolute Change (MAC) to console based on period, SSP, RI, duration and bias correction scheme.


### Other Files
* **`LICENSE`**

---

## Sample Figure Output
Sample of relative IDF change in projections: 
<img width="2566" height="2621" alt="map_RC_bc_unadjustedLiessPrecip_ssp370_2080-2099_2yr_1da" src="https://github.com/user-attachments/assets/eaf90f70-5768-4f65-8f38-ac9e47b6ccf5" />

Sample of historical and future projection side-by-side comparison: 
<img width="3986" height="1633" alt="map_RC_bc_qmappPrismPrecip_ssp370_2080-2099_2yr_1da" src="https://github.com/user-attachments/assets/1fee76e2-ce63-404d-b38c-9afa807ade1b" />
<img width="2969" height="1764" alt="distribs_bc_qmappPrismPrecip_ssp370_2080-2099_2yr_1da" src="https://github.com/user-attachments/assets/99c48056-548d-48db-b143-ef9843431359" />

