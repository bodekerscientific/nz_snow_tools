"""
code to create land mask for analysis of snow model results

"""

import numpy as np
import pickle

catchment = 'SI'
output_dem = 'nztm250m'  # identifier for output dem
years_to_take = range(2000, 2016 + 1)  # [2013 + 1]  # range(2001, 2013 + 1)
modis_sc_threshold = 50  # value of fsca (in percent) that is counted as being snow covered
modis_output_folder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/DSC Snow/MODIS'

[ann_ts_av_sca_m, ann_ts_av_sca_thres_m, ann_dt_m, ann_scd_m] = pickle.load(open(
    modis_output_folder + '/summary_MODIS_{}_{}_{}_{}_thres{}.pkl'.format(years_to_take[0], years_to_take[-1], catchment, output_dem,
                                                                          modis_sc_threshold), 'rb'))
# keep point if the SCD duration in all years is valid (inland water and ocean points are set to -1 in ann_scd_m). Points with elevation = 0 set to nan.
modis_mask = np.min(np.asarray(ann_scd_m), axis=0) >= 0
np.save(r'C:\Users\conwayjp\OneDrive - NIWA\projects\DSC Snow\MODIS\modis_mask_2000_2016.npy', modis_mask)

modis_mask = np.logical_and(np.nanmean(ann_scd_m, axis=0) < 360, modis_mask)
np.save(r'C:\Users\conwayjp\OneDrive - NIWA\projects\DSC Snow\MODIS\modis_mask_plus_permanent_snow_2010_2016.npy', modis_mask)

modis_mask = np.nanmean(ann_scd_m, axis=0) > 360
np.save(r'C:\Users\conwayjp\OneDrive - NIWA\projects\DSC Snow\MODIS\modis_permanent_snow_2010_2016.npy', modis_mask)
