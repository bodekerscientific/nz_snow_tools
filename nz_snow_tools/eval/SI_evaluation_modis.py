"""

reads in and computes statistics on MODIS data for specific catchments. Requires that mask has been created using
nz_snow_tools/util/generate_mask.py

Jono Conway
"""
from __future__ import division

import numpy as np
import pickle
# from nz_snow_tools.eval.catchment_evaluation import *
from nz_snow_tools.eval.catchment_evaluation_annual import load_subset_modis_annual
from nz_snow_tools.util.utils import setup_nztm_dem

import os

# os.environ['PROJ_LIB'] = 'C:/miniconda/envs/nz_snow27/Library/share'

if __name__ == '__main__':

    origin = 'topleft'
    catchment = 'SI'  # string identifying catchment modelled
    output_dem = 'nztm250m'  # identifier for output dem
    years_to_take = range(2000, 2016 + 1)  # range(2016, 2016 + 1)  # [2013 + 1]  # range(2001, 2013 + 1)
    modis_sc_threshold = 50  # value of fsca (in percent) that is counted as being snow covered
    # mask_folder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/DSC Snow/MODIS/masks'
    # catchment_shp_folder = ''
    modis_folder = 'C:/Users/conwayjp/OneDrive - NIWA/Data/MODIS_snow/NSDI_SI_cloudfilled'
    dem_folder = 'C:/Users/conwayjp/OneDrive - NIWA/Data/GIS_DATA/Topography/DEM_NZSOS/'
    modis_dem = 'modis_si_dem_250m'
    output_folder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/DSC Snow/MODIS'

    # load si dem and trim to size. (modis has smaller extent to west (1.085e6)
    si_dem_file = dem_folder + 'si_dem_250m' + '.tif'
    nztm_dem, x_centres, y_centres, lat_array, lon_array = setup_nztm_dem(si_dem_file, extent_w=1.08e6, extent_e=1.72e6, extent_n=5.52e6, extent_s=4.82e6,
                                                                          resolution=250)

    nztm_dem = nztm_dem[:, 20:]
    x_centres = x_centres[20:]
    lat_array = lat_array[:, 20:]
    lon_array = lon_array[:, 20:]

    # set up lists
    ann_ts_av_sca_m = []
    ann_ts_av_sca_thres_m = []
    ann_dt_m = []
    ann_scd_m = []

    for year_to_take in years_to_take:
        print('loading modis data {}'.format(year_to_take))
        # load modis data for evaluation - trims to extent of catchment
        modis_fsca, modis_dt, modis_mask = load_subset_modis_annual(catchment, None, year_to_take, modis_folder, None, modis_dem, None,
                                                                    None)
        modis_sc = modis_fsca >= modis_sc_threshold
        # modis_fsca = None # get rid of fsca out of memory
        modis_mask = nztm_dem > 0  # set mask to land area only

        # print('calculating basin average sca')
        num_modis_gridpoints = np.sum(modis_mask)
        # ba_modis_sca = []
        # ba_modis_sca_thres = []
        # for i in np.arange(modis_fsca.shape[0], dtype=np.int):
        #     # ba = np.nanmean(modis_fsca[i, modis_mask]) / 100.0
        #     ba_thres = np.nansum(modis_sc[i, modis_mask]).astype(np.double) / num_modis_gridpoints
        #     # ba_modis_sca.append(ba)
        #     ba_modis_sca_thres.append(ba_thres)

        # create timeseries of average snow covered area
        ba_modis_sca = np.nanmean(modis_fsca[:, modis_mask], axis=(1)) / 100.0
        ba_modis_sca_thres = np.nansum(modis_sc[:, modis_mask], axis=(1)).astype(np.double) / num_modis_gridpoints

        # print('adding to annual series')
        ann_ts_av_sca_m.append(np.asarray(ba_modis_sca))
        ann_ts_av_sca_thres_m.append(np.asarray(ba_modis_sca_thres))

        # print('calc snow cover duration')
        modis_scd = np.sum(modis_sc, axis=0,dtype=np.float)  # count days with snow over threshold
        modis_scd[modis_mask == 0] = np.nan  # set areas outside catchment to -999
        modis_scd[np.logical_and(np.isnan(modis_fsca[0]), modis_mask == 1)] = -1  # set areas of water to -1

        # add to annual series
        ann_scd_m.append(modis_scd)
        ann_dt_m.append(modis_dt)

        # empty arrays so we don't run out of memory (probably not needed)
        modis_fsca = None
        modis_dt = None
        modis_mask = None
        modis_sc = None
        modis_scd = None

    ann = [ann_ts_av_sca_m, ann_ts_av_sca_thres_m, ann_dt_m, ann_scd_m]
    pickle.dump(ann, open(
        output_folder + '/summary_MODIS_{}_{}_{}_{}_thres{}.pkl'.format(years_to_take[0], years_to_take[-1], catchment, output_dem,
                                                                        modis_sc_threshold), 'wb'), -1)
