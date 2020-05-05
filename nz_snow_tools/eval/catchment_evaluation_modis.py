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

import os

# os.environ['PROJ_LIB']=r'C:\miniconda\envs\nz_snow27\Library\share'

def calc_modis_catchment_metrics(catchment, years_to_take, mask_folder, modis_folder, modis_dem, output_folder):
    """
    reads in and computes statistics on MODIS data for specific catchments. Requires that mask has been created using
    nz_snow_tools/util/generate_mask.py
    saves output to a pickle file
    :param catchment,  # string identifying catchment modelled
    :param years_to_take
    :param modis_sc_threshold = 50  # value of fsca (in percent) that is counted as being snow covered
    :param mask_folder  folder with numpy masks
    :param modis_folder  folder with cloudfilled modis data
    :param modis_dem name of modis grid 'modis_nz_dem_250m'
    :param output_folder folder to put output pickle file
    """

    # set up lists
    ann_ts_av_sca_m = []
    ann_ts_av_sca_thres_m = []
    ann_dt_m = []
    ann_scd_m = []

    for year_to_take in years_to_take:

        print('loading modis data {}'.format(year_to_take))
        # load modis data for evaluation - trims to extent of catchment
        modis_fsca, modis_dt, modis_mask = load_subset_modis_annual(catchment, year_to_take, modis_folder, modis_dem, mask_folder)
        modis_sc = modis_fsca >= modis_sc_threshold

        # print('calculating basin average sca')
        num_modis_gridpoints = np.sum(modis_mask)
        ba_modis_sca = []
        ba_modis_sca_thres = []
        for i in range(modis_fsca.shape[0]):
            ba_modis_sca.append(np.nanmean(modis_fsca[i, modis_mask]) / 100.0)
            ba_modis_sca_thres.append(np.nansum(modis_sc[i, modis_mask]).astype(np.double) / num_modis_gridpoints)

        # print('adding to annual series')
        ann_ts_av_sca_m.append(np.asarray(ba_modis_sca))
        ann_ts_av_sca_thres_m.append(np.asarray(ba_modis_sca_thres))

        # print('calc snow cover duration')
        modis_scd = np.sum(modis_sc, axis=0)  # count days with snow over threshold
        modis_scd[modis_mask == 0] = -999  # set areas outside catchment to -999
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
    outfile = '/summary_MODIS_{}_{}_{}_{}_thres{}.pkl'.format(years_to_take[0], years_to_take[-1], catchment, modis_dem, modis_sc_threshold)
    pickle.dump(ann, open(output_folder + outfile, 'wb'), -1)


if __name__ == '__main__':

    catchment = 'Tar_Patea_NZTM'  # string identifying catchment modelled
    years_to_take = np.arange(2010, 2019 + 1)  # range(2016, 2016 + 1)  # [2013 + 1]  # range(2001, 2013 + 1)
    modis_sc_threshold = 50  # value of fsca (in percent) that is counted as being snow covered
    mask_folder = '/nesi/nobackup/niwa00026/Observation/Snow_RemoteSensing/catchment_masks'  # location of numpy catchment mask. must be writeable if mask_created == False
    modis_folder = '/nesi/nobackup/niwa00026/Observation/Snow_RemoteSensing/'
    modis_dem = 'modis_nz_dem_250m'
    output_folder = '/nesi/nobackup/niwa00026/Observation/Snow_RemoteSensing/catchment_output'

    calc_modis_catchment_metrics(catchment, years_to_take, mask_folder, modis_folder, modis_dem, output_folder)
