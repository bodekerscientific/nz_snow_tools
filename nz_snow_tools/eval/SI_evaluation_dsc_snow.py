"""

reads in and computes statistics on dsc_snow model data for SI domain

Jono Conway
"""
from __future__ import division

import numpy as np
import netCDF4 as nc
import pickle

def load_dsc_snow_output_annual_without_mask(catchment, output_dem, hydro_year_to_take, dsc_snow_output_folder, run_opt, origin='bottomleft'):
    """
    load output from dsc_snow model previously run from linux VM
    :param catchment: string giving catchment area to run model on
    :param output_dem: string identifying the grid to run model on
    :param hydro_year_to_take: integer specifying the hydrological year to run model over. 2001 = 1/4/2000 to 31/3/2001
    :return: st_swe, st_melt, st_acc, out_dt. daily grids of SWE at day's end, total melt and accumulation over the previous day, and datetimes of ouput
    """
    data_id = '{}_{}'.format(catchment, output_dem)

    dsc_snow_output = nc.Dataset(dsc_snow_output_folder + '/{}_{}_{}.nc'.format(data_id, hydro_year_to_take, run_opt), 'r')

    out_dt = nc.num2date(dsc_snow_output.variables['time'][:], dsc_snow_output.variables['time'].units)

    st_swe = dsc_snow_output.variables['snow_water_equivalent'][:]
    # st_melt_total = dsc_snow_output.variables['ablation_total'][:]
    # st_acc_total = dsc_snow_output.variables['accumulation_total'][:]
    # st_water_output_total = dsc_snow_output.variables['water_output_total'][:]

    if origin == 'topleft':
        st_swe = np.flip(st_swe, axis=1)
    #     st_melt_total = np.flip(st_melt_total, axis=1)
    #     st_acc_total = np.flip(st_acc_total, axis=1)
    #
    # # convert to daily sums
    # st_melt = np.concatenate((st_melt_total[:1, :], np.diff(st_melt_total, axis=0)))
    # st_acc = np.concatenate((st_melt_total[:1, :], np.diff(st_acc_total, axis=0)))
    # st_water_output = np.concatenate((st_water_output_total[:1, :], np.diff(st_water_output_total, axis=0)))

    return st_swe * 1e3, out_dt  # convert to mm w.e.


if __name__ == '__main__':

    run_id = 'norton_5_topleft'
    origin = 'topleft'
    catchment = 'SouthIsland'  # string identifying catchment modelled
    output_dem = 'nztm250m'  # identifier for output dem
    years_to_take = range(2000, 2016 + 1)  # range(2016, 2016 + 1)  # [2013 + 1]  # range(2001, 2013 + 1)
    model_swe_sc_threshold = 20  # threshold for treating a grid cell as snow covered (mm w.e)
    mask_file = '/home/jonoconway/work/DSC_snow/modis_mask_2000_2016.npy' # masked for any ocean/inland water cells in modis record + elevation >0
    dsc_snow_output_folder = '/home/jonoconway/work/DSC_snow/NoCorrection' # path to snow model output
    output_folder = '/home/jonoconway/work/DSC_snow/'

    modis_mask = np.load(mask_file)

    # set up lists
    ann_ts_av_swe_m = []
    ann_ts_av_sca_thres_m = []
    ann_dt_m = []
    ann_scd_m = []
    ann_swe_m = []

    for year_to_take in years_to_take:

        print('loading dsc_snow model data {}'.format(year_to_take))
        # load previously run simulations from netCDF
        st_swe, out_dt = load_dsc_snow_output_annual_without_mask(catchment, output_dem, year_to_take, dsc_snow_output_folder,
                                                                 run_id, origin=origin)

        # cut to same shape as MODIS data(2800L, 2540L), compared to SI dem of (2800L, 2560L), the W edge is further east in modis, so remove first 20 columns
        st_swe = st_swe[:, :, 20:]

        # create binary snow cover
        st_sc = np.zeros(st_swe.shape, dtype=np.float32)
        st_sc[st_swe > model_swe_sc_threshold] = 1
        st_sc[:, modis_mask == False] = np.nan

        num_model_gridpoints = np.sum(modis_mask)

        # create timeseries of average swe and snow covered area
        ba_model_swe = np.nanmean(st_swe[:, modis_mask], axis=1)
        ba_model_sca_thres = np.nansum(st_sc[:, modis_mask], axis=1).astype(np.double) / num_model_gridpoints

        # print('adding to annual series')
        ann_ts_av_swe_m.append(np.asarray(ba_model_swe))
        ann_ts_av_sca_thres_m.append(np.asarray(ba_model_sca_thres))

        # print('calc snow cover duration')
        model_scd = np.sum(st_sc, axis=0, dtype=np.float)  # count days with snow over threshold. mask already applied to snow cover data
        model_swe = np.sum(st_swe, axis=0, dtype=np.float)  # count days with snow over threshold. mask already applied to snow cover data

        # add to annual series
        ann_swe_m.append(model_swe)
        ann_scd_m.append(model_scd)
        ann_dt_m.append(out_dt)

        # empty arrays so we don't run out of memory (probably not needed)
        st_swe = None
        st_sc = None


    ann = [ann_ts_av_swe_m, ann_ts_av_sca_thres_m, ann_dt_m, ann_scd_m, ann_swe_m]
    pickle.dump(ann, open(
        output_folder + '/summary_MODEL_{}_{}_{}_{}_thres{}.pkl'.format(years_to_take[0], years_to_take[-1], catchment, run_id,
                                                                        model_swe_sc_threshold), 'wb'), protocol=3)
