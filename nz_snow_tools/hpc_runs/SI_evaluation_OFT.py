"""

reads in and computes statistics on dsc_snow model data for SI domain

Jono Conway
"""
from __future__ import division

import numpy as np
import netCDF4 as nc
import pickle


def load_oft_output_annual_without_mask(met_inp, which_model, catchment, output_dem, run_id, hydro_year_to_take, dsc_snow_output_folder, origin='bottomleft'):
    """
    load output from python otf model
    :param catchment: string giving catchment area to run model on
    :param output_dem: string identifying the grid to run model on
    :param hydro_year_to_take: integer specifying the hydrological year to run model over. 2001 = 1/4/2000 to 31/3/2001
    :return: st_swe, st_melt, st_acc, out_dt. daily grids of SWE at day's end, total melt and accumulation over the previous day, and datetimes of ouput
    """

    dsc_snow_output = nc.Dataset(
        dsc_snow_output_folder + '/snow_out_{}_{}_{}_{}_{}_{}.nc'.format(met_inp, which_model, catchment, output_dem, run_id, hydro_year_to_take), 'r')

    out_dt = nc.num2date(dsc_snow_output.variables['time'][:], dsc_snow_output.variables['time'].units, only_use_cftime_datetimes=False,
                         only_use_python_datetimes=True)

    st_swe = dsc_snow_output.variables['swe'][:]  # swe in mm w.e.
    # st_melt_total = dsc_snow_output.variables['melt'][:]
    # st_acc_total = dsc_snow_output.variables['acc'][:]
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

    return st_swe, out_dt  # convert to mm w.e.


def SI_evaluation(run_id, met_inp, which_model, hydro_years_to_take, catchment, output_dem, origin, model_swe_sc_threshold, mask_file, dsc_snow_output_folder,
                  output_folder):
    modis_mask = np.load(mask_file)

    # set up lists
    ann_ts_av_swe_m = []
    ann_ts_av_sca_thres_m = []
    ann_dt_m = []
    ann_scd_m = []
    ann_av_swe_m = []
    ann_max_swe_m = []
    ann_metadata = {}
    ann_metadata['modis_mask'] = mask_file
    num_model_gridpoints = np.sum(modis_mask)
    print('{} grid land points in domain'.format(num_model_gridpoints))
    area_domain = num_model_gridpoints * 0.250 * 0.250  # domain area in km^2
    print('domain size {} km**2 (excluding inland water/ocean)'.format(area_domain))
    ann_metadata['area_domain'] = area_domain
    ann_metadata['num_model_gridpoints'] = num_model_gridpoints
    ann_metadata['num_nan_model_gridpoints'] = {}

    for hydro_year_to_take in hydro_years_to_take:
        print('loading otf model data {}'.format(hydro_year_to_take))
        # load previously run simulations from netCDF
        st_swe, out_dt = load_oft_output_annual_without_mask(met_inp, which_model, catchment, output_dem, run_id, hydro_year_to_take, dsc_snow_output_folder,
                                                             origin=origin)

        # cut to same shape as MODIS data(2800L, 2540L), compared to SI dem of (2800L, 2560L), the W edge is further east in modis, so remove first 20 columns
        st_swe = st_swe[:, :, 20:]
        st_swe[:, modis_mask == False] = np.nan

        # create binary snow cover
        st_sc = np.zeros(st_swe.shape, dtype=np.float32)  # needs to be float to handle nan's. Start with 0's i.e. no snow.
        st_sc[st_swe > model_swe_sc_threshold] = 1 # set to snow cover when over threshold
        st_sc[:, modis_mask == False] = np.nan  # set non-land points to np.nan. Includes inland water/ocean points from MODIS 2000-2016 and dem ==0

        num_nan_model_gridpoints = np.sum(np.any(np.isnan(st_swe[:, modis_mask]), axis=0))
        print('{} land grid points with nan hydro year {}'.format(num_nan_model_gridpoints,hydro_year_to_take))
        ann_metadata['num_nan_model_gridpoints'][hydro_year_to_take] = num_nan_model_gridpoints

        # create timeseries of average swe (mm w.e.) and snow covered area (0-1)
        ba_model_swe = np.nanmean(st_swe[:, modis_mask], axis=1)
        ba_model_sca_thres = np.nanmean(st_sc[:, modis_mask], axis=1).astype(np.double)  # use nan mean * area to get around nan's on land points

        # print('adding to annual series')
        ann_ts_av_swe_m.append(np.asarray(ba_model_swe))
        ann_ts_av_sca_thres_m.append(np.asarray(ba_model_sca_thres))

        # print('calc snow cover duration and average and max swe')
        model_scd = np.sum(st_sc, axis=0)  # count days with snow over threshold. mask already applied to snow cover data
        model_av_swe = np.mean(st_swe, axis=0)  # average swe. mask already applied to snow cover data
        model_max_swe = np.max(st_swe, axis=0)

        # add to annual series
        ann_av_swe_m.append(model_av_swe)
        ann_max_swe_m.append(model_max_swe)
        ann_scd_m.append(model_scd)
        ann_dt_m.append(out_dt)

        # empty arrays so we don't run out of memory (probably not needed)
        st_swe = None
        st_sc = None

    ann = [ann_ts_av_swe_m, ann_ts_av_sca_thres_m, ann_dt_m, ann_scd_m, ann_av_swe_m, ann_max_swe_m, ann_metadata]
    pickle.dump(ann, open(
        output_folder + '/summary_MODEL_{}_{}_{}_{}_{}_{}_{}_thres{}.pkl'.format(hydro_years_to_take[0], hydro_years_to_take[-1], met_inp, which_model,
                                                                                 catchment, output_dem, run_id,
                                                                                 model_swe_sc_threshold), 'wb'), protocol=3)


if __name__ == '__main__':
    run_id = 'dsc_default'
    met_inp = 'vcsn_norton'  # identifier for input meteorology
    which_model = 'dsc_snow'  # 'clark2009'  # 'dsc_snow'  # string identifying the model to be run. options include 'clark2009', 'dsc_snow' # future will include 'fsm'
    # time and grid extent options
    hydro_years_to_take = np.arange(2001, 2020 + 1)
    catchment = 'SI'  # string identifying the catchment to run. must match the naming of the catchment mask file
    output_dem = 'si_dem_250m'  # string identifying output DEM

    origin = 'bottomleft' # origin of model domain. 'bottomleft' for python otf
    model_swe_sc_threshold = 20  # threshold for treating a grid cell as snow covered (mm w.e)
    mask_file = '/nesi/nobackup/niwa00004/jonoconway/snow_sims_nz/modis_mask_2000_2016.npy'  # masked for any ocean/inland water cells in modis record + elevation >0
    dsc_snow_output_folder = '/nesi/nobackup/niwa00004/jonoconway/snow_sims_nz/vcsn'  # path to snow model output
    output_folder = '/nesi/nobackup/niwa00004/jonoconway/snow_sims_nz/vcsn'

    SI_evaluation(run_id, met_inp, which_model, hydro_years_to_take, catchment, output_dem, origin, model_swe_sc_threshold, mask_file, dsc_snow_output_folder,
                  output_folder)
