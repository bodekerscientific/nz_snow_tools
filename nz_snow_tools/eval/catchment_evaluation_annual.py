"""
code to evaluate snow models at catchment scale (i.e. Nevis or Clutha river)
options to call a series of models then compute summary statistics
reads in a computes statistics on MODIS data to evaluate against

requires that dsc_snow model has been pre run either using Fortran version or using run_snow_model
the Clark2009 model can be run on-the-fly or prerun

Jono Conway
"""
from __future__ import division

import netCDF4 as nc
import datetime as dt
import numpy as np
import pickle
from nz_snow_tools.eval.catchment_evaluation import *
from nz_snow_tools.util.utils import convert_date_hydro_DOY, trim_lat_lon_bounds, setup_nztm_dem


def load_dsc_snow_output_annual(catchment, output_dem, hydro_year_to_take, dsc_snow_output_folder, dsc_snow_dem_folder, run_opt):
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
    st_melt_total = dsc_snow_output.variables['ablation_total'][:]
    st_acc_total = dsc_snow_output.variables['accumulation_total'][:]
    # convert to daily sums
    st_melt = np.concatenate((st_melt_total[:1, :], np.diff(st_melt_total, axis=0)))
    st_acc = np.concatenate((st_melt_total[:1, :], np.diff(st_acc_total, axis=0)))

    topo_file = nc.Dataset(dsc_snow_dem_folder + '/{}_topo_no_ice.nc'.format(data_id), 'r')
    mask = topo_file.variables['catchment'][:].astype('int')
    mask = mask != 0  # convert to boolean

    # mask out values outside of catchment
    st_swe[:, mask == False] = np.nan
    st_melt[:, mask == False] = np.nan
    st_acc[:, mask == False] = np.nan

    return st_swe * 1e3, st_melt * 1e3, st_acc * 1e3, out_dt, mask  # convert to mm w.e.


def load_subset_modis_annual(catchment, output_dem, year_to_take, modis_folder, dem_folder, modis_dem, mask_folder, catchment_shp_folder):
    """
    load modis data from file and cut to catchment of interest
    :param catchment: string giving catchment area to run model on
    :param output_dem: string identifying the grid to run model on
    :param year_to_take: integer specifying the hydrological year to run model over. 2001 = 1/4/2000 to 31/3/2001
    :return: trimmed_fsca, modis_dt, trimmed_mask. The data, datetimes and catchment mask
    """
    # load a file
    nc_file = nc.Dataset(modis_folder + '/DSC_MOD10A1_{}_v0_nosparse_interp.nc'.format(year_to_take))
    ndsi = nc_file.variables['NDSI_Snow_Cover_Cloudfree'][:]  # .astype('float32')  # nsdi in %

    # trim to only the catchment desired
    mask, trimmed_mask = load_mask_modis(catchment, output_dem, mask_folder, dem_folder, modis_dem, mask_created=True,
                                         shapefile_folder=catchment_shp_folder)

    # trimmed_fsca = trim_data_bounds(mask, lat_array, lon_array, fsca[183].copy(), y_centres, x_centres)
    trimmed_ndsi = trim_data_to_mask(ndsi, mask)
    trimmed_ndsi = trimmed_ndsi.astype(np.float32, copy=False)
    trimmed_fsca = -1 + 1.45 * trimmed_ndsi  # convert to snow cover fraction in % (as per Modis collection 5)
    trimmed_fsca[trimmed_ndsi > 100] = np.nan  # set all points with inland water or ocean(237 or 239) to -999, then convert to nan once trimmed
    trimmed_fsca[trimmed_fsca > 100] = 100  # limit fsca to 100%
    trimmed_fsca[trimmed_fsca < 0] = 0  # limit fsca to 0

    # read date and convert into hydrological year
    modis_dt = nc.num2date(nc_file.variables['time'][:], nc_file.variables['time'].units)
    # mask out values outside of catchment and reset water values to np.nan
    trimmed_fsca[:, trimmed_mask == 0] = np.nan

    return trimmed_fsca, modis_dt, trimmed_mask


if __name__ == '__main__':

    which_model = 'dsc_snow'  # string identifying the model to be run. options include 'clark2009', 'dsc_snow', or 'all' # future will include 'fsm'
    clark2009run = True  # boolean specifying if the run already exists
    dsc_snow_opt = 'fortran'  # string identifying which version of the dsc snow model to use output from 'python' or 'fortran'
    dsc_snow_opt2 = 'netCDF'  # string identifying which version of output from python model 'netCDF' of 'pickle'
    catchment = 'Clutha'
    output_dem = 'nztm250m'  # identifier for output dem
    run_id = 'norton_4'  # string identifying fortran dsc_snow run. everything after the year
    years_to_take = range(2000, 2016 + 1)  # range(2016, 2016 + 1)  # [2013 + 1]  # range(2001, 2013 + 1)
    modis_sc_threshold = 50  # value of fsca (in percent) that is counted as being snow covered
    model_swe_sc_threshold = 5  # threshold for treating a grid cell as snow covered (mm w.e)
    dsc_snow_output_folder = 'T:/DSC-Snow/runs/output/clutha_nztm250m_erebus'
    clark2009_output_folder = 'T:/DSC-Snow/nz_snow_runs/baseline_clutha1'
    mask_folder = 'T:/DSC-Snow/Masks'
    catchment_shp_folder = 'Z:/GIS_DATA/Hydrology/Catchments'
    modis_folder = 'T:/sync_to_data/MODIS_snow/NSDI_SI_cloudfilled'  # ''Y:/sync_to_data/MODIS_snow/MODIS_NetCDF'
    # dem_file = 'Z:/GIS_DATA/Topography/DEM_NZSOS/clutha_dem_250m.tif'
    dem_folder = 'Z:/GIS_DATA/Topography/DEM_NZSOS/'
    modis_dem = 'modis_si_dem_250m'
    met_inp_folder = 'T:/DSC-Snow/input_data_hourly'
    dsc_snow_dem_folder = 'P:/Projects/DSC-Snow/runs/input_DEM'
    output_folder = 'P:/Projects/DSC-Snow/runs/output/clutha_nztm250m_erebus'

    # set up lists
    ann_ts_av_sca_m = []
    ann_ts_av_sca_thres_m = []
    ann_hydro_days_m = []
    ann_dt_m = []
    ann_scd_m = []

    ann_ts_av_sca = []
    ann_ts_av_swe = []
    # ann_ts_av_melt = []
    # ann_ts_av_acc = []
    ann_hydro_days = []
    ann_dt = []
    ann_scd = []

    ann_ts_av_sca2 = []
    ann_ts_av_swe2 = []
    # ann_ts_av_melt2 = []
    # ann_ts_av_acc2 = []
    ann_hydro_days2 = []
    ann_dt2 = []
    ann_scd2 = []
    configs = []

    for year_to_take in years_to_take:

        print('loading modis data {}'.format(year_to_take))

        # load modis data for evaluation
        modis_fsca, modis_dt, modis_mask = load_subset_modis_annual(catchment, output_dem, year_to_take, modis_folder, dem_folder, modis_dem, mask_folder,
                                                                    catchment_shp_folder)
        modis_hydro_days = convert_date_hydro_DOY(modis_dt)
        modis_sc = modis_fsca >= modis_sc_threshold

        # print('calculating basin average sca')
        # modis
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
        modis_scd = np.sum(modis_sc, axis=0)
        modis_scd[modis_mask == 0] = -999 # set areas outside catchment to -999
        modis_scd[np.logical_and(modis_sc[0]==np.nan, modis_mask=1)] = -1 # set areas of water to -1

        # add to annual series
        ann_scd_m.append(modis_scd)
        ann_hydro_days_m.append(modis_hydro_days)
        ann_dt_m.append(modis_dt)

        modis_fsca = None
        modis_dt = None
        modis_mask = None
        modis_sc = None
        modis_scd = None

        if which_model == 'clark2009' or which_model == 'all':
            print('loading clark2009 model data {}'.format(year_to_take))
            if clark2009run == False:
                # run model and return timeseries of daily swe, acc and melt.
                st_swe, st_melt, st_acc, out_dt, mask = run_clark2009(catchment, output_dem, year_to_take, met_inp_folder, catchment_shp_folder)
                pickle.dump([st_swe, st_melt, st_acc, out_dt, mask], open(clark2009_output_folder + '/{}_{}_hy{}.pkl'.format(catchment, output_dem,
                                                                                                                             year_to_take), 'wb'), -1)
            elif clark2009run == True:
                # load previously run simulations from pickle file
                st_snow = pickle.load(open(clark2009_output_folder + '/{}_{}_hy{}_clark2009.pkl'.format(catchment, output_dem, year_to_take), 'rb'))
                st_swe = st_snow[0]
                st_melt = st_snow[1]
                st_acc = st_snow[2]
                out_dt = st_snow[3]
                mask = st_snow[4]
                config1 = st_snow[5]
                configs.append(config1)
            # compute timeseries of basin average sca
            num_gridpoints = np.sum(mask)  # st_swe.shape[1] * st_swe.shape[2]
            ba_swe = []
            ba_sca = []
            # ba_melt = []
            # ba_acc = []

            for i in range(st_swe.shape[0]):
                ba_swe.append(np.nanmean(st_swe[i, mask]))  # some points don't have input data, so are nan
                ba_sca.append(np.nansum(st_swe[i, mask] > model_swe_sc_threshold).astype('d') / num_gridpoints)
                # ba_melt.append(np.mean(st_melt[i, mask.astype('int')]))
                # ba_acc.append(np.mean(st_acc[i, mask.astype('int')]))
            # add to annual series
            ann_ts_av_sca.append(np.asarray(ba_sca))
            ann_ts_av_swe.append(np.asarray(ba_swe))
            # ann_ts_av_melt.append(np.asarray(ba_melt))
            # ann_ts_av_acc.append(np.asarray(ba_acc))
            ann_hydro_days.append(convert_date_hydro_DOY(out_dt))
            ann_dt.append(out_dt)
            # calculate snow cover duration
            st_sc = st_swe > model_swe_sc_threshold
            mod1_scd = np.sum(st_sc, axis=0)
            mod1_scd[mask == 0] = -999
            ann_scd.append(mod1_scd)

            # clear arrays
            st_swe = None
            st_melt = None
            st_acc = None
            out_dt = None
            mod1_scd = None
            mask = None
            st_sc = None
            st_snow = None

        if which_model == 'dsc_snow' or which_model == 'all':
            print('loading dsc_snow model data {}'.format(year_to_take))

            if dsc_snow_opt == 'fortran':
                # load previously run simulations from netCDF
                st_swe, st_melt, st_acc, out_dt, mask = load_dsc_snow_output_annual(catchment, output_dem, year_to_take, dsc_snow_output_folder,
                                                                                    dsc_snow_dem_folder, run_id)
            elif dsc_snow_opt == 'python':
                if dsc_snow_opt2 == 'netCDF':
                    st_swe, st_melt, st_acc, out_dt = load_dsc_snow_output_python_otf(catchment, output_dem, year_to_take, dsc_snow_output_folder)
                    # load mask
                    dem = 'si_dem_250m'
                    dem_folder = 'Z:/GIS_DATA/Topography/DEM_NZSOS/'
                    dem_file = dem_folder + dem + '.tif'
                    nztm_dem, x_centres, y_centres, lat_array, lon_array = setup_nztm_dem(dem_file)
                    mask = np.load(mask_folder + '/{}_{}.npy'.format(catchment, dem))
                    _, _, trimmed_mask, _, _ = trim_lat_lon_bounds(mask, lat_array, lon_array, mask.copy(), y_centres, x_centres)
                    mask = trimmed_mask
                elif dsc_snow_opt2 == 'pickle':
                    # load previously run simulations from pickle file
                    st_snow = pickle.load(open(dsc_snow_output_folder + '/{}_{}_hy{}_dsc_snow.pkl'.format(catchment, output_dem, year_to_take), 'rb'))
                    st_swe = st_snow[0]
                    st_melt = st_snow[1]
                    st_acc = st_snow[2]
                    out_dt = st_snow[3]
                    mask = st_snow[4]
                    config2 = st_snow[5]
                    configs.append(config2)
            # print('calculating basin average sca')
            num_gridpoints2 = np.sum(mask)
            ba_swe2 = []
            ba_sca2 = []
            for i in range(st_swe.shape[0]):
                ba_swe2.append(np.nanmean(st_swe[i, mask]))
                ba_sca2.append(np.nansum(st_swe[i, mask] > model_swe_sc_threshold).astype('d') / num_gridpoints2)

            # print('adding to annual series')
            # add to annual timeseries

            if which_model == 'all':
                ann_ts_av_sca2.append(np.asarray(ba_sca2))
                ann_ts_av_swe2.append(np.asarray(ba_swe2))
                # ann_ts_av_melt.append(np.asarray(ba_melt))
                # ann_ts_av_acc.append(np.asarray(ba_acc))
                ann_hydro_days2.append(convert_date_hydro_DOY(out_dt))
                ann_dt2.append(out_dt)
            elif which_model == 'dsc_snow':
                ann_ts_av_sca.append(np.asarray(ba_sca2))
                ann_ts_av_swe.append(np.asarray(ba_swe2))
                # ann_ts_av_melt.append(np.asarray(ba_melt))
                # ann_ts_av_acc.append(np.asarray(ba_acc))
                ann_hydro_days.append(convert_date_hydro_DOY(out_dt))
                ann_dt.append(out_dt)

            # print('calc snow cover duration')
            st_sc = st_swe > model_swe_sc_threshold
            mod_scd = np.sum(st_sc, axis=0)
            mod_scd[mask == 0] = -999

            if which_model == 'all':
                ann_scd2.append(mod_scd)
            elif which_model == 'dsc_snow':
                ann_scd.append(mod_scd)

            # clear arrays
            st_snow = None
            st_swe = None
            st_melt = None
            st_acc = None
            out_dt = None
            mod_scd = None
            mask = None
            st_sc = None

        #
        # if which_model == 'all':
        #     if clark2009run == False:
        #         # run model and return timeseries of daily swe, acc and melt.
        #         st_swe, st_melt, st_acc, out_dt, mask = run_clark2009(catchment, output_dem, year_to_take, met_inp_folder, catchment_shp_folder)
        #     elif clark2009run == True:
        #         # load previously run simulations from pickle file
        #         st_snow = pickle.load(open(clark2009_output_folder + '/{}_{}_hy{}_clark2009.pkl'.format(catchment, output_dem, year_to_take), 'rb'))
        #         st_swe = st_snow[0]
        #         st_melt = st_snow[1]
        #         st_acc = st_snow[2]
        #         out_dt = st_snow[3]
        #         mask = st_snow[4]
        #
        #     # load previously run simulations from netCDF or pickle file
        #     if dsc_snow_opt == 'fortran':
        #         st_swe2, st_melt2, st_acc2, out_dt2, mask2 = load_dsc_snow_output(catchment, output_dem, year_to_take, dsc_snow_output_folder,
        #                                                                           dsc_snow_dem_folder)
        #     elif dsc_snow_opt == 'python':
        #         st_snow2 = pickle.load(open(dsc_snow_output_folder + '/{}_{}_hy{}_dsc_snow.pkl'.format(catchment, output_dem, year_to_take), 'rb'))
        #         st_swe2 = st_snow2[0]
        #         st_melt2 = st_snow2[1]
        #         st_acc2 = st_snow2[2]
        #         out_dt2 = st_snow2[3]
        #         mask2 = st_snow2[4]

    ann = [ann_ts_av_sca_m, ann_hydro_days_m, ann_dt_m, ann_scd_m, ann_ts_av_sca, ann_ts_av_swe, ann_hydro_days, ann_dt, ann_scd, ann_ts_av_sca2,
           ann_ts_av_swe2, ann_hydro_days2, ann_dt2, ann_scd2, ann_ts_av_sca_thres_m, configs]
    pickle.dump(ann, open(
        output_folder + '/summary_{}_{}_thres{}_swe{}_{}_{}.pkl'.format(catchment, output_dem, modis_sc_threshold, model_swe_sc_threshold, which_model, run_id),
        'wb'), -1)
