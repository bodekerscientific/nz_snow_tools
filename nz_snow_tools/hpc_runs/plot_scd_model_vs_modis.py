import numpy as np
import pickle
import copy
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
import matplotlib.dates as mdates
import datetime as dt
from nz_snow_tools.util.utils import convert_datetime_julian_day
from nz_snow_tools.util.utils import setup_nztm_dem, trim_data_to_mask, trim_lat_lon_bounds

#TODO # todos indicate which parameters need to change to switch between VCSN and NZCSM
hydro_years_to_take = np.arange(2018, 2020 + 1)  # [2013 + 1]  # range(2001, 2013 + 1)
plot_folder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/CARH2101/snow reanalysis/NZ/august2021' #TODO
# plot_folder = '/nesi/nobackup/niwa00004/jonoconway/snow_sims_nz'
# model_analysis_area = 145378  # sq km.
catchment = 'NZ'  # string identifying catchment modelled #TODO
mask_folder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/CARH2101/snow reanalysis'
dem_folder = 'C:/Users/conwayjp/OneDrive - NIWA/Data/GIS_DATA/Topography/DEM_NZSOS'
modis_dem = 'modis_nz_dem_250m' #TODO

if modis_dem == 'modis_si_dem_250m':

    si_dem_file = dem_folder + '/si_dem_250m' + '.tif'
    nztm_dem, x_centres, y_centres, lat_array, lon_array = setup_nztm_dem(si_dem_file, extent_w=1.08e6, extent_e=1.72e6, extent_n=5.52e6, extent_s=4.82e6,
                                                                          resolution=250)
    nztm_dem = nztm_dem[:, 20:]
    x_centres = x_centres[20:]
    lat_array = lat_array[:, 20:]
    lon_array = lon_array[:, 20:]
    modis_output_dem = 'si_dem_250m'
    mask = np.load(mask_folder + '/{}_{}.npy'.format(catchment, modis_dem))

elif modis_dem == 'modis_nz_dem_250m':
    si_dem_file = dem_folder + '/nz_dem_250m' + '.tif'
    _, x_centres, y_centres, lat_array, lon_array = setup_nztm_dem(None, extent_w=1.085e6, extent_e=2.10e6, extent_n=6.20e6, extent_s=4.70e6,
                                                                   resolution=250, origin='bottomleft')
    nztm_dem = np.load(dem_folder + '/{}.npy'.format(modis_dem))
    modis_output_dem = 'modis_nz_dem_250m'
    mask = np.load(mask_folder + '/{}_{}.npy'.format(catchment,
                                                     modis_dem))  # just load the mask the chooses land points from the dem. snow data has modis hy2018_2020 landpoints mask applied in NZ_evaluation_otf
    # mask = np.load("C:/Users/conwayjp/OneDrive - NIWA/projects/CARH2101/snow reanalysis/modis_mask_hy2018_2020_landpoints.npy")

lat_array, lon_array, nztm_dem, y_centres, x_centres = trim_lat_lon_bounds(mask, lat_array, lon_array, nztm_dem, y_centres, x_centres)

# # modis options
modis_sc_threshold = 50  # value of fsca (in percent) that is counted as being snow covered
modis_output_folder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/CARH2101/snow reanalysis'
# modis_output_folder = '/nesi/nobackup/niwa00004/jonoconway/snow_sims_nz'

[ann_ts_av_sca_m, ann_ts_av_sca_thres_m, ann_dt_m, ann_scd_m] = pickle.load(open(
    modis_output_folder + '/summary_MODIS_{}_{}_{}_{}_thres{}.pkl'.format(hydro_years_to_take[0], hydro_years_to_take[-1], catchment, modis_output_dem,
                                                                          modis_sc_threshold), 'rb'))
# model options

run_id = 'cl09_default_ros'  ## 'cl09_tmelt275'#'cl09_default' #'cl09_tmelt275_ros' ##TODO
which_model = 'clark2009'  #TODO
# run_id = 'dsc_default'  #'dsc_mueller_TF2p4_tmelt278_ros'  #
# which_model = 'dsc_snow'  # 'clark2009'  # 'dsc_snow'#
met_inp = 'nzcsm7-12'  # 'vcsn_norton'#'nzcsm7-12'#vcsn_norton' #nzcsm7-12'  # 'vcsn_norton' #   # identifier for input meteorology #TODO

output_dem = 'nz_dem_250m' #TODO
model_swe_sc_threshold = 30  # threshold for treating a grid cell as snow covered (mm w.e)#TODO
model_output_folder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/CARH2101/snow reanalysis'
# model_output_folder = '/nesi/nobackup/niwa00004/jonoconway/snow_sims_nz/nzcsm'


[ann_ts_av_swe, ann_ts_av_sca_thres, ann_dt, ann_scd, ann_av_swe, ann_max_swe, ann_metadata] = pickle.load(open(
    model_output_folder + '/summary_MODEL_{}_{}_{}_{}_{}_{}_{}_thres{}.pkl'.format(hydro_years_to_take[0], hydro_years_to_take[-1], met_inp, which_model,
                                                                                   catchment, output_dem, run_id, model_swe_sc_threshold), 'rb'))
# cut down model data to trimmed modis SI domain.

if modis_dem == 'modis_si_dem_250m':
    ann_scd = [trim_data_to_mask(a, mask) for a in ann_scd]
    ann_max_swe = [trim_data_to_mask(a, mask) for a in ann_max_swe]
    ann_av_swe = [trim_data_to_mask(a, mask) for a in ann_av_swe]
modis_scd = np.nanmean(ann_scd_m, axis=0)
model_scd = np.nanmean(ann_scd, axis=0)

plot_scd_bias = model_scd - modis_scd

# perm_snow = np.load('C:/Users/conwayjp/OneDrive - NIWA/projects/DSC Snow/MODIS/modis_permanent_snow_2010_2016.npy')
#
# plot_scd_bias[perm_snow] == np.nan

subsets = {
    'Arthurs Pass': {
        'xlim': [1.47e6, 1.50e6],
        'ylim': [5.23e6, 5.26e6]
    },
    'St Bathans and Hawkduns': {
        'xlim': [1.33e6, 1.38e6],
        'ylim': [5.02e6, 5.07e6]
    },
    'Tititea-Mt Aspring': {
        'xlim': [1.25e6, 1.28e6],
        'ylim': [5.055e6, 5.085e6]
    },
    'Doubtful Sound': {
        'xlim': [1.13e6, 1.16e6],
        'ylim': [4.935e6, 4.965e6]
    },
    'Aoraki-Mt Cook': {
        'xlim': [1.36e6, 1.39e6],
        'ylim': [5.145e6, 5.175e6]
    },
    'HaastPass': {
        'xlim': [1.295e6, 1.32e6],
        'ylim': [5.095e6, 5.12e6]
    },
    'Cardrona': {
        'xlim': [1.27e6, 1.3e6],
        'ylim': [5.015e6, 5.045e6]
    },
    'Central-Nth-Island': {
        'xlim': [1.66e6, 1.91e6],
        'ylim': [5.41e6, 5.69e6]
    },
    'South-Island': {
        'xlim': [1.08e6, 1.72e6],
        'ylim': [4.82e6, 5.52e6]
    },

    'Ruapehu': {
        'xlim': [1.81e6, 1.84e6],
        'ylim': [5.64e6, 5.67e6]
    }

}

plt.rcParams.update({'font.size': 6})
plt.rcParams.update({'axes.titlesize': 6})

# plt.figure()
# ind = np.logical_and(modis_scd>1,model_scd>1)
# plt.hexbin(modis_scd[ind].ravel(), model_scd[ind].ravel(),gridsize=30,bins='log')
# plt.plot([0,365],[0,365],'k--',alpha=.5)
# plt.colorbar()
# plt.xlabel('MODIS SCD')
# plt.ylabel('Model SCD')
# plt.yticks(range(0,361,90))
# plt.xticks(range(0,361,90))
# plt.savefig(plot_folder + '/SCD hexbin HY {} to {} {}_{}_{}_{}_{}_swe_thres{}_sca_thres{}.png'.format(hydro_years_to_take[0], hydro_years_to_take[-1],
#                                                                                                                 met_inp, which_model,
#                                                                                                                 catchment, output_dem, run_id,
#                                                                                                                 model_swe_sc_threshold,
#                                                                                                                 modis_sc_threshold), dpi=600)

# plot change in elevation of snowline (120 days SCD)

# model_scd = np.nanmean(ann_scd, axis=0)
# mask = np.load("C:/Users/conwayjp/OneDrive - NIWA/projects/CARH2101/snow reanalysis/modis_mask_hy2018_2020_landpoints.npy")
# model_scd[~mask] = np.nan
# model_snowline = []
# modis_snowline = []
# for bin_s in np.arange(4.7e6, 6.2e6, 5.e4):
#     # select points within northing range\
#     ind_y = np.logical_and(y_centres > bin_s, y_centres <= bin_s + 5e4)
#     model_scd_y = model_scd[ind_y, :]
#     modis_scd_y = modis_scd[ind_y, :]
#     dem_y = nztm_dem[ind_y, :]
#     model_snowline.append(np.nanmean(dem_y[np.logical_and(model_scd_y > 110, model_scd_y < 130)]))
#     modis_snowline.append(np.nanmean(dem_y[np.logical_and(modis_scd_y > 110, modis_scd_y < 130)]))

# plt.plot(modis_snowline, np.arange(4.7e6, 6.2e6, 5.e4) + 2.5e4)
# plt.plot(model_snowline, np.arange(4.7e6, 6.2e6, 5.e4) + 2.5e4)
# plt.grid()

# plt.legend()
# plt.ylabel('NZTM northing (m)')
# plt.xlabel('Elevation (m)')
# plt.tight_layout()
# plt.savefig(r'C:\Users\conwayjp\OneDrive - NIWA\projects\CARH2101\snow reanalysis\snowline elevation with northing.png',dpi=300)


model_swe = np.nanmean(ann_av_swe, axis=0)
mean_swe = np.full(np.arange(0, 3600 + 1, 200).shape, np.nan)
area = np.full(np.arange(0, 3600 + 1, 200).shape, np.nan)
for i, x in enumerate(np.arange(0, 3600 + 1, 200)):
    mean_swe[i] = np.nanmean(model_swe[np.logical_and(nztm_dem > x, nztm_dem <= x + 200)])
    area[i] = np.nansum(np.logical_and(nztm_dem > x, nztm_dem <= x + 200)) * .25 * .25
fig, ax = plt.subplots(figsize=(4, 4))
# plt.barh(np.arange(0, 3600 + 1, 200) + 100, mean_swe_dsc * area / 1e6, height=200, label='dsc_snow')
plt.barh(np.arange(0, 3600 + 1, 200) + 100, mean_swe * area / 1e6, height=200, label='clark')
plt.yticks(np.arange(0, 3600 + 1, 400))
plt.ylim(0, 3600)
plt.ylabel('Elevation (m)')
plt.xlabel('Average snow storage (cubic km)')
plt.tight_layout()
# fig.savefig(plot_folder + '/hist av snow storage clark.png')
#
# model_max_swe = np.nanmean(ann_max_swe, axis=0)
# max_swe = np.full(np.arange(0, 3600 + 1, 200).shape, np.nan)
# area = np.full(np.arange(0, 3600 + 1, 200).shape, np.nan)
# for i, x in enumerate(np.arange(0, 3600 + 1, 200)):
#     max_swe[i] = np.nanmean(model_max_swe[np.logical_and(nztm_dem > x, nztm_dem <= x + 200)])
#     area[i] = np.nansum(np.logical_and(nztm_dem > x, nztm_dem <= x + 200)) * .25 * .25
# fig, ax = plt.subplots(figsize=(4, 4))
# # plt.barh(np.arange(0, 3600 + 1, 200) + 100, mean_swe_dsc * area / 1e6, height=200, label='dsc_snow')
# plt.barh(np.arange(0, 3600 + 1, 200) + 100, max_swe * area / 1e6, height=200, label='clark')
# plt.yticks(np.arange(0, 3600 + 1, 400))
# plt.ylim(0, 3600)
# plt.ylabel('Elevation (m)')
# plt.xlabel('Max snow storage (cubic km)')
# plt.tight_layout()
# fig.savefig(plot_folder + '/hist max snow storage clark.png')


fig1 = plt.figure(figsize=[4, 4])
bin_edges = [-60, -30, -7, 7, 30, 60]  # use small negative number to include 0 in the interpolation
CS1 = plt.contourf(x_centres, y_centres, plot_scd_bias, levels=bin_edges, cmap=copy.copy(plt.cm.RdBu), extend='both')
# CS1.cmap.set_bad('grey')
# CS1.cmap.set_over([0.47,0.72,0.77])
plt.gca().set_aspect('equal')
fig1.gca().set_facecolor('grey')
# plt.imshow(modis_scd, origin=0, interpolation='none', vmin=0, vmax=365, cmap='magma_r')
plt.xticks([])
plt.yticks([])
cbar = plt.colorbar()
cbar.set_label('Snow cover duration (days)', rotation=90)
plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
plt.ylabel('NZTM northing')
plt.xlabel('NZTM easting')
plt.title('SCD Bias: Model-MODIS {} to {}'.format(hydro_years_to_take[0], hydro_years_to_take[-1]))
plt.xlim((np.min(x_centres), np.max(x_centres)))
plt.ylim((np.min(y_centres), np.max(y_centres)))
if catchment == 'SI':
    plt.xticks(np.arange(12e5, 17e5, 2e5))
    plt.yticks(np.arange(50e5, 55e5, 2e5))
else:
    plt.xticks(np.arange(12e5, 21e5, 2e5))
    plt.yticks(np.arange(48e5, 63e5, 2e5))

plt.tight_layout()
plt.savefig(plot_folder + '/SCD bias model-modis HY {} to {} {}_{}_{}_{}_{}_swe_thres{}_sca_thres{}.png'.format(hydro_years_to_take[0], hydro_years_to_take[-1],
                                                                                                                met_inp, which_model,
                                                                                                                catchment, output_dem, run_id,
                                                                                                                model_swe_sc_threshold,
                                                                                                                modis_sc_threshold), dpi=600)
name = 'Central-Nth-Island'
plt.title('(f) SCD Bias:Model-MODIS',fontweight='bold',fontsize=10)
plt.xlim(subsets[name]['xlim'])
plt.ylim(subsets[name]['ylim'])
plt.xticks(np.arange(subsets[name]['xlim'][0], subsets[name]['xlim'][1] + 1, 100e3))
plt.yticks(np.arange(subsets[name]['ylim'][0], subsets[name]['ylim'][1] + 1, 100e3))
plt.tight_layout()
plt.savefig(
    plot_folder + '/SCD bias model-modis {} HY {} to {} {}_{}_{}_{}_{}_swe_thres{}_sca_thres{}_no_contour.png'.format(name, hydro_years_to_take[0],
                                                                                                                      hydro_years_to_take[-1],
                                                                                                                      met_inp, which_model,
                                                                                                                      catchment, output_dem, run_id,
                                                                                                                      model_swe_sc_threshold,
                                                                                                                      modis_sc_threshold), dpi=600)
name = 'South-Island'
plt.title('(e) SCD Bias:Model-MODIS',fontweight='bold',fontsize=10)
plt.xlim(subsets[name]['xlim'])
plt.ylim(subsets[name]['ylim'])
plt.xticks(np.arange(subsets[name]['xlim'][0], subsets[name]['xlim'][1] + 1, 200e3))
plt.yticks(np.arange(subsets[name]['ylim'][0], subsets[name]['ylim'][1] + 1, 200e3))
plt.tight_layout()
plt.savefig(
    plot_folder + '/SCD bias model-modis {} HY {} to {} {}_{}_{}_{}_{}_swe_thres{}_sca_thres{}_no_contour.png'.format(name, hydro_years_to_take[0],
                                                                                                                      hydro_years_to_take[-1],
                                                                                                                      met_inp, which_model,
                                                                                                                      catchment, output_dem, run_id,
                                                                                                                      model_swe_sc_threshold,
                                                                                                                      modis_sc_threshold), dpi=600)

plt.contour(x_centres, y_centres, nztm_dem, np.arange(0, 4000, 200), colors='k', linewidths=0.5)
plt.contour(x_centres, y_centres, nztm_dem, np.arange(0, 4000, 1000), colors='k', linewidths=1)
plt.title('(c) SCD Bias: Model-MODIS',fontweight='bold',fontsize=10)

for name in subsets.keys():
    plt.xlim(subsets[name]['xlim'])
    plt.ylim(subsets[name]['ylim'])
    plt.xticks(np.arange(subsets[name]['xlim'][0], subsets[name]['xlim'][1] + 1, 5e3))
    plt.yticks(np.arange(subsets[name]['ylim'][0], subsets[name]['ylim'][1] + 1, 5e3))
    plt.tight_layout()
    plt.savefig(
        plot_folder + '/SCD bias model-modis {} HY {} to {} {}_{}_{}_{}_{}_swe_thres{}_sca_thres{}.png'.format(name, hydro_years_to_take[0],
                                                                                                               hydro_years_to_take[-1],
                                                                                                               met_inp, which_model,
                                                                                                               catchment, output_dem, run_id,
                                                                                                               model_swe_sc_threshold,
                                                                                                               modis_sc_threshold), dpi=600)
# plt.clf()


bin_edges = [-0.001, 30, 60, 90, 120, 180, 270, 360]  # use small negative number to include 0 in the interpolation

fig1 = plt.figure(figsize=[4, 4])
CS1 = plt.contourf(x_centres, y_centres, modis_scd, levels=bin_edges, cmap=copy.copy(plt.cm.get_cmap('magma_r')), extend='max')
# CS1.cmap.set_bad('grey')
CS1.cmap.set_over([0.47, 0.72, 0.77])
# CS1.cmap.set_under('none')
plt.gca().set_aspect('equal')
# plt.imshow(modis_scd, origin=0, interpolation='none', vmin=0, vmax=365, cmap='magma_r')
plt.xticks([])
plt.yticks([])
cbar = plt.colorbar()
cbar.set_label('Snow cover duration (days)', rotation=90)
plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
plt.ylabel('NZTM northing')
plt.xlabel('NZTM easting')
plt.title('MODIS mean SCD {} to {}'.format(hydro_years_to_take[0], hydro_years_to_take[-1]))
plt.xlim((np.min(x_centres), np.max(x_centres)))
plt.ylim((np.min(y_centres), np.max(y_centres)))
if catchment == 'SI':
    plt.xticks(np.arange(12e5, 17e5, 2e5))
    plt.yticks(np.arange(50e5, 55e5, 2e5))
else:
    plt.xticks(np.arange(12e5, 21e5, 2e5))
    plt.yticks(np.arange(48e5, 63e5, 2e5))

plt.tight_layout()
plt.savefig(
    plot_folder + '/SCD modis mean HY {} to {} thres{} {}.png'.format(hydro_years_to_take[0], hydro_years_to_take[-1], modis_sc_threshold, catchment), dpi=300)

name = 'Central-Nth-Island'
plt.title('(d) MODIS mean SCD',fontweight='bold',fontsize=10)
plt.xlim(subsets[name]['xlim'])
plt.ylim(subsets[name]['ylim'])
plt.xticks(np.arange(subsets[name]['xlim'][0], subsets[name]['xlim'][1] + 1, 100e3))
plt.yticks(np.arange(subsets[name]['ylim'][0], subsets[name]['ylim'][1] + 1, 100e3))
plt.tight_layout()
plt.savefig(
    plot_folder + '/SCD modis mean {} HY {} to {} thres{} {} no contour.png'.format(name, hydro_years_to_take[0], hydro_years_to_take[-1], modis_sc_threshold,
                                                                                    catchment),
    dpi=300)

name = 'South-Island'
plt.title('(c) MODIS mean SCD',fontweight='bold',fontsize=10)
plt.xlim(subsets[name]['xlim'])
plt.ylim(subsets[name]['ylim'])
plt.xticks(np.arange(subsets[name]['xlim'][0], subsets[name]['xlim'][1] + 1, 200e3))
plt.yticks(np.arange(subsets[name]['ylim'][0], subsets[name]['ylim'][1] + 1, 200e3))
plt.tight_layout()
plt.savefig(
    plot_folder + '/SCD modis mean {} HY {} to {} thres{} {} no contour.png'.format(name, hydro_years_to_take[0], hydro_years_to_take[-1], modis_sc_threshold,
                                                                                    catchment),
    dpi=300)

plt.contour(x_centres, y_centres, nztm_dem, np.arange(0, 4000, 200), colors='k', linewidths=0.5)
plt.contour(x_centres, y_centres, nztm_dem, np.arange(0, 4000, 1000), colors='k', linewidths=1)
plt.title('(b) MODIS mean SCD',fontweight='bold',fontsize=10)

for name in subsets.keys():
    plt.xlim(subsets[name]['xlim'])
    plt.ylim(subsets[name]['ylim'])
    plt.xticks(np.arange(subsets[name]['xlim'][0], subsets[name]['xlim'][1] + 1, 5e3))
    plt.yticks(np.arange(subsets[name]['ylim'][0], subsets[name]['ylim'][1] + 1, 5e3))
    plt.tight_layout()
    plt.savefig(
        plot_folder + '/SCD modis mean {} HY {} to {} thres{} {}.png'.format(name, hydro_years_to_take[0], hydro_years_to_take[-1], modis_sc_threshold,
                                                                             catchment),
        dpi=300)

fig1 = plt.figure(figsize=[4, 4])
CS1 = plt.contourf(x_centres, y_centres, model_scd, levels=bin_edges, cmap=copy.copy(plt.cm.get_cmap('magma_r')), extend='max')
# CS1.cmap.set_bad('grey')
CS1.cmap.set_over([0.47, 0.72, 0.77])
# CS1.cmap.set_under('none')
plt.gca().set_aspect('equal')
# plt.imshow(modis_scd, origin=0, interpolation='none', vmin=0, vmax=365, cmap='magma_r')
plt.xticks([])
plt.yticks([])
cbar = plt.colorbar()
cbar.set_label('Snow cover duration (days)', rotation=90)
plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
plt.ylabel('NZTM northing')
plt.xlabel('NZTM easting')
plt.title('Model mean SCD HY {} to {}'.format(hydro_years_to_take[0], hydro_years_to_take[-1]))
plt.xlim((np.min(x_centres), np.max(x_centres)))
plt.ylim((np.min(y_centres), np.max(y_centres)))
if catchment == 'SI':
    plt.xticks(np.arange(12e5, 17e5, 2e5))
    plt.yticks(np.arange(50e5, 55e5, 2e5))
else:
    plt.xticks(np.arange(12e5, 21e5, 2e5))
    plt.yticks(np.arange(48e5, 63e5, 2e5))
plt.tight_layout()
plt.savefig(
    plot_folder + '/SCD model mean HY {} to {} thres{} {} {} {} {}.png'.format(hydro_years_to_take[0], hydro_years_to_take[-1], model_swe_sc_threshold, run_id,
                                                                               met_inp, which_model, catchment), dpi=300)

name = 'Central-Nth-Island'
plt.title('(b) Model mean SCD',fontweight='bold',fontsize=10)

plt.xlim(subsets[name]['xlim'])
plt.ylim(subsets[name]['ylim'])
plt.xticks(np.arange(subsets[name]['xlim'][0], subsets[name]['xlim'][1], 100e3))
plt.yticks(np.arange(subsets[name]['ylim'][0], subsets[name]['ylim'][1], 100e3))
plt.tight_layout()
plt.savefig(
    plot_folder + '/SCD model mean {} HY {} to {} thres{} {} {} {} {} no contour.png'.format(name, hydro_years_to_take[0], hydro_years_to_take[-1],
                                                                                             model_swe_sc_threshold,
                                                                                             run_id,
                                                                                             met_inp, which_model, catchment), dpi=300)

name = 'South-Island'
plt.title('(a) Model mean SCD',fontweight='bold',fontsize=10)
plt.xlim(subsets[name]['xlim'])
plt.ylim(subsets[name]['ylim'])
plt.xticks(np.arange(subsets[name]['xlim'][0], subsets[name]['xlim'][1], 100e3))
plt.yticks(np.arange(subsets[name]['ylim'][0], subsets[name]['ylim'][1], 100e3))
plt.tight_layout()
plt.savefig(
    plot_folder + '/SCD model mean {} HY {} to {} thres{} {} {} {} {} no contour.png'.format(name, hydro_years_to_take[0], hydro_years_to_take[-1],
                                                                                             model_swe_sc_threshold,
                                                                                             run_id,
                                                                                             met_inp, which_model, catchment), dpi=300)

plt.contour(x_centres, y_centres, nztm_dem, np.arange(0, 4000, 200), colors='k', linewidths=0.5)
plt.contour(x_centres, y_centres, nztm_dem, np.arange(0, 4000, 1000), colors='k', linewidths=1)
plt.title('(a) Model mean SCD',fontweight='bold',fontsize=10)

for name in subsets.keys():
    plt.xlim(subsets[name]['xlim'])
    plt.ylim(subsets[name]['ylim'])
    plt.xticks(np.arange(subsets[name]['xlim'][0], subsets[name]['xlim'][1] + 1, 5e3))
    plt.yticks(np.arange(subsets[name]['ylim'][0], subsets[name]['ylim'][1] + 1, 5e3))
    plt.tight_layout()
    plt.savefig(
        plot_folder + '/SCD model mean {} HY {} to {} thres{} {} {} {} {}.png'.format(name, hydro_years_to_take[0], hydro_years_to_take[-1],
                                                                                      model_swe_sc_threshold,
                                                                                      run_id,
                                                                                      met_inp, which_model, catchment), dpi=300)

# plot years separately
for i, year_to_take in enumerate(hydro_years_to_take):
    fig1 = plt.figure(figsize=[4, 4])
    print('loading data for year {}'.format(year_to_take))
    model_scd_1year = ann_scd[i]
    CS1 = plt.contourf(x_centres, y_centres, model_scd_1year, levels=bin_edges, cmap=copy.copy(plt.cm.get_cmap('magma_r')), extend='max')
    # CS1.cmap.set_bad('grey')
    CS1.cmap.set_over([0.47, 0.72, 0.77])
    # CS1.cmap.set_under('none')
    plt.gca().set_aspect('equal')
    # plt.imshow(modis_scd, origin=0, interpolation='none', vmin=0, vmax=365, cmap='magma_r')
    plt.xticks([])
    plt.yticks([])
    cbar = plt.colorbar()
    cbar.set_label('Snow cover duration (days)', rotation=90)
    plt.xticks(np.arange(12e5, 21e5, 2e5))
    plt.yticks(np.arange(48e5, 63e5, 2e5))
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    plt.ylabel('NZTM northing')
    plt.xlabel('NZTM easting')
    plt.title('Snow cover duration HY{}'.format(year_to_take))
    plt.tight_layout()
    plt.savefig(plot_folder + '/SCD model HY{} thres{} {} {} {} {}.png'.format(year_to_take, model_swe_sc_threshold, run_id, met_inp, which_model, catchment),
                dpi=300)

for i, year_to_take in enumerate(hydro_years_to_take):
    fig1 = plt.figure(figsize=[4, 4])
    print('loading data for year {}'.format(year_to_take))
    modis_scd_1year = ann_scd_m[i]
    CS1 = plt.contourf(x_centres, y_centres, modis_scd_1year, levels=bin_edges, cmap=copy.copy(plt.cm.get_cmap('magma_r')), extend='max')
    # CS1.cmap.set_bad('grey')
    CS1.cmap.set_over([0.47, 0.72, 0.77])
    # CS1.cmap.set_under('none')
    plt.gca().set_aspect('equal')
    # plt.imshow(modis_scd, origin=0, interpolation='none', vmin=0, vmax=365, cmap='magma_r')
    plt.xticks([])
    plt.yticks([])
    cbar = plt.colorbar()
    cbar.set_label('Snow cover duration (days)', rotation=90)
    plt.xticks(np.arange(12e5, 21e5, 2e5))
    plt.yticks(np.arange(48e5, 63e5, 2e5))
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    plt.ylabel('NZTM northing')
    plt.xlabel('NZTM easting')
    plt.title('Snow cover duration HY{}'.format(year_to_take))
    plt.tight_layout()
    plt.savefig(plot_folder + '/SCD modis HY{} thres{} {} {}.png'.format(year_to_take, modis_sc_threshold, run_id, catchment), dpi=300)

# plot modis scd vs elevation
fig1 = plt.figure(figsize=[4, 4])
h2d = plt.hist2d(modis_scd.ravel()[~np.isnan(modis_scd.ravel())], nztm_dem.ravel()[~np.isnan(modis_scd.ravel())],
                 bins=(np.arange(0, 420, 30), np.arange(0, 4000, 200)), norm=LogNorm())
plt.colorbar()
plt.xticks(h2d[1][:-1])
plt.yticks(h2d[2][:-1])
plt.ylabel('Elevation (m)')
plt.xlabel('SCD (days)')
plt.title('(b) MODIS SCD by elevation',fontweight='bold',fontsize=10)
mode_modis = np.argmax(h2d[0], axis=0)
scd_x = h2d[1][:-1]
mode_modis_scd = np.asarray([scd_x[m] for m in mode_modis])
mode_modis_scd += 15
# plt.scatter(mode_modis_scd,h2d[2][:-1]+100,18,'k')
plt.scatter(mode_modis_scd, h2d[2][:-1] + 100, 18, 'k', label='mode (MODIS)')
plt.legend(loc='upper left')
plt.tight_layout()
fig1.savefig(
    plot_folder + '/hist elevation modis SCD with mode HY{}to{} thres{} {}.png'.format(hydro_years_to_take[0], hydro_years_to_take[-1], modis_sc_threshold,
                                                                                       run_id),
    dpi=300)

# plot model scd vs elevation
fig1 = plt.figure(figsize=[4, 4])
h2d = plt.hist2d(model_scd.ravel()[~np.isnan(model_scd.ravel())], nztm_dem.ravel()[~np.isnan(model_scd.ravel())],
                 bins=(np.arange(0, 420, 30), np.arange(0, 4000, 200)), norm=LogNorm())
plt.colorbar()
plt.xticks(h2d[1][:-1])
plt.yticks(h2d[2][:-1])
plt.ylabel('Elevation (m)')
plt.xlabel('SCD (days)')
plt.title('(a) Model SCD by elevation',fontweight='bold',fontsize=10)
mode = np.argmax(h2d[0], axis=0)
scd_x = h2d[1][:-1]
mode_scd = np.asarray([scd_x[m] for m in mode])
mode_scd += 15
# plt.scatter(mode_modis_scd,h2d[2][:-1]+100,18,'k')
# plt.scatter(mode_scd,h2d[2][:-1]+100,6,'r')
plt.scatter(mode_scd, h2d[2][:-1] + 100, 18, 'b', label='mode (Model)')
plt.scatter(mode_modis_scd, h2d[2][:-1] + 100,18, label='mode (MODIS)',facecolor='none',edgecolor='k')
plt.legend(loc='upper left')
plt.tight_layout()
fig1.savefig(
    plot_folder + '/hist elevation model SCD with mode HY {} to {} {}_{}_{}_{}_{}_swe_thres{}.png'.format(hydro_years_to_take[0], hydro_years_to_take[-1],
                                                                                                          met_inp, which_model,
                                                                                                          catchment, output_dem, run_id,
                                                                                                          model_swe_sc_threshold), dpi=600)

# plot histogram of bias vs elevation
fig1 = plt.figure(figsize=[4, 4])
h2d = plt.hist2d(plot_scd_bias.ravel()[~np.isnan(plot_scd_bias.ravel())], nztm_dem.ravel()[~np.isnan(plot_scd_bias.ravel())],
                 bins=(np.linspace(-180, 180, 30), np.arange(0, 4000, 200)), norm=LogNorm())
# plt.xticks(h2d[1])
plt.yticks(h2d[2][:-1])
plt.colorbar()
plt.ylabel('Elevation (m)')
plt.xlabel('SCD bias [model-modis] (days)')
plt.title('Model SCD bias by elevation')
plt.tight_layout()
fig1.savefig(plot_folder + '/hist elevation model SCD bias HY {} to {} {}_{}_{}_{}_{}_swe_thres{}_sca_thres{}.png'.format(hydro_years_to_take[0],
                                                                                                                          hydro_years_to_take[-1],
                                                                                                                          met_inp, which_model,
                                                                                                                          catchment, output_dem, run_id,
                                                                                                                          model_swe_sc_threshold,
                                                                                                                          modis_sc_threshold), dpi=600)

# plot histogram of bias vs scd
fig1 = plt.figure(figsize=[4, 4])
h2d = plt.hist2d(plot_scd_bias.ravel()[~np.isnan(plot_scd_bias.ravel())], modis_scd.ravel()[~np.isnan(plot_scd_bias.ravel())],
                 bins=(np.linspace(-180, 180, 30), np.arange(0, 420, 30)), norm=LogNorm())
plt.colorbar()
plt.yticks(h2d[2][:-1])
plt.xlabel('SCD bias [model-modis] (days)')
plt.ylabel('MODIS SCD (days)')
plt.tight_layout()
fig1.savefig(plot_folder + '/hist modis SCD model SCD bias HY {} to {} {}_{}_{}_{}_{}_swe_thres{}_sca_thres{}.png'.format(hydro_years_to_take[0],
                                                                                                                          hydro_years_to_take[-1],
                                                                                                                          met_inp, which_model,
                                                                                                                          catchment, output_dem, run_id,
                                                                                                                          model_swe_sc_threshold,
                                                                                                                          modis_sc_threshold), dpi=600)

h2d_modis_av = plt.hist2d(modis_scd.ravel()[~np.isnan(plot_scd_bias.ravel())], nztm_dem.ravel()[~np.isnan(plot_scd_bias.ravel())],
                          bins=(np.arange(0, 365 + 1, 10), np.arange(0, 3600 + 1, 500)), norm=LogNorm())
h2d_mod2 = plt.hist2d(model_scd.ravel()[~np.isnan(plot_scd_bias.ravel())], nztm_dem.ravel()[~np.isnan(plot_scd_bias.ravel())],
                      bins=(np.arange(0, 365 + 1, 10), np.arange(0, 3600 + 1, 500)), norm=LogNorm())
colors = plt.cm.tab10
fig, ax = plt.subplots(figsize=(4, 4))
for i in np.arange(0, 5):
    ax.semilogy(h2d_mod2[1][1:] - 5, h2d_mod2[0][:, i] / 16, '-.',
                color=colors(i / 10))  # , label='model cl {}-{} m'.format(h2d_mod2[2][i], h2d_mod2[2][i + 1]))
    ax.semilogy(h2d_modis_av[1][1:] - 5, h2d_modis_av[0][:, i] / 16, '-', color=colors(i / 10),
                label='{}-{} m'.format(h2d_modis_av[2][i], h2d_modis_av[2][i + 1]))
plt.legend()
plt.ylim(bottom=10)
plt.xticks(np.arange(0, 365, 30))
plt.ylabel('Area  (km^2)')
plt.xlabel('SCD')
fig.savefig(plot_folder + '/hist modis vs model SCD HY {} to {} {}_{}_{}_{}_{}_swe_thres{}_sca_thres{}.png'.format(hydro_years_to_take[0],
                                                                                                                   hydro_years_to_take[-1],
                                                                                                                   met_inp, which_model,
                                                                                                                   catchment, output_dem, run_id,
                                                                                                                   model_swe_sc_threshold,
                                                                                                                   modis_sc_threshold), dpi=600)
plt.figure()
h2d_modis_av = plt.hist2d(modis_scd.ravel()[~np.isnan(plot_scd_bias.ravel())], nztm_dem.ravel()[~np.isnan(plot_scd_bias.ravel())],
                          bins=(np.arange(0, 365 + 1, 10), np.arange(1000, 3600 + 1, 500)), norm=LogNorm())
h2d_mod2 = plt.hist2d(model_scd.ravel()[~np.isnan(plot_scd_bias.ravel())], nztm_dem.ravel()[~np.isnan(plot_scd_bias.ravel())],
                      bins=(np.arange(0, 365 + 1, 10), np.arange(1000, 3600 + 1, 500)), norm=LogNorm())
colors = plt.cm.tab10
fig, ax = plt.subplots(figsize=(4, 4))
for i in np.arange(0, 3):
    ax.plot(h2d_mod2[1][1:] - 5, h2d_mod2[0][:, i] / np.sum(h2d_mod2[0][:, i]), '--', color=colors(i / 10),
            label='MODEL {}-{} m'.format(h2d_mod2[2][i], h2d_mod2[2][i + 1]))  # , label='model cl {}-{} m'.format(h2d_mod2[2][i], h2d_mod2[2][i + 1]))
    ax.plot(h2d_modis_av[1][1:] - 5, h2d_modis_av[0][:, i] / np.sum(h2d_modis_av[0][:, i]), '-', color=colors(i / 10),
            label='MODIS {}-{} m'.format(h2d_modis_av[2][i], h2d_modis_av[2][i + 1]))
# plt.text(0,0.19,'(c)',fontsize=10,fontweight='bold')
plt.legend()
plt.xticks(np.arange(0, 365, 30))
plt.ylabel('Fraction of points')
plt.xlabel('SCD (days)')
plt.title('(c) SCD in elevation bands',fontsize=10,fontweight='bold')
plt.tight_layout()
fig.savefig(plot_folder + '/hist modis vs model linear SCD HY {} to {} {}_{}_{}_{}_{}_swe_thres{}_sca_thres{}.png'.format(hydro_years_to_take[0],
                                                                                                                   hydro_years_to_take[-1],
                                                                                                                   met_inp, which_model,
                                                                                                                   catchment, output_dem, run_id,
                                                                                                                   model_swe_sc_threshold,
                                                                                                                   modis_sc_threshold), dpi=600)

#
# h2d = plt.hist2d(modis_scd.ravel()[~np.isnan(plot_scd_bias.ravel())], nztm_dem.ravel()[~np.isnan(plot_scd_bias.ravel())],
#                  bins=(np.arange(0, 365 + 1, 10), np.arange(0, 3600 + 1, 500)), norm=LogNorm())
#
# h2d_mod = plt.hist2d(model_scd.ravel()[~np.isnan(plot_scd_bias.ravel())], nztm_dem.ravel()[~np.isnan(plot_scd_bias.ravel())],
#                      bins=(np.arange(0, 365 + 1, 10), np.arange(0, 3600 + 1, 500)), norm=LogNorm())
#
# run_id = 'cl09_default'  # 'dsc_default'#'cl09_default'  # #'cl09_default'# #  #  #
# which_model = 'clark2009'  # 'dsc_snow'# # # ### 'dsc_snow'  #
#
# [ann_ts_av_swe, ann_ts_av_sca_thres, ann_dt, ann_scd, ann_av_swe, ann_max_swe, ann_metadata] = pickle.load(open(
#     model_output_folder + '/summary_MODEL_{}_{}_{}_{}_{}_{}_{}_thres{}.pkl'.format(hydro_years_to_take[0], hydro_years_to_take[-1], met_inp, which_model,
#                                                                                    catchment, output_dem, run_id, model_swe_sc_threshold), 'rb'))
# ann_scd = [trim_data_to_mask(a, mask) for a in ann_scd]
# model_scd = np.nanmean(ann_scd, axis=0)
# h2d_mod2 = plt.hist2d(model_scd.ravel()[~np.isnan(plot_scd_bias.ravel())], nztm_dem.ravel()[~np.isnan(plot_scd_bias.ravel())],
#                       bins=(np.arange(0, 365 + 1, 10), np.arange(0, 3600 + 1, 500)), norm=LogNorm())
#
# colors = plt.cm.tab10
# fig, ax = plt.subplots(figsize=(4, 4))
# for i in np.arange(0, 5):
#     ax.semilogy(h2d_mod2[1][1:] - 5, h2d_mod2[0][:, i] / 16, '-.',
#                 color=colors(i / 10))  # , label='model cl {}-{} m'.format(h2d_mod2[2][i], h2d_mod2[2][i + 1]))
#     ax.semilogy(h2d_mod[1][1:] - 5, h2d_mod[0][:, i] / 16, '--', color=colors(i / 10))  # , label='model dsc {}-{} m'.format(h2d_mod[2][i],h2d_mod[2][i+1]))
#     ax.semilogy(h2d[1][1:] - 5, h2d[0][:, i] / 16, '-', color=colors(i / 10), label='{}-{} m'.format(h2d_mod[2][i], h2d_mod[2][i + 1]))
# plt.legend()
# plt.ylim(bottom=10)
# plt.xticks(np.arange(0, 365, 30))
# plt.ylabel('Area  (km^2)')
# plt.xlabel('SCD')
# fig.savefig(plot_folder + '/hist modis vs models SCD HY {} to {} {}_{}_{}_{}_{}_swe_thres{}_sca_thres{}.png'.format(hydro_years_to_take[0],
#                                                                                                                     hydro_years_to_take[-1],
#                                                                                                                     met_inp, which_model,
#                                                                                                                     catchment, output_dem, run_id,
#                                                                                                                     model_swe_sc_threshold,
#                                                                                                                     modis_sc_threshold), dpi=600)
#
# run_id = 'cl09_default'
# which_model = 'clark2009'  #
#
# [ann_ts_av_swe, ann_ts_av_sca_thres, ann_dt, ann_scd, ann_av_swe, ann_max_swe, ann_metadata] = pickle.load(open(
#     model_output_folder + '/summary_MODEL_{}_{}_{}_{}_{}_{}_{}_thres{}.pkl'.format(hydro_years_to_take[0], hydro_years_to_take[-1], met_inp, which_model,
#                                                                                    catchment, output_dem, run_id, model_swe_sc_threshold), 'rb'))
# ann_av_swe = [trim_data_to_mask(a, mask) for a in ann_av_swe]
# model_swe = np.nanmean(ann_av_swe, axis=0)
# mean_swe = np.full(np.arange(0, 3600 + 1, 200).shape, np.nan)
# area = np.full(np.arange(0, 3600 + 1, 200).shape, np.nan)
# for i, x in enumerate(np.arange(0, 3600 + 1, 200)):
#     mean_swe[i] = np.nanmean(model_swe[np.logical_and(nztm_dem > x, nztm_dem <= x + 200)])
#     area[i] = np.nansum(np.logical_and(nztm_dem > x, nztm_dem <= x + 200)) * .25 * .25
#
# run_id = 'dsc_default'  # #'cl09_default'  # #'cl09_default'# #  #  #
# which_model = 'dsc_snow'  #
#
# [ann_ts_av_swe, ann_ts_av_sca_thres, ann_dt, ann_scd, ann_av_swe, ann_max_swe, ann_metadata] = pickle.load(open(
#     model_output_folder + '/summary_MODEL_{}_{}_{}_{}_{}_{}_{}_thres{}.pkl'.format(hydro_years_to_take[0], hydro_years_to_take[-1], met_inp, which_model,
#                                                                                    catchment, output_dem, run_id, model_swe_sc_threshold), 'rb'))
#
# ann_av_swe = [trim_data_to_mask(a, mask) for a in ann_av_swe]
# model_swe_dsc = np.nanmean(ann_av_swe, axis=0)
# mean_swe_dsc = np.full(np.arange(0, 3600 + 1, 200).shape, np.nan)
# for i, x in enumerate(np.arange(0, 3600 + 1, 200)):
#     mean_swe_dsc[i] = np.nanmean(model_swe_dsc[np.logical_and(nztm_dem > x, nztm_dem <= x + 200)])
#     area[i] = np.nansum(np.logical_and(nztm_dem > x, nztm_dem <= x + 200)) * .25 * .25
#
# fig, ax = plt.subplots(figsize=(4, 4))
# plt.barh(np.arange(0, 3600 + 1, 200) + 100, mean_swe_dsc * area / 1e6, height=200, label='dsc_snow')
# plt.barh(np.arange(0, 3600 + 1, 200) + 100, mean_swe * area / 1e6, height=200, label='clark')
# plt.yticks(np.arange(0, 3600 + 1, 400))
# plt.ylim(0, 3600)
# plt.ylabel('Elevation (m)')
# plt.xlabel('Average snow storage (cubic km)')
# fig.savefig(plot_folder + '/hist_snow storage.png')
#
# plt.show()
# plt.close()
