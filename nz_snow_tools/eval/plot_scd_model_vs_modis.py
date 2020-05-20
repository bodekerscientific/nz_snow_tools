import numpy as np
import pickle
import matplotlib.pylab as plt
import matplotlib.dates as mdates
import datetime as dt
from nz_snow_tools.util.utils import convert_datetime_julian_day
from nz_snow_tools.util.utils import setup_nztm_dem


plot_folder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/DSC Snow/SouthIsland_results/modis_comparison'
catchment = 'SI'  # string identifying catchment modelled
modis_dem = 'nztm250m'  # identifier for output dem
years_to_take = np.arange(2000, 2016 + 1)  # range(2016, 2016 + 1)  # [2013 + 1]  # range(2001, 2013 + 1)
modis_sc_threshold = 50  # value of fsca (in percent) that is counted as being snow covered
output_folder = r'C:\Users\conwayjp\OneDrive - NIWA\projects\DSC Snow\MODIS'
dem_folder = 'C:/Users/conwayjp/OneDrive - NIWA/Data/GIS_DATA/Topography/DEM_NZSOS/'

# load si dem and trim to size. (modis has smaller extent to west (1.085e6)
si_dem_file = dem_folder + 'si_dem_250m' + '.tif'
nztm_dem, x_centres, y_centres, lat_array, lon_array = setup_nztm_dem(si_dem_file, extent_w=1.08e6, extent_e=1.72e6, extent_n=5.52e6, extent_s=4.82e6,
                                                                      resolution=250)

nztm_dem = nztm_dem[:, 20:]
x_centres = x_centres[20:]
lat_array = lat_array[:, 20:]
lon_array = lon_array[:, 20:]

[ann_ts_av_sca_m, ann_ts_av_sca_thres_m, ann_dt_m, ann_scd_m] = pickle.load(open(
    output_folder + '/summary_MODIS_{}_{}_{}_{}_thres{}.pkl'.format(years_to_take[0], years_to_take[-1], catchment, modis_dem,
                                                                    modis_sc_threshold), 'rb'))

run_id = 'norton_5_topleft'
catchment = 'SouthIsland'
years_to_take = range(2000, 2016 + 1)  # [2013 + 1]  # range(2001, 2013 + 1)
model_swe_sc_threshold = 20  # threshold for treating a grid cell as snow covered (mm w.e)
model_output_folder = 'C:/Users/conwayjp/Documents/Temp'


[ann_ts_av_swe, ann_ts_av_sca_thres, ann_dt, ann_scd, ann_swe] = pickle.load(open(
    model_output_folder + '/summary_MODEL_{}_{}_{}_{}_thres{}.pkl'.format(years_to_take[0], years_to_take[-1], catchment, run_id,
                                                                          model_swe_sc_threshold), 'rb'))

# np.sum(~np.isnan(ann_scd[0]))/16 = 145378
ann_scd = np.asarray(ann_scd) / 100.  # convert to fraction from %

years_to_take = years_to_take[1:12]

modis_scd = np.nanmean(ann_scd_m[1:12], axis=0)
model_scd = np.nanmean(ann_scd[1:12], axis=0)

plot_scd_bias = model_scd - modis_scd

plt.rcParams.update({'font.size': 6})
plt.rcParams.update({'axes.titlesize' : 6})
fig1 = plt.figure(figsize=[4,4])

bin_edges = [-60, -30, -7, 7, 14, 30, 60]  # use small negative number to include 0 in the interpolation
CS1 = plt.contourf(x_centres, y_centres, plot_scd_bias, levels=bin_edges, cmap=plt.cm.RdBu,extend='both')
# CS1.cmap.set_bad('grey')
#CS1.cmap.set_over([0.47,0.72,0.77])
plt.gca().set_aspect('equal')
# plt.imshow(modis_scd, origin=0, interpolation='none', vmin=0, vmax=365, cmap='magma_r')
plt.xticks([])
plt.yticks([])
cbar = plt.colorbar()
cbar.set_label('Snow cover duration (days)', rotation=90)
plt.xticks(np.arange(12e5, 17e5, 2e5))
plt.yticks(np.arange(50e5, 55e5, 2e5))
plt.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
plt.ylabel('NZTM northing')
plt.xlabel('NZTM easting')
plt.title('SCD Bias: Model-MODIS {} to {}'.format(years_to_take[0], years_to_take[-1]))
plt.tight_layout()
plt.savefig(plot_folder + '/SCD bias model-modis {} to {}.png'.format(years_to_take[0], years_to_take[-1]), dpi=600)
plt.clf()