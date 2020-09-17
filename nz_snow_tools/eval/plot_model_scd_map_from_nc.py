"""
code to plot maps of snow covered area for individual years from summary lists generated by catchment_evalutation.py
"""

from __future__ import division

import numpy as np
import pickle
import matplotlib.pylab as plt
from nz_snow_tools.util.utils import setup_nztm_dem
from matplotlib.colors import BoundaryNorm
import netCDF4 as nc
import copy

run_id = 'dsc_default'
catchment = 'SI'
output_dem = 'si_dem_250m'  # identifier for output dem
years_to_take = [2016,2018,2019]  # [2013 + 1]  # range(2001, 2013 + 1)
model_swe_sc_threshold = 20  # threshold for treating a grid cell as snow covered (mm w.e)
model_output_folder = '/nesi/nobackup/niwa00004/jonoconway/snow_sims_nz/dsc_snow'
plot_folder = '/nesi/nobackup/niwa00004/jonoconway/snow_sims_nz/dsc_snow'
dem_folder = '/nesi/project/niwa00004/jonoconway'  # dem used for output #'C:/Users/conwayjp/OneDrive - NIWA/Data/GIS_DATA/Topography/DEM_NZSOS'#

print('loading dem')
# load si dem and trim to size. (modis has smaller extent to west (1.085e6)
si_dem_file = dem_folder + '/si_dem_250m' + '.tif'
nztm_dem, x_centres, y_centres, lat_array, lon_array = setup_nztm_dem(si_dem_file, extent_w=1.08e6, extent_e=1.72e6, extent_n=5.52e6, extent_s=4.82e6,
                                                                      resolution=250)
nztm_dem = nztm_dem[:, 20:]
x_centres = x_centres[20:]
lat_array = lat_array[:, 20:]
lon_array = lon_array[:, 20:]

plt.rcParams.update({'font.size': 6})
plt.rcParams.update({'axes.titlesize' : 6})
fig1 = plt.figure(figsize=[4,4])

bin_edges = [-0.001, 30, 60, 90, 120, 180, 270, 360]  # use small negative number to include 0 in the interpolation

for i, year_to_take in enumerate(years_to_take):
    print('loading data for year {}'.format(year_to_take))
    nc_file = nc.Dataset(model_output_folder + '/snow_out_SI_si_dem_250m_{}_{}.nc'.format(run_id, year_to_take), 'r')
    model_scd = np.sum(nc_file.variables['swe'][:] > model_swe_sc_threshold,axis=0).astype(np.float)
    model_scd = model_scd[:, 20:]
    model_scd[nztm_dem==0] = np.nan
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
    plt.xticks(np.arange(12e5, 17e5, 2e5))
    plt.yticks(np.arange(50e5, 55e5, 2e5))
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    plt.ylabel('NZTM northing')
    plt.xlabel('NZTM easting')
    plt.title('Snow cover duration {}'.format(year_to_take))
    plt.tight_layout()

    plt.savefig(plot_folder + '/SCD model {} thres{} {}.png'.format(year_to_take, model_swe_sc_threshold, run_id), dpi=300)
    plt.show()
    plt.clf()
