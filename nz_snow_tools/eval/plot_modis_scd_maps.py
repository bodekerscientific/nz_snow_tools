"""
code to plot maps of snow covered area for individual years from summary lists generated by catchment_evalutation.py
"""

from __future__ import division

import numpy as np
import pickle
import matplotlib.pylab as plt
from nz_snow_tools.util.utils import setup_nztm_dem
from matplotlib.colors import BoundaryNorm

average_scd = True # boolean specifying if all years are to be averaged together - now plots difference between
catchment = 'SI'
output_dem = 'nztm250m'  # identifier for output dem
years_to_take = range(2000, 2016 + 1)  # [2013 + 1]  # range(2001, 2013 + 1)
modis_sc_threshold = 50  # value of fsca (in percent) that is counted as being snow covered
modis_output_folder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/DSC Snow/MODIS'
plot_folder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/DSC Snow/MODIS'
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
    modis_output_folder + '/summary_MODIS_{}_{}_{}_{}_thres{}.pkl'.format(years_to_take[0], years_to_take[-1], catchment, output_dem,
                                                                    modis_sc_threshold), 'rb'))


years_to_take = years_to_take[1:12] # remove 2012 onwards as errornous runs


plt.rcParams.update({'font.size': 6})
plt.rcParams.update({'axes.titlesize' : 6})
fig1 = plt.figure(figsize=[4,4])

if average_scd ==True:
    bin_edges = [0, 30, 60, 90, 120, 180, 270, 360]  # use small negative number to include 0 in the interpolation
    modis_scd = np.nanmean(ann_scd_m[1:12], axis=0)
    CS1 = plt.contourf(x_centres, y_centres, modis_scd, levels=bin_edges, cmap=plt.cm.magma_r,extend='max')
    # CS1.cmap.set_bad('grey')
    CS1.cmap.set_over([0.47,0.72,0.77])
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
    plt.title('MODIS mean SCD {} to {}'.format(years_to_take[0], years_to_take[-1]))
    plt.tight_layout()
    plt.savefig(plot_folder + '/SCD modis {} to {}.png'.format(years_to_take[0], years_to_take[-1]), dpi=600)
    plt.clf()

    modis_scd = np.nanstd(ann_scd_m[1:12], axis=0)
    bin_edges = [0, 7, 14, 21, 28, 35]  # use small negative number to include 0 in the interpolation
    CS1 = plt.contourf(x_centres, y_centres, modis_scd, levels=bin_edges, cmap=plt.cm.magma_r,extend='max')
    # CS1.cmap.set_bad('grey')
    CS1.cmap.set_over([0.47,0.72,0.77])
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
    plt.title('MODIS interannual std dev {} to {}'.format(years_to_take[0], years_to_take[-1]))
    plt.tight_layout()
    plt.savefig(plot_folder + '/SCD stdev modis {} to {}.png'.format(years_to_take[0], years_to_take[-1]), dpi=600)

else:
    for i, year_to_take in enumerate(years_to_take):
        bin_edges = [-0.001, 30, 60, 90, 120, 180, 270, 360]  # use small negative number to include 0 in the interpolation
        modis_scd = ann_scd_m[i]
        CS1 = plt.contourf(x_centres, y_centres, modis_scd, levels=bin_edges, cmap=plt.cm.magma_r, extend='max')
        # CS1.cmap.set_bad('grey')
        CS1.cmap.set_over([0.47, 0.72, 0.77])
        plt.gca().set_aspect('equal')
        # plt.imshow(modis_scd, origin=0, interpolation='none', vmin=0, vmax=365, cmap='magma_r')
        plt.xticks([])
        plt.yticks([])
        cbar = plt.colorbar()
        cbar.set_label('Snow cover duration (days)', rotation=90)
        plt.xticks(np.arange(12e5, 17e5, 2e5))
        plt.yticks(np.arange(50e5, 55e5, 2e5))
        plt.ylabel('NZTM northing')
        plt.xlabel('NZTM easting')
        plt.title('Snow cover duration {}'.format(year_to_take))
        plt.tight_layout()

        plt.savefig(plot_folder + '/SCD modis {}_{}_{}_fsca{}_update.png'.format(year_to_take, catchment, output_dem, modis_sc_threshold), dpi=300)
        plt.clf()


# plt.figure(figsize=[10,10])
#     cmap = plt.get_cmap('magma_r')
#     # cmap.set_bad('none')
#     cmap.set_over([0.47,0.72,0.77])
#     norm = BoundaryNorm(bin_edges[:], ncolors=cmap.N,clip=True)
#     im = plt.pcolormesh(x_centres, y_centres, modis_scd, cmap=cmap, norm=norm)
#     plt.gca().set_aspect('equal')
#     # plt.imshow(modis_scd, origin=0, interpolation='none', vmin=0, vmax=365, cmap='magma_r')
#     plt.xticks([])
#     plt.yticks([])
#     plt.colorbar()
#     plt.xticks(np.arange(12e5, 17e5, 2e5))
#     plt.yticks(np.arange(50e5, 55e5, 2e5))
#     plt.ylabel('NZTM northing')
#     plt.xlabel('NZTM easting')
#     plt.title('Mean snow cover duration (days) {} to {}'.format(years_to_take[0], years_to_take[-1]))
#     plt.tight_layout()
#     plt.savefig(plot_folder + '/SCD modis pcolor{} to {}.png'.format(years_to_take[0], years_to_take[-1]), dpi=600)