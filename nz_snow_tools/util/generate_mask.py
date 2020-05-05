"""
code to generate catchment masks for modelling
updated to load for all shapefiles in a folder
"""
from __future__ import division

import numpy as np
from nz_snow_tools.util.utils import create_mask_from_shpfile, setup_nztm_dem
# from nz_snow_tools.met.interp_met_data_hourly_vcsn_data import
import os

#
# # os.environ['PROJ_LIB'] = r'C:\miniconda\envs\nz_snow27\Library\share'

# catchments = ['Wilkin']
# catchments = ['Clutha','Wilkin','Wanaka northern inflows','Upper Dart','Rees', 'Shotover', 'Teviot','Taieri','Upper Matukituki','Roaring Meg','Pomahaka','Motutapu',\
#               'Moonlight Creek','Matukituki', 'Manuherikia','Luggate Creek', 'Lochy','Lindis',\
#               'Kawarau','Greenstone','Hawea','Fraser','Clutha above Clyde Dam','Cardrona','Arrow' ,'Bannockburn Creek', 'Nevis'] # string identifying the catchment to run. must match the naming of the catchment shapefile

dem = 'modis_nz_dem_250m' # identifier for modis grid - extent specified below
mask_folder = '/nesi/nobackup/niwa00026/Observation/Snow_RemoteSensing/catchment_masks'  # location of numpy catchment mask. must be writeable if mask_created == False
catchment_shp_folder = '/nesi/nobackup/niwa00026/Observation/Snow_RemoteSensing/Catchments'  # shapefile containing polyline or polygon of catchment in WGS84

# read names of shapefiles
contents = os.listdir(catchment_shp_folder)
shps = [s.split('.')[0] for s in contents if ".shp" in s and ".xml" not in s]

# calculate model grid etc:
# output DEM

if dem == 'clutha_dem_250m':
    nztm_dem, x_centres, y_centres, lat_array, lon_array = setup_nztm_dem(None, extent_w=1.2e6, extent_e=1.4e6, extent_n=5.13e6, extent_s=4.82e6,
                                                                          resolution=250, origin='bottomleft')

if dem == 'si_dem_250m':
    nztm_dem, x_centres, y_centres, lat_array, lon_array = setup_nztm_dem(None, extent_w=1.08e6, extent_e=1.72e6, extent_n=5.52e6, extent_s=4.82e6,
                                                                          resolution=250, origin='bottomleft')
if dem == 'modis_si_dem_250m':
    nztm_dem, x_centres, y_centres, lat_array, lon_array = setup_nztm_dem(None, extent_w=1.085e6, extent_e=1.72e6, extent_n=5.52e6, extent_s=4.82e6,
                                                                          resolution=250, origin='bottomleft')

if dem == 'modis_nz_dem_250m':
    nztm_dem, x_centres, y_centres, lat_array, lon_array = setup_nztm_dem(None, extent_w=1.085e6, extent_e=2.10e6, extent_n=6.20e6, extent_s=4.70e6,
                                                                          resolution=250, origin='bottomleft')

for catchment in shps:
    if '.shp' in catchment:
        mask_shpfile = catchment_shp_folder + '/{}'.format(catchment)
    else:
        mask_shpfile = catchment_shp_folder + '/{}.shp'.format(catchment)
    if dem == 'modis_nz_dem_250m':
        mask = create_mask_from_shpfile(y_centres, x_centres, mask_shpfile)
    else:
        mask = create_mask_from_shpfile(lat_array, lon_array, mask_shpfile)
    np.save(mask_folder + '/{}_{}.npy'.format(catchment, dem), mask)
