
# read DEM geotiffs and create netCDF required for snow model

# jono conway
#

import netCDF4 as netCDF4
import numpy as np
# from osgeo import gdal, ogr
# from gdalconst import *
# import pyproj
# import argparse
from jc_scripts.dsc_snow.met_data.A2_interpolate_hourly_vcsn_data import setup_clutha_dem_250m,create_mask_from_shpfile,trim_lat_lon_bounds
from bs.core import env
from bs.core.util import nc_common_attr, JONO

mask_dem = True  # boolean to set whether or not to mask the output dem
catchment = 'Clutha'
MASK_SHPFILE = env.data('/GIS_DATA/Hydrology/Catchments/{} full WGS84.shp'.format(catchment))
mask_created = True  # boolean to set whether or not the mask has already been created
input_dem = 'clutha_dem_250m'
output_dem = 'nztm250m'  # identifier for output dem
data_id = '{}_{}'.format(catchment, output_dem)  # name to identify the output data
out_file = 'P:/Projects/DSC-Snow/runs/input_DEM/{}_topo_no_ice.nc'.format(data_id)

nztm_dem, x_centres, y_centres, lat_array, lon_array = setup_clutha_dem_250m()

if mask_dem == True:
    # # Get the masks for the individual regions of interest
    if mask_created == True:  # load precalculated mask
        mask = np.load('T:/DSC-Snow/Masks/{}_{}.npy'.format(catchment, input_dem))
        # mask = np.load(env.data('/GIS_DATA/Hydrology/Catchments/Masks/{}_{}.npy'.format(catchment, output_dem)))
    else:  # create mask and save to npy file
        # masks = get_masks() #TODO set up for multiple masks
        mask = create_mask_from_shpfile(lat_array, lon_array, MASK_SHPFILE)
        np.save('T:/DSC-Snow/Masks/{}_{}.npy'.format(catchment, input_dem), mask)
        # np.save(env.data('/GIS_DATA/Hydrology/Catchments/Masks/{}_{}.npy'.format(catchment, output_dem)), mask)
    masks = [[data_id, mask]]
    # Trim down the number of latitudes requested so it all stays in memory
    lats, lons, elev, northings, eastings = trim_lat_lon_bounds(mask, lat_array, lon_array, nztm_dem, y_centres,
                                                                x_centres)
    _, _, trimmed_mask, _, _ = trim_lat_lon_bounds(mask, lat_array, lon_array, mask, y_centres,
                                                                x_centres)
else:
    masks = [[data_id, None]]
    lats = lat_array
    lons = lon_array
    elev = nztm_dem
    northings = y_centres
    eastings = x_centres

# assume a square, axis-oriented grid
gx, gy = np.gradient(elev,250.0)

# write out the topographic info to netcdf file

file_out=netCDF4.Dataset(out_file,'w')
nc_common_attr(file_out, JONO, title='topographic fields for snow model',source='prep_dem2nc.py')
#setattr(file_out,'versionNumber',1)

# start with georeferencing
file_out.createDimension('rows',len(northings))
file_out.createDimension('columns',len(eastings))

file_out.createDimension('latitude',len(northings))
file_out.createDimension('longitude',len(eastings))

# latitude
lat_out=file_out.createVariable('latitude','f',('rows','columns'))
lat_out[:]=lats
setattr(lat_out,'units','degrees')
# longitude
lon_out=file_out.createVariable('longitude','f',('rows','columns'))
lon_out[:]=lons
setattr(lon_out,'units','degrees')
# easting
east_out=file_out.createVariable('easting','f',('columns',))
east_out[:]=eastings
setattr(east_out,'units','m NZTM')
# northing
north_out=file_out.createVariable('northing','f',('rows',))
north_out[:]=northings
setattr(north_out,'units','m NZTM')


# and now the grids themselves
grd_names=['DEM','ice','catchment','viewfield','debris','slope','aspect']

for gname in grd_names:
    if gname=='debris' or gname == 'ice':
        # not required for enhanced DDM
        data = np.zeros(elev.shape, dtype='float32')
        units = 'boolean'
    if gname=='slope':
        data = np.arctan(np.sqrt(gx*gx + gy*gy))
        units = 'radians'
    if gname=='aspect':
        data = - np.pi/2. - np.arctan2(gx, gy)
        data = np.where(data < -np.pi, data + 2 * np.pi , data)
        units = 'radians'
    if gname == 'catchment':
        data = trimmed_mask
        units = 'boolean'
    if gname =='viewfield':
        data = np.ones(elev.shape, dtype='float32') # set all to ones TODO: use actual sky view grid
        units = 'fraction 0-1'
    if gname == 'DEM':
        data = elev
        units = 'metres asl'
    raster_out=file_out.createVariable(gname, 'f', ('rows', 'columns'))
    raster_out[:]=data
    setattr(raster_out,'units',units)

file_out.close()

