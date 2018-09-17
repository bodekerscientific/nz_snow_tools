# read DEM geotiffs and create netCDF required for snow model

# jono conway
#
# from osgeo import gdal, ogr
# from gdalconst import *
# import pyproj
# import argparse

import netCDF4 as netCDF4
import numpy as np
import matplotlib.pylab as plt
from nz_snow_tools.util.utils import setup_nztm_dem

# create dem surface
origin = 'topleft'  # or 'bottomleft'
_, eastings, northings, lats, lons = setup_nztm_dem(dem_file=None, extent_w=1.235e6, extent_e=1.26e6, extent_n=5.05e6, extent_s=5.025e6, resolution=250,
                                                    origin=origin)

for catchment in ['flat_2000', 'north_facing', 'south_facing', 'bell_4000', 'bell_2000', 'real']:
    if catchment == 'flat_2000':
        elev = np.ones((100, 100)) * 2000
    elif catchment == 'north_facing':
        elev = np.linspace(4000, 0, 100)[:, np.newaxis] * np.ones((100, 100))
        if origin == 'topleft':
            np.flipud(elev)
    elif catchment == 'south_facing':
        elev = np.linspace(0, 4000, 100)[:, np.newaxis] * np.ones((100, 100))
        if origin == 'topleft':
            np.flipud(elev)
    elif catchment == 'bell_4000':
        x, y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
        d = np.sqrt(x * x + y * y)
        sigma, mu = 0.25, 0.0
        g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
        elev = g * 4000
    elif catchment == 'bell_2000':
        x, y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
        d = np.sqrt(x * x + y * y)
        sigma, mu = 0.25, 0.0
        g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
        elev = g * 2000
    elif catchment == 'real':
        input_dem = 'nztm250m'
        nc_file = netCDF4.Dataset('P:/Projects/DSC-Snow/runs/idealised/met_inp_{}_{}_{}_origin{}.nc'.format(catchment, input_dem, 2016, origin))
        elev = nc_file.variables['elevation'][:]
    # assume a square, axis-oriented grid
    gx, gy = np.gradient(elev, 250.0)

    # write out the topographic info to netcdf file
    output_dem = 'nztm250m'  # identifier for output dem
    data_id = '{}_{}'.format(catchment, output_dem)  # name to identify the output data
    out_file = 'P:/Projects/DSC-Snow/runs/idealised/{}_topo_no_ice_origin{}.nc'.format(data_id,origin)

    file_out = netCDF4.Dataset(out_file, 'w')
    # nc_common_attr(file_out, JONO, title='topographic fields for snow model',source='prep_dem2nc.py')
    # setattr(file_out,'versionNumber',1)

    # start with georeferencing
    file_out.createDimension('rows', len(northings))
    file_out.createDimension('columns', len(eastings))

    file_out.createDimension('latitude', len(northings))
    file_out.createDimension('longitude', len(eastings))

    # latitude
    lat_out = file_out.createVariable('latitude', 'f', ('rows', 'columns'))
    lat_out[:] = lats
    setattr(lat_out, 'units', 'degrees')
    # longitude
    lon_out = file_out.createVariable('longitude', 'f', ('rows', 'columns'))
    lon_out[:] = lons
    setattr(lon_out, 'units', 'degrees')
    # easting
    east_out = file_out.createVariable('easting', 'f', ('columns',))
    east_out[:] = eastings
    setattr(east_out, 'units', 'm NZTM')
    # northing
    north_out = file_out.createVariable('northing', 'f', ('rows',))
    north_out[:] = northings
    setattr(north_out, 'units', 'm NZTM')

    # and now the grids themselves
    grd_names = ['DEM', 'ice', 'catchment', 'viewfield', 'debris', 'slope', 'aspect']

    for gname in grd_names:
        if gname == 'debris' or gname == 'ice':
            # not required for enhanced DDM
            data = np.zeros(elev.shape, dtype='float32')
            units = 'boolean'
        if gname == 'slope':
            data = np.arctan(np.sqrt(gx * gx + gy * gy))
            units = 'radians'
        if gname == 'aspect':
            if origin == 'topleft':
                data = - np.pi / 2. - np.arctan2(-gx, gy)
            elif origin == 'bottomleft':
                data = - np.pi / 2. - np.arctan2(gx, gy)
            data = np.where(data < -np.pi, data + 2 * np.pi, data)
            units = 'radians'
        if gname == 'catchment':
            data = np.ones(elev.shape, dtype='float32')
            units = 'boolean'
        if gname == 'viewfield':
            data = np.ones(elev.shape, dtype='float32')  # set all to ones TODO: use actual sky view grid
            units = 'fraction 0-1'
        if gname == 'DEM':
            data = elev
            units = 'metres asl'
        raster_out = file_out.createVariable(gname, 'f', ('rows', 'columns'))
        raster_out[:] = data
        setattr(raster_out, 'units', units)

    file_out.close()

x, y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
d = np.sqrt(x * x + y * y)
sigma, mu = 0.25, 0.0
g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
elev = g * 4000
gx, gy = np.gradient(elev, 250.0)

data = np.arctan(np.sqrt(gx * gx + gy * gy))

plt.figure()
plt.plot(np.rad2deg(data[50]))
plt.ylabel('slope (degrees)')

plt.figure()
plt.plot(elev[50])
plt.xlabel('grid point')
plt.ylabel('elevation (metres)')


plt.figure()
if origin == 'topleft':
    data = - np.pi / 2. - np.arctan2(-gx, gy)
    data = np.where(data < -np.pi, data + 2 * np.pi, data)
    plt.imshow(np.rad2deg(data))
elif origin == 'bottomleft':
    data = - np.pi / 2. - np.arctan2(gx, gy)
    data = np.where(data < -np.pi, data + 2 * np.pi, data)
    plt.imshow(np.rad2deg(data),origin=0)
plt.imshow(np.rad2deg(data))
plt.colorbar()
plt.xlabel('easting')
plt.ylabel('northing')
plt.title('aspect')
plt.show()
