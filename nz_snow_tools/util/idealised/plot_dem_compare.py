# script to run the Fortran version of dsc_snow model and iterate through different namelist options, file names etc

import subprocess
import os
import shutil
import f90nml
import netCDF4 as nc
import numpy as np
import matplotlib.pylab as plt

# compatibility for windows or linux
projects = 'P:/Projects'  # '/mnt/shareddrive/Projects'  #
# git_commit = '7c43f67' # 503d26c (post sw fixes) or 7c43f67 (pre NC cache) or 'edec30c' - lat/lon specified for solar rad
# nztm_dem, x_centres, y_centres, lat_array, lon_array = setup_nztm_dem(dem_file, origin=origin)
# lat_array[lat_array.shape[0]/2,lat_array.shape[1]/2]
# -45.31551716333078
# lon_array[lon_array.shape[0]/2,lon_array.shape[1]/2]
# 169.1741339885213
dem_new = nc.Dataset(r"P:\Projects\DSC-Snow\dsc_snow\brewster_topo.nc")
dem_old = nc.Dataset(r"D:\Snow project\from Ruschle\brewster\brewster_topo.nc")

grd_names=['DEM','ice','catchment','viewfield','debris','slope','aspect']

for gname in grd_names:

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(dem_old.variables[gname][:])
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('old' + gname)
    plt.subplot(1,2,2)
    plt.imshow(dem_new.variables[gname][:])
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('new' + gname)
    plt.tight_layout()
    plt.savefig(projects + '/DSC-Snow/runs/idealised/checking_dem_{}.png'.format(gname), dpi=300)
    plt.close()
