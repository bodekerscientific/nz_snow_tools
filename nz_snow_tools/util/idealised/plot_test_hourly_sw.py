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


for git_commit in ['7c43f67', '503d26c', 'edec30c', 'edec30c_timezone_neg12']:
    topo_file = nc.Dataset(r"P:\Projects\DSC-Snow\runs\idealised\bell_4000_nztm250m_topo_no_ice.nc")
    met_file = nc.Dataset(r"P:\Projects\DSC-Snow\runs\idealised\met_inp_bell_4000_nztm250m_2016_origintopleft.nc")
    output_file = nc.Dataset(
        r"P:\Projects\DSC-Snow\runs\idealised\checking_SW\snow_out_2016_dembell_4000_metbell_4000_5_origintopleft_test_{}.nc".format(git_commit))
    nc_dt = nc.num2date(output_file.variables['time'][:], output_file.variables['time'].units)
    met_dt = nc.num2date(met_file.variables['time'][:], met_file.variables['time'].units)
    # abl = output_file.variables['ablation_total'][:]
    # acc = output_file.variables['accumulation_total'][:]
    # swe = output_file.variables['snow_water_equivalent'][:]
    # water = output_file.variables['water_output_total'][:]
    sw_net = output_file.variables['net_shortwave_radiation_at_surface'][:]
    albedo = output_file.variables['albedo'][:]
    sw_inp = output_file.variables['incoming_solar_radiation_to_surface_from_input'][:]
    cloud = output_file.variables['cloud'][:]

    sw_in = sw_net / (1 - albedo)

    for i in range(24):
        plt.imshow(sw_net[i])
        plt.colorbar()
        plt.contour(topo_file.variables['DEM'][:],range(0,4000,500),colors='k',linewidths=0.5)
        plt.savefig(projects + '/DSC-Snow/runs/idealised/checking_SW/p_swnet_{}_{}.png'.format(git_commit,i),dpi=300)
        plt.close()

    # nc_hours = [d.hour for d in nc_dt]
    plt.plot(nc_dt[:24], sw_inp[:24, 49, 49], 'o-', label='sw_input')
    plt.plot(nc_dt[:24], sw_in[:24, 49, 49], 'o-', label='sw_mod [49,49]')
    plt.plot(nc_dt[:24], sw_in[:24, 49, 50], 'o-', label='sw_mod [49,50]')
    plt.plot(nc_dt[:24], cloud[:24, 49, 49] * 100, 'o-', label='cloudiness (%)')
    fig = plt.gcf()
    fig.autofmt_xdate()
    plt.legend()
    plt.ylabel('Flux density (W m^-2) or cloudiness (%)')
    plt.xlabel('Time (NZST)')
    plt.savefig(projects + '/DSC-Snow/runs/idealised/checking_SW/time series sw {}.png'.format(git_commit), dpi=300)
    plt.close()

    plt.plot(nc_dt[:24], sw_in[:24, 39, 49], 'o-', label='north facing')
    plt.plot(nc_dt[:24], sw_in[:24, 49, 49], 'o-', label='top')
    plt.plot(nc_dt[:24], sw_in[:24, 59, 49], 'o-', label='south facing')
    plt.plot(nc_dt[:24], sw_in[:24, 49, 59], 'o-', label='east facing')
    plt.plot(nc_dt[:24], sw_in[:24, 49, 39], 'o-', label='west facing')
    plt.plot(nc_dt[:24], sw_inp[:24, 49, 49], 'o-k', label='input')
    fig = plt.gcf()
    fig.autofmt_xdate()
    plt.legend()
    plt.ylabel('Flux density (W m^-2)')
    plt.xlabel('Time (NZST)')
    plt.savefig(projects + '/DSC-Snow/runs/idealised/checking_SW/time series sw slopes {}.png'.format(git_commit), dpi=300)
    plt.close()
