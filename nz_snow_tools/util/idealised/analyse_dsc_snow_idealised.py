# script to run the Fortran version of dsc_snow model and iterate through different namelist options, file names etc

import subprocess
import os
import shutil
import f90nml
import netCDF4 as nc
import numpy as np
import matplotlib.pylab as plt

# compatibility for windows or linux
projects = 'P:/Projects'#'/mnt/shareddrive/Projects'  #

path_to_ddf_run = "/home/bs/dsc_snow/ddf/ddf_run"  # path to fortran executable
config_id_in = '4'  # string identifying suffix to the namelist configuration file used as default
config_id_out = '4'  # string identifying suffix to the namelist configuration file used by the model
path_to_namelist = projects + '/DSC-Snow/runs/input/clutha_nztm250m_erebus/ddf_config_{}.txt'.format(config_id_in)
years = [2016]

met_inps = ['bell_4000', 'flat_2000', 'north_facing', 'south_facing' ]


fig1 = plt.figure()

ax = fig1.subplots(2,2)
axs = ax.flatten()

for met_inp in met_inps:
    for year in years:
        for catchment in met_inps:

            # paths to files
            topo_file = nc.Dataset(projects + '/DSC-Snow/runs/idealised/{}_nztm250m_topo_no_ice.nc'.format(catchment))
            climate_file = nc.Dataset(projects + '/DSC-Snow/runs/idealised/met_inp_{}_nztm250m_{}.nc'.format(met_inp, year))
            output_file = nc.Dataset(projects + '/DSC-Snow/runs/idealised/snow_out_{}_dem{}_met{}_{}.nc'.format(year, catchment, met_inp, config_id_out))
            namelist_file = projects + '/DSC-Snow/runs/idealised/ddf_config_{}_dem{}_met{}_{}.txt'.format(year, catchment, met_inp, config_id_out)

            nc_dt = nc.num2date(output_file.variables['time'][:], output_file.variables['time'].units)
            abl = output_file.variables['ablation_total'][:]
            acc = output_file.variables['accumulation_total'][:]
            swe = output_file.variables['snow_water_equivalent'][:]
            water = output_file.variables['water_output_total'][:]
            #mb = output_file.variables['mass_balance'][:]


            plt.imshow(np.mean(swe,axis=0),origin=0)
            plt.colorbar()
            plt.contour(topo_file.variables['DEM'][:],range(0,4000,500),colors='k',linewidths=0.5)
            plt.savefig(projects + '/DSC-Snow/runs/idealised/p_mean_swe_{}_dem{}_met{}_{}.png'.format(year, catchment, met_inp, config_id_out),dpi=300)
            plt.close()