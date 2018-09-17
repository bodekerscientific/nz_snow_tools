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

origin = 'topleft'  # origin of new DEM surface
path_to_ddf_run = "/home/bs/dsc_snow/ddf/ddf_run"  # path to fortran executable
config_id_in = '5'  # string identifying suffix to the namelist configuration file used as default
config_id_out = '5'  # string identifying suffix to the namelist configuration file used by the model
path_to_namelist = projects + '/DSC-Snow/runs/input/clutha_nztm250m_erebus/ddf_config_{}.txt'.format(config_id_in)
years = [2016]

met_inps = ['flat_2000', 'north_facing', 'south_facing', 'bell_4000', 'bell_2000', 'real']

for met_inp in met_inps:
    for year in years:
        for catchment in met_inps:
            # paths to files
            topo_file = nc.Dataset(projects + '/DSC-Snow/runs/idealised/{}_nztm250m_topo_no_ice_origin{}.nc'.format(catchment, origin))
            climate_file = nc.Dataset(projects + '/DSC-Snow/runs/idealised/met_inp_{}_nztm250m_{}_origin{}.nc'.format(met_inp, year, origin))
            output_file = nc.Dataset(
                projects + '/DSC-Snow/runs/idealised/snow_out_{}_dem{}_met{}_{}_origin{}.nc'.format(year, catchment, met_inp, config_id_out, origin))
            namelist_file = projects + '/DSC-Snow/runs/idealised/ddf_config_{}_dem{}_met{}_{}_origin{}.txt'.format(year, catchment, met_inp, config_id_out,
                                                                                                                   origin)

            nc_dt = nc.num2date(output_file.variables['time'][:], output_file.variables['time'].units)
            # abl = output_file.variables['ablation_total'][:]
            # acc = output_file.variables['accumulation_total'][:]
            swe = output_file.variables['snow_water_equivalent'][:]
            # water = output_file.variables['water_output_total'][:]
            # sw_net = output_file.variables['net_shortwave_radiation_at_surface'][:]

            plt.imshow(np.mean(swe, axis=0))  # ,origin=0
            plt.colorbar()
            plt.contour(topo_file.variables['DEM'][:], range(0, 4000, 500), colors='k', linewidths=0.5)
            plt.title('mean SWE (m w.e.) with 500m contours')
            plt.savefig(r'P:\Projects\DSC-Snow\runs\idealised\p_meanSWE_{}_dem{}_met{}_{}_origin{}.png'.format(year, catchment, met_inp, config_id_out, origin))
            plt.close()
