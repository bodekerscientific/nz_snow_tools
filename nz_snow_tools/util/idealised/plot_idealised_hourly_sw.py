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
config_id_in = '5'  # string identifying suffix to the namelist configuration file used as default
config_id_out = '5'  # string identifying suffix to the namelist configuration file used by the model
path_to_namelist = projects + '/DSC-Snow/runs/input/clutha_nztm250m_erebus/ddf_config_{}.txt'.format(config_id_in)
years = [2016]

catchment = 'bell_4000'
met_inp = 'flat_2000'
topo_file = nc.Dataset(r"P:\Projects\DSC-Snow\runs\idealised\bell_4000_nztm250m_topo_no_ice.nc")
output_file = nc.Dataset(r"P:\Projects\DSC-Snow\runs\idealised\snow_out_2016_dembell_4000_metflat_2000_5_hourly.nc")
nc_dt = nc.num2date(output_file.variables['time'][:], output_file.variables['time'].units)
abl = output_file.variables['ablation_total'][:]
acc = output_file.variables['accumulation_total'][:]
swe = output_file.variables['snow_water_equivalent'][:]
water = output_file.variables['water_output_total'][:]
sw_net = output_file.variables['net_shortwave_radiation_at_surface'][:]
#mb = output_file.variables['mass_balance'][:]

for i in range(24):
    plt.imshow(sw_net[i])
    plt.colorbar()
    plt.contour(topo_file.variables['DEM'][:],range(0,4000,500),colors='k',linewidths=0.5)
    plt.savefig(projects + '/DSC-Snow/runs/idealised/p_swnet{}_{}_dem{}_met{}_{}.png'.format(i,2016, catchment, met_inp, config_id_out),dpi=300)
    plt.close()