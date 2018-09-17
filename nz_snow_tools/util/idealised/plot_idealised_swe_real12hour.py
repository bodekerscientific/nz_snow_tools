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

catchment = 'real'  # 'bell_4000'
met_inp = 'real'  # 'flat_2000'

topo_file = nc.Dataset(r"P:\Projects\DSC-Snow\runs\idealised\{}_nztm250m_topo_no_ice_origin{}.nc".format(catchment, origin))
climate_file = nc.Dataset(r"P:\Projects\DSC-Snow\runs\idealised\met_inp_{}_nztm250m_2016_origintopleft.nc".format(met_inp))
output_file = nc.Dataset(r"P:\Projects\DSC-Snow\runs\idealised\snow_out_2016_dem{}_met{}_5_12hourly_origintopleft.nc".format(catchment, met_inp))
nc_dt = nc.num2date(output_file.variables['time'][:], output_file.variables['time'].units)
abl = output_file.variables['ablation_total'][:]
acc = output_file.variables['accumulation_total'][:]
swe = output_file.variables['snow_water_equivalent'][:]
water = output_file.variables['water_output_total'][:]
sw_net = output_file.variables['net_shortwave_radiation_at_surface'][:]

plt.imshow(topo_file.variables["DEM"][:])

# topo_file_clutha = nc.Dataset(r"P:\Projects\DSC-Snow\runs\input_DEM\Clutha_nztm250m_topo_no_ice_origintopleft.nc")
# plt.figure()
# plt.imshow(topo_file_clutha.variables["DEM"][:])

plt.figure()

plt.imshow(np.mean(swe, axis=0))  # ,origin=0
plt.colorbar()
plt.contour(topo_file.variables['DEM'][:], range(0, 4000, 500), colors='k', linewidths=0.5)
plt.show()

# check that chosen point is at same elevation with north and south facing aspect
aspect = topo_file.variables["aspect"][:]
print(aspect[6, 64])
print(aspect[13, 63])
elev = topo_file.variables["DEM"][:]
print(elev[6, 64])
print(elev[13, 63])

# plot north vs south facing slope to check sanity of results
plt.plot(nc_dt, swe[:, 6, 64], label='north facing')
plt.plot(nc_dt, swe[:, 13, 63], label='south facing')
plt.legend()
plt.ylabel('SWE (m w.e.)')
plt.title('SWE during 2016 @ 1500m')
plt.savefig(r'P:\Projects\DSC-Snow\runs\idealised\north_vs_south_realSWE.png')

# check water output and precipitation input match
print(np.sum(climate_file.variables['precipitation_amount'][:, 6, 64]))  # 2520.1838567974073
print(np.sum(climate_file.variables['precipitation_amount'][:, 13, 63]))  # 2450.2297459604074
print(water[-1, 6, 64] * 1000.)  # 2520.179033279419
print(water[-1, 13, 63] * 1000.)  # 2450.230121612549

# nc_dt[487] == datetime.datetime(2016, 9, 1, 0, 0)

plt.imshow(swe[487],cmap=plt.cm.Blues_r)
plt.colorbar()
plt.contour(topo_file.variables['DEM'][:], range(0, 4000, 100), colors='k', linewidths=0.5)
plt.title('SWE on 1st September ')
plt.savefig(r'P:\Projects\DSC-Snow\runs\idealised\domain swe 1st sept.png')

