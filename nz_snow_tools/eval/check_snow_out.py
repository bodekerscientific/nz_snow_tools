"""
code to check consistency of snow model output from dsc_snow
"""

from __future__ import division

import netCDF4 as nc
import matplotlib.pylab as plt
import datetime as dt
import numpy as np

# infile = r"P:\Projects\DSC-Snow\runs\output\nevis_brew_hy12_new_units_test\nevis_brew_hy12_new_units_test18.nc"
# nc_file18 = nc.Dataset(infile)
# abl_18 = nc_file18.variables['ablation_total'][:]
# acc_18 = nc_file18.variables['accumulation_total'][:]
# swe_18 = nc_file18.variables['snow_water_equivalent'][:]
# water_18 = nc_file18.variables['water_output_total'][:]
infile = r"T:\DSC-Snow\runs\output\clutha_nztm250m_erebus\Clutha_nztm250m_2000_jobst_ucc_4.nc"
#infile = r"P:\Projects\DSC-Snow\runs\output\clutha_2D_test_erebus\Clutha_2D_2016_jobst_ucc.nc"
# infile = r"P:\Projects\DSC-Snow\runs\output\nevis_brew_hy12_new_units_test\nevis_brew_hy12_new_units_test17.nc"
nc_file17 = nc.Dataset(infile)
nc_dt = nc.num2date(nc_file17.variables['time'][:], nc_file17.variables['time'].units)
abl_17 = nc_file17.variables['ablation_total'][:]
acc_17 = nc_file17.variables['accumulation_total'][:]
swe_17 = nc_file17.variables['snow_water_equivalent'][:]
water_17 = nc_file17.variables['water_output_total'][:]
mb_17 = nc_file17.variables['mass_balance'][:]


# apply catchment mask
infile_mask = r"P:\Projects\DSC-Snow\runs\input_DEM\Clutha_nztm250m_topo_no_ice.nc"
nc_file_mask = nc.Dataset(infile_mask)
mask = nc_file_mask.variables['catchment'][:]

abl_17[:, mask == 0] = np.nan
acc_17[:, mask == 0] = np.nan
swe_17[:, mask == 0] = np.nan
water_17[:, mask == 0] = np.nan
mb_17[:, mask == 0] = np.nan

# plt.figure()
# # plt.imshow(acc_18[-1, :, :] - abl_18[-1, :, :], origin=0)
# # plt.colorbar()

plt.figure()
plt.imshow(acc_17[-1, :, :] - abl_17[-1, :, :], origin=0)
plt.colorbar()

plt.figure()
plt.imshow(acc_17[-1, :, :], origin=0)
plt.colorbar()

plt.figure()
plt.imshow(abl_17[-1, :, :], origin=0)
plt.colorbar()

plt.figure()
plt.imshow(swe_17[-1, :, :], origin=0)
plt.colorbar()

plt.figure()
plt.plot(np.nanmean(water_17, axis=(1, 2)) - np.nanmean(abl_17, axis=(1, 2)) + np.nanmean(acc_17, axis=(1, 2)), label='precipitation')  # precipitation
plt.plot(np.nanmean(water_17, axis=(1, 2)), label='runoff')  # water output
plt.plot(np.nanmean(acc_17, axis=(1, 2)), label='accumulation')
plt.plot(np.nanmean(abl_17, axis=(1, 2)), label='melt')
plt.plot(np.nanmean(swe_17, axis=(1, 2)), label='snow storage')

# plot mean swe with contours over it
plt.figure()
#nc_file = nc.Dataset(r"T:\DSC-Snow\runs\output\nevis_2D_test_erebus\nevis_2D_test_erebus26_norton_9825edc_snow_zenith.nc") # with azumith hack
topo_file = nc.Dataset(r"P:\Projects\DSC-Snow\runs\input_DEM\Clutha_nztm250m_topo_no_ice.nc")
plt.imshow(np.mean(nc_file17.variables['snow_water_equivalent'][:],axis=0),origin=0)
plt.colorbar()
plt.contour(topo_file.variables['DEM'][:],range(0,3000,100),colors='k',linewidths=0.5)

plt.show()