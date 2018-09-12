
import netCDF4 as nc
import numpy as np
import matplotlib.pylab as plt
#f326c54 # without TOA hack
nc_file = nc.Dataset(r"T:\DSC-Snow\runs\output\nevis_2D_test_erebus\nevis_2D_test_erebus26_norton_9825edc_snow_zenith.nc") # with azimuth fixed
#nc_file = nc.Dataset(r"T:\DSC-Snow\runs\output\nevis_2D_test_erebus\nevis_2D_test_erebus26_norton_f326c54_snow.nc") # without TOA hack
# nc_file = nc.Dataset(r"T:\DSC-Snow\runs\output\nevis_2D_test_erebus\nevis_2D_test_erebus26_norton_425a1d2_snow.nc") # TOA hack to check TOA rad vs measured
# nc_file = nc.Dataset(r"T:\DSC-Snow\runs\output\nevis_2D_test_erebus\nevis_2D_test_erebus26_norton_5bebe0e_snow.nc")
#nc_file = nc.Dataset(r"T:\DSC-Snow\runs\output\nevis_2D_test_erebus\nevis_2D_test_erebus26_ef0a6c5_snow.nc")
#nc_file = nc.Dataset(r"T:\DSC-Snow\runs\output\nevis_2D_test_erebus\nevis_2D_test_erebus26_5ee15cf3e9d6bfb73b75e272aaf5734a8e31a78d.nc")
#nc_file = nc.Dataset(r"T:\DSC-Snow\runs\output\nevis_2D_test_erebus\nevis_2D_test_erebus26_2f173070bd9949d9b54a4bc229e8186f3593ab59.nc")
#nc_file = nc.Dataset(r"T:\DSC-Snow\runs\output\nevis_2D_test_erebus\nevis_2D_test_erebus26.nc")
topo_file = nc.Dataset(r"P:\Projects\DSC-Snow\runs\input_DEM\Nevis_nztm250m_topo_no_ice.nc")
climate_file = nc.Dataset(r"T:\DSC-Snow\input_data_hourly\met_inp_Nevis_nztm250m_2016_norton.nc")

n = 240

y_point = 125 # 144
x_point = 27 # 39

# slope = np.rad2deg(topo_file.variables['slope'][y_point,x_point])
# aspect = np.rad2deg(topo_file.variables['aspect'][y_point,x_point])
#
# print('slope = {} aspect = {}'.format(slope,aspect))
#
# albedo = nc_file.variables['albedo'][:,y_point,x_point]
# sw_net = nc_file.variables['net_shortwave_radiation_at_surface'][:,y_point,x_point]
sw_input = nc_file.variables['incoming_solar_radiation_to_surface_from_input'][:n,y_point,x_point]
# cloud = nc_file.variables['cloud'][:,y_point,x_point]
# sw_in = sw_net/(1.0-albedo)
# plt.plot(sw_input)
# plt.plot(sw_in)
# plt.plot(sw_net)
# plt.plot(cloud*100)

nc_dt = nc.num2date(nc_file.variables['time'][:n], nc_file.variables['time'].units)


plt.figure()
for y_point in range(120,130):
    albedo = nc_file.variables['albedo'][:n, y_point, x_point]
    sw_net = nc_file.variables['net_shortwave_radiation_at_surface'][:n, y_point, x_point]
    #sw_input = nc_file.variables['incoming_solar_radiation_to_surface_from_input'][:, y_point, x_point]
    cloud = nc_file.variables['cloud'][:n, y_point, x_point]
    sw_in = sw_net / (1.0 - albedo)
    slope = np.rad2deg(topo_file.variables['slope'][y_point, x_point])
    aspect = np.rad2deg(topo_file.variables['aspect'][y_point, x_point])

    print('slope = {} aspect = {}'.format(slope, aspect))
    plt.plot(nc_dt,sw_in,label = 'slope = {:2.0f} aspect = {:3.0f}'.format(slope, aspect))


plt.plot(nc_dt,cloud*100,'r--',label='cloud * 100')
plt.plot(nc_dt,sw_input[:n],'ko',label='input')

sw_clim_file = climate_file.variables['surface_downwelling_shortwave_flux'][:n, y_point, x_point]
nc_dt2 = nc.num2date(climate_file.variables['time'][:n], climate_file.variables['time'].units)

plt.plot(nc_dt2,sw_clim_file,label='climate file')

plt.legend(loc=7)
plt.show()