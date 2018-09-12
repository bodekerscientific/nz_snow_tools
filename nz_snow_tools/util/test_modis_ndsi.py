
import netCDF4 as nc
import numpy as np
import matplotlib.pylab as plt

point_x = 611
point_y = 937

nc_file = nc.Dataset(r"T:\sync_to_data\MODIS_snow\NSDI_SI_cloudfilled\DSC_MOD10A1_2000_v0_nosparse_interp001.nc")
ndsi_raw = nc_file.variables['NDSI_Snow_Cover'][:,point_y,point_x]
ndsi_filled = nc_file.variables['NDSI_Snow_Cover_Cloudfree'][:,point_y,point_x]
nc_dt = nc.num2date(nc_file.variables['time'][:], nc_file.variables['time'].units)
y = nc_file.variables['y'][:]
x = nc_file.variables['x'][:]

trimmed_ndsi = ndsi_filled.astype(np.float32, copy=False)
trimmed_fsca = -1 + 1.45 * trimmed_ndsi  # convert to snow cover fraction in % (as per Modis collection 5)
trimmed_fsca[trimmed_ndsi > 100] = np.nan  # set all points with inland water or ocean(237 or 239) to -999, then convert to nan once trimmed
trimmed_fsca[trimmed_fsca > 100] = 100  # limit fsca to 100%
trimmed_fsca[trimmed_fsca < 0] = 0  # limit fsca to 0

nc2 = nc.Dataset(r"Z:\MODIS_snow\MODIS_NetCDF\fsca_2001hy_comp9.nc")
fsca_old = nc2.variables['fsca'][:,point_y,point_x-460]
nc2_dt = nc.num2date(nc2.variables['time'][:], nc2.variables['time'].units)
nc2_e = nc2.variables['easting'][:]
nc2_n = nc2.variables['northing'][:]

fig1 = plt.figure()
plt.plot(nc_dt,ndsi_filled)
plt.plot(nc_dt,ndsi_raw,'o')
plt.plot(nc2_dt,fsca_old)
plt.plot(nc_dt,trimmed_fsca)
plt.legend(['cloud filled ndsi', 'ndsi','fsca(Todd)','fsca(new)'],loc=7)
plt.title('x= {}, y= {}  '.format(x[point_x],y[point_y]) + 'e2= {}, n2= {}'.format(nc2_e[point_x-460],nc2_n[point_y]))
plt.show()
