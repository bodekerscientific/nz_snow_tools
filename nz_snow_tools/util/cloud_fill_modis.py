import netCDF4 as nc
import numpy as np
import matplotlib.pylab as plt
from scipy.interpolate import interp1d

nc_file = nc.Dataset(r"T:\sync_to_data\MODIS_snow\NSDI_SI_cloudfilled\DSC_MOD10A1_2000_v0_nosparse_interp.nc")
nc_dt = nc.num2date(nc_file.variables['time'][:], nc_file.variables['time'].units)
nc_time = nc_file.variables['time'][:]
y = nc_file.variables['y'][:]
x = nc_file.variables['x'][:]

ndsi_filled = np.zeros(nc_file.variables['NDSI_Snow_Cover'].shape, dtype=np.uint8)

for point_y in range(len(y)):
    for point_x in range(len(x)):
        # point_x = 611
        # point_y = 937
        ndsi_raw = nc_file.variables['NDSI_Snow_Cover'][:, point_y, point_x]
        if ndsi_raw[0] == 237: # inland water
            ndsi_filled[:, point_y, point_x] = 237
        elif ndsi_raw[0] == 239: # ocean
            ndsi_filled[:, point_y, point_x] = 239
        else:
            clean_ndsi = ndsi_raw.data.copy()
            clean_ndsi[(ndsi_raw.mask)] = 255  # set masked values to 255 (max for uint8)
            clean_time = nc_time[(clean_ndsi <= 100)]
            clean_ndsi = clean_ndsi[(clean_ndsi <= 100)]
            if len(clean_time) == 0:
                ndsi_filled[:, point_y, point_x] = 200 # no good points
                pass
            else:
                try:
                    f = interp1d(clean_time, clean_ndsi, kind='cubic', bounds_error=False, fill_value=np.mean(clean_ndsi))

                    ndsi_fill = f(nc_time)
                    ndsi_filled[:, point_y, point_x] = ndsi_fill
                except:
                    ndsi_filled[:, point_y, point_x] = 200 # bad interpolation

plt.imshow(np.mean(ndsi_filled,axis=0),origin=0)
plt.show()

# point_x = 611
# point_y = 937
# nc2 = nc.Dataset(r"Z:\MODIS_snow\MODIS_NetCDF\fsca_2001hy_comp9.nc")
# fsca_old = nc2.variables['fsca'][:,point_y,point_x-460]
# nc2_dt = nc.num2date(nc2.variables['time'][:], nc2.variables['time'].units)
# nc2_e = nc2.variables['easting'][:]
# nc2_n = nc2.variables['northing'][:]
#
# fig1 = plt.figure()
# plt.plot(nc_dt,ndsi_filled)
# plt.plot(nc_dt,ndsi_raw,'o')
# plt.plot(nc2_dt,fsca_old)
# plt.legend(['cloud filled ndsi', 'ndsi','fsca(Todd)'],loc=7)
# plt.title('x= {}, y= {}  '.format(x[point_x],y[point_y]) + 'e2= {}, n2= {}'.format(nc2_e[point_x-460],nc2_n[point_y]))
#
