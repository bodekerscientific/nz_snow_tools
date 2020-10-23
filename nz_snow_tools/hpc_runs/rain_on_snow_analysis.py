import numpy as np
import matplotlib.pylab as plt
import netCDF4 as nc

nc_file = r"C:\Users\conwayjp\OneDrive - NIWA\projects\CARH2101\snow reanalysis\rain_on_snow\snow_out_nzcsm7-12_clark2009_MtCook_si_dem_250m_cl09_default_ros_2018.nc"
ds = nc.Dataset(nc_file,'r')

np.sum(ds.variables['ros'])
np.sum(ds.variables['rain'])
np.sum(ds.variables['acc'])
np.sum(ds.variables['melt'])
np.sum(ds.variables['swe'])
np.mean(ds.variables['ros'])
np.mean(ds.variables['rain'])
np.mean(ds.variables['acc'])
np.mean(ds.variables['melt'])
np.mean(ds.variables['swe'])

plt.figure()
plt.title('plot final swe')
plt.imshow(ds.variables['swe'][-1],origin='lower')
plt.colorbar()

plt.figure()
plt.title('difference between total accum and melt and final swe')
plt.imshow((np.sum(ds.variables['acc'],axis=0)-np.sum(ds.variables['melt'],axis=0))-ds.variables['swe'][-1],origin='lower')
plt.colorbar()

plt.figure()
plt.title('total rain-on-snow')
plt.imshow(np.sum(ds.variables['ros'],axis=0),origin='lower')
plt.colorbar()

plt.figure()
plt.title('rain-on-snow as fraction of total precip')
plt.imshow(np.sum(ds.variables['ros'],axis=0)/(np.sum(ds.variables['rain'],axis=0)+np.sum(ds.variables['acc'],axis=0)),origin='lower')
plt.colorbar()

plt.figure()
plt.title('snowfall as fraction of total precipitation')
plt.imshow(np.sum(ds.variables['acc'],axis=0)/(np.sum(ds.variables['rain'],axis=0)+np.sum(ds.variables['acc'],axis=0)),origin='lower')
plt.colorbar()

plt.figure()
plt.title('fraction of total precipitation falling on snow')
plt.imshow((np.sum(ds.variables['acc'],axis=0)+np.sum(ds.variables['ros'],axis=0))/(np.sum(ds.variables['rain'],axis=0)+np.sum(ds.variables['acc'],axis=0)),origin='lower')
plt.colorbar()

plt.figure()
plt.title('amount of rain-on-snow melt as fraction of total precip')
plt.imshow(np.sum(ds.variables['ros_melt'],axis=0)/(np.sum(ds.variables['rain'],axis=0)+np.sum(ds.variables['acc'],axis=0)),origin='lower')
plt.colorbar()

plt.figure()
plt.title('amount of rain-on-snow melt as fraction of total melt (without ros)')
plt.imshow(np.sum(ds.variables['ros_melt'],axis=0)/np.sum(ds.variables['melt'],axis=0),origin='lower')
plt.colorbar()

plt.figure()
plt.title('total precip')
plt.imshow((np.sum(ds.variables['rain'],axis=0)+np.sum(ds.variables['acc'],axis=0)),origin='lower')
plt.colorbar()

plt.figure()
plt.title('total rain-on-snow melt')
plt.imshow(np.sum(ds.variables['ros_melt'],axis=0),origin='lower')
plt.colorbar()


plt.show()
