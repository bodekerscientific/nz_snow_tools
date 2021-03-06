"""

create masks for NZ and SI wide domains for simulations

Jono Conway
"""

import matplotlib.pylab as plt
import numpy as np
from nz_snow_tools.util.utils import setup_nztm_dem

# # set up modis 250m nz grid
nztm_dem2, x_centres2, y_centres2, lat_array2, lon_array2 = setup_nztm_dem(None, extent_w=1.085e6, extent_e=2.10e6, extent_n=6.20e6, extent_s=4.70e6,
                                                                           resolution=250, origin='bottomleft')
# read in 250m grid topography from
nztm_dem, x_centres, y_centres, lat_array, lon_array = setup_nztm_dem(r"C:\Users\conwayjp\OneDrive - NIWA\Data\GIS_DATA\Topography\DEM_NZSOS\nz_dem_250m.tif",
                                                                      extent_w=1.05e6, extent_e=2.10e6, extent_n=6.275e6,
                                                                      extent_s=4.70e6, resolution=250, origin='bottomleft')

# mask for nz modis domain on nz 250m dem domain
modis_nz_ew_extent = np.logical_and(x_centres > 1.085e6, x_centres < 2.10e6)
modis_nz_ns_extent = np.logical_and(y_centres < 6.20e6, y_centres > 4.70e6)
modis_nz_mask = modis_nz_ns_extent[:, np.newaxis] * modis_nz_ew_extent[np.newaxis, :]
np.save(r'C:\Users\conwayjp\OneDrive - NIWA\projects\CARH2101\snow reanalysis\modis_nz_mask_on_nz_dem_250m.npy', modis_nz_mask)
np.save(r'C:\Users\conwayjp\OneDrive - NIWA\projects\CARH2101\snow reanalysis\modis_NZ_nz_dem_250m.npy', modis_nz_mask)

NZ_mask_nz = nztm_dem > 0
np.save(r'C:\Users\conwayjp\OneDrive - NIWA\projects\CARH2101\snow reanalysis\NZ_nz_dem_250m.npy', NZ_mask_nz)

si_ew_extent = np.logical_and(x_centres > 1.08e6, x_centres < 1.72e6)
si_ns_extent = np.logical_and(y_centres < 5.52e6, y_centres > 4.82e6)
si_mask = si_ns_extent[:, np.newaxis] * si_ew_extent[np.newaxis, :]

modis_ew_extent = np.logical_and(x_centres > 1.085e6, x_centres < 1.72e6)
modis_si_mask = si_ns_extent[:, np.newaxis] * modis_ew_extent[np.newaxis, :]

plt.imshow(nztm_dem, origin=0)
plt.imshow(NZ_mask_nz, origin=0, alpha=.2)
plt.imshow(modis_nz_mask, origin=0, alpha=0.2)
plt.imshow(modis_si_mask, origin=0, alpha=0.2)
plt.imshow(si_mask, origin=0, alpha=0.2)
plt.savefig(r'C:\Users\conwayjp\OneDrive - NIWA\projects\CARH2101\snow reanalysis\nz_dem_masks.png', dpi=300)

plt.figure()
# trim 250m dem to nz modis domain
nztm_dem2 = nztm_dem[modis_nz_mask].reshape(lat_array2.shape)
plt.imshow(nztm_dem2, origin=0, interpolation='none')
np.save(r'C:\Users\conwayjp\OneDrive - NIWA\projects\CARH2101\snow reanalysis\modis_nz_dem_250m.npy', nztm_dem2)

# plot pre-computed clutha catchemnt on NZ modis domain
clutha_mask = np.load(r"C:\Users\conwayjp\Downloads\ORC_Clutha_NZTM_modis_nz_dem_250m.npy")
plt.imshow(clutha_mask, origin=0, alpha=.2)

# mask for SI modis domain
modis_si_mask_modis_nz = np.logical_and(y_centres2 < 5.52e6, y_centres2 > 4.82e6)[:, np.newaxis] * np.logical_and(x_centres2 > 1.085e6, x_centres2 < 1.72e6)[
                                                                                                   np.newaxis, :]
plt.imshow(modis_si_mask_modis_nz, origin=0, alpha=.2)
np.save(r'C:\Users\conwayjp\OneDrive - NIWA\projects\CARH2101\snow reanalysis\modis_si_mask_on_modis_nz_dem_250m.npy', modis_si_mask_modis_nz)

# mask for SI wide domain - all non-ocean points
SI_mask_modis_nz = np.logical_and(modis_si_mask_modis_nz, nztm_dem2 > 0)
plt.imshow(SI_mask_modis_nz, origin=0, alpha=.2)
np.save(r'C:\Users\conwayjp\OneDrive - NIWA\projects\CARH2101\snow reanalysis\SI_modis_nz_dem_250m.npy', SI_mask_modis_nz)

# mask for NZ wide domain - all non-ocean points
NZ_mask_modis_nz = nztm_dem2 > 0
plt.imshow(NZ_mask_modis_nz, origin=0, alpha=.2)
np.save(r'C:\Users\conwayjp\OneDrive - NIWA\projects\CARH2101\snow reanalysis\NZ_modis_nz_dem_250m.npy', NZ_mask_modis_nz)

plt.savefig(r'C:\Users\conwayjp\OneDrive - NIWA\projects\CARH2101\snow reanalysis\modis_nz_masks.png', dpi=300)
# plt.show()

# plt.figure()
# plt.hist(nztm_dem2[NZ_mask_modis_nz].ravel(), np.arange(0, 3600, 100), density=True, histtype='step', cumulative=-1)
# plt.hist(nztm_dem2[SI_mask_modis_nz].ravel(), np.arange(0, 3600, 100), density=True, histtype='step', cumulative=-1)

n, bins, _ = plt.hist(nztm_dem2[SI_mask_modis_nz].ravel(), np.arange(0, 3600, 10))
plt.figure()
plt.plot(np.cumsum(n[::-1])/16.,bins[1:][::-1])
plt.grid('on')
plt.ylim(bottom=0)
plt.xlim(left=-1000,right = 155e3)
plt.ylabel('Elevation (m)')
plt.xlabel('Cumulative area (km2)')
plt.savefig(r'C:\Users\conwayjp\OneDrive - NIWA\projects\CARH2101\snow reanalysis\hypsometry-10m cumsum.png', dpi=300)
# plt.show()
# np.where(bins[1:][::-1]==1100)
# (np.cumsum(n[::-1])/16.)[249]


plt.figure()
# plot histogram of elevation in bins.
n_NZ, bins_NZ, _ = plt.hist(nztm_dem2[NZ_mask_modis_nz].ravel(), np.arange(0, 3600, 100))
n, bins, _ = plt.hist(nztm_dem2[SI_mask_modis_nz].ravel(), np.arange(0, 3600, 100))
fig, ax = plt.subplots()
ax.barh(bins_NZ[:-1] + 50, n_NZ / 16, height=100, label='NZ')
ax.barh(bins[:-1] + 50, n / 16, height=100, label='SI')
plt.legend(loc=1)
plt.xlim(-100, 52000)
plt.ylabel('Elevation (m)')
plt.xlabel('Area (km^2)')
plt.savefig(r'C:\Users\conwayjp\OneDrive - NIWA\projects\CARH2101\snow reanalysis\hypsometry-100m.png', dpi=300)
# plt.show()


plt.figure()

nztm_dem, x_centres, y_centres, lat_array, lon_array = setup_nztm_dem(r"C:\Users\conwayjp\OneDrive - NIWA\Data\GIS_DATA\Topography\DEM_NZSOS\si_dem_250m.tif",
                                                                      extent_w=1.08e6, extent_e=1.72e6, extent_n=5.52e6, extent_s=4.82e6,
                                                                      resolution=250, origin='bottomleft')

pisa_mask_si_dem = np.logical_and(y_centres > 4.995e6,y_centres < 5.055e6)[:, np.newaxis] * np.logical_and(x_centres > 1.255e6,x_centres < 1.315e6)[ np.newaxis, :]
np.save(r'C:\Users\conwayjp\OneDrive - NIWA\projects\CARH2101\snow reanalysis\Pisa_si_dem_250m.npy', pisa_mask_si_dem)

modis_si_dem_250m = nztm_dem[:, 20:]
plt.imshow(modis_si_dem_250m, origin=0, interpolation='none')
np.save(r'C:\Users\conwayjp\OneDrive - NIWA\projects\CARH2101\snow reanalysis\modis_si_dem_250m.npy', modis_si_dem_250m)

mask = nztm_dem * 0
mask[1000:1500, 1000:1500] = 1
plt.imshow(mask, origin='lower', alpha=.2, interpolation='none')
np.save(r'C:\Users\conwayjp\OneDrive - NIWA\projects\CARH2101\snow reanalysis\MtCook_large_si_dem_250m.npy', mask)

mask = nztm_dem * 0
mask[1300:1400, 1100:1200] = 1
plt.imshow(mask, origin='lower', alpha=.2, interpolation='none')
np.save(r'C:\Users\conwayjp\OneDrive - NIWA\projects\CARH2101\snow reanalysis\MtCook_si_dem_250m.npy', mask)




# 'modis_si_dem_250m':
nztm_dem, x_centres, y_centres, lat_array, lon_array = setup_nztm_dem(None, extent_w=1.085e6, extent_e=1.72e6, extent_n=5.52e6, extent_s=4.82e6,
                                                                      resolution=250, origin='bottomleft')

# assert modis_si_dem_250m == nztm_dem2[modis_si_mask_modis_nz]

SI_mask_modis_si = modis_si_dem_250m > 0
plt.imshow(SI_mask_modis_si, origin=0, alpha=.2)
np.save(r'C:\Users\conwayjp\OneDrive - NIWA\projects\CARH2101\snow reanalysis\SI_modis_si_dem_250m.npy', SI_mask_modis_si)

plt.figure()

plt.imshow(modis_si_dem_250m - nztm_dem2[modis_si_mask_modis_nz].reshape(modis_si_dem_250m.shape), origin=0, interpolation='none')

plt.figure()
plt.imshow(SI_mask_modis_si - nztm_dem2[modis_si_mask_modis_nz].reshape(modis_si_dem_250m.shape) > 0, origin=0, interpolation='none')
plt.show()


#to get correct coordinates of trimmed modis and model grids

from nz_snow_tools.util.utils import trim_data_to_mask,trim_lat_lon_bounds

# # set up modis 250m nz grid
nztm_dem2, x_centres2, y_centres2, lat_array2, lon_array2 = setup_nztm_dem(None, extent_w=1.085e6, extent_e=2.10e6, extent_n=6.20e6, extent_s=4.70e6,
                                                                           resolution=250, origin='bottomleft')
# read in 250m grid topography from
nztm_dem, x_centres, y_centres, lat_array, lon_array = setup_nztm_dem(r"C:\Users\conwayjp\OneDrive - NIWA\Data\GIS_DATA\Topography\DEM_NZSOS\nz_dem_250m.tif",
                                                                      extent_w=1.05e6, extent_e=2.10e6, extent_n=6.275e6,
                                                                      extent_s=4.70e6, resolution=250, origin='bottomleft')

# mask for nz modis domain on nz 250m dem domain
modis_nz_ew_extent = np.logical_and(x_centres > 1.085e6, x_centres < 2.10e6)
modis_nz_ns_extent = np.logical_and(y_centres < 6.20e6, y_centres > 4.70e6)
modis_nz_mask = modis_nz_ns_extent[:, np.newaxis] * modis_nz_ew_extent[np.newaxis, :]
nztm_dem2 = nztm_dem[modis_nz_mask].reshape(lat_array2.shape)

NZ_mask_nz = nztm_dem > 0
NZ_mask_modis_nz = nztm_dem2 > 0

#to get trimmed coordinates for modis domain trimmed to elevation > 0
trim_lat_lon_bounds(NZ_mask_modis_nz, lat_array2, lon_array2, nztm_dem2, y_centres2, x_centres2)

print(NZ_mask_nz.shape)
print(NZ_mask_modis_nz.shape)
#trim 116 points off the top of the NZ masked domain to make the same as the modis masked domain
assert np.all(trim_lat_lon_bounds(NZ_mask_modis_nz, lat_array2, lon_array2, nztm_dem2, y_centres2, x_centres2)[0] == trim_lat_lon_bounds(NZ_mask_nz, lat_array, lon_array,nztm_dem, y_centres, x_centres)[0][:-116])

print(trim_data_to_mask(NZ_mask_modis_nz,NZ_mask_modis_nz).shape)



