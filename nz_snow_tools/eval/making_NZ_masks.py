"""

create masks for NZ and SI wide domains for simulations

Jono Conway
"""

import matplotlib.pylab as plt
import numpy as np
from nz_snow_tools.util.utils import setup_nztm_dem

# set up modis 250m nz grid
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

si_ew_extent= np.logical_and(x_centres>1.08e6,x_centres<1.72e6)
si_ns_extent= np.logical_and(y_centres<5.52e6, y_centres>4.82e6)
si_mask = si_ns_extent[:,np.newaxis]*si_ew_extent[np.newaxis,:]

modis_ew_extent = np.logical_and(x_centres>1.085e6,x_centres<1.72e6)
modis_si_mask = si_ns_extent[:,np.newaxis]*modis_ew_extent[np.newaxis,:]

plt.imshow(nztm_dem,origin=0)
plt.imshow(modis_nz_mask,origin=0,alpha=0.2)
plt.imshow(modis_si_mask,origin=0,alpha=0.2)
plt.imshow(si_mask,origin=0,alpha=0.2)
plt.savefig(r'C:\Users\conwayjp\OneDrive - NIWA\projects\CARH2101\snow reanalysis\nz_dem_masks.png',dpi=300)

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

plt.savefig(r'C:\Users\conwayjp\OneDrive - NIWA\projects\CARH2101\snow reanalysis\modis_nz_masks.png',dpi=300)
plt.show()

plt.figure()
plt.hist(nztm_dem2[NZ_mask_modis_nz].ravel(), np.arange(0, 3600, 100), density=True, histtype='step',cumulative=-1)
plt.hist(nztm_dem2[SI_mask_modis_nz].ravel(), np.arange(0, 3600, 100), density=True, histtype='step',cumulative=-1)

plt.figure()
# plot histogram of elevation in bins.
n_NZ, bins_NZ, _ = plt.hist(nztm_dem2[NZ_mask_modis_nz].ravel(), np.arange(0, 3600, 100))
n, bins, _ = plt.hist(nztm_dem2[SI_mask_modis_nz].ravel(), np.arange(0, 3600, 100))
fig,ax = plt.subplots()
ax.barh(bins_NZ[:-1] + 50, n_NZ / 16, height=100,label='NZ')
ax.barh(bins[:-1] + 50, n / 16, height=100,label='SI')
plt.legend(loc=1)
plt.xlim(-100, 52000)
plt.ylabel('Elevation (m)')
plt.xlabel('Area (km^2)')
plt.savefig(r'C:\Users\conwayjp\OneDrive - NIWA\projects\CARH2101\snow reanalysis\hypsometry-100m.png',dpi=300)
plt.show()