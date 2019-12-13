"""
plots points on

Author: J. Conway
Date: Sept 2019

"""

import numpy as np
import matplotlib.pylab as plt

import pickle as pkl
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import cartopy.feature as feature
from nz_snow_tools.util.utils import setup_nztm_dem

store_folder = 'C:/Users/conwayjp/OneDrive - NIWA/projects/DSC Hydro/SIN sites overview/map'

swe_sites = {}
swe_sites['names'] = np.genfromtxt(store_folder + '/SIN swe sites.csv', delimiter=',', skip_header=1, usecols=0, dtype=(str))
swe_sites['easting'] = np.genfromtxt(store_folder + '/SIN swe sites.csv', delimiter=',', skip_header=1, usecols=1)
swe_sites['northing'] = np.genfromtxt(store_folder + '/SIN swe sites.csv', delimiter=',', skip_header=1, usecols=2)
swe_sites['elevation'] = np.genfromtxt(store_folder + '/SIN swe sites.csv', delimiter=',', skip_header=1, usecols=3)

hs_sites = {}
hs_sites['names'] = np.genfromtxt(store_folder + '/SIN snow depth sites.csv', delimiter=',', skip_header=1, usecols=0, dtype=(str))
hs_sites['easting'] = np.genfromtxt(store_folder + '/SIN snow depth sites.csv', delimiter=',', skip_header=1, usecols=1)
hs_sites['northing'] = np.genfromtxt(store_folder + '/SIN snow depth sites.csv', delimiter=',', skip_header=1, usecols=2)
hs_sites['elevation'] = np.genfromtxt(store_folder + '/SIN snow depth sites.csv', delimiter=',', skip_header=1, usecols=3)

dem_folder = 'C:/Users/conwayjp/OneDrive - NIWA/Data/GIS_DATA/Topography/DEM_NZSOS/'
dem = 'nz_dem_500m'
dem_file = dem_folder + dem + '.tif'

nztm_dem, x_centres, y_centres, lat_array, lon_array = setup_nztm_dem(dem_file, extent_w=1.05e6, extent_e=2.10e6, extent_n=6.275e6, extent_s=4.70e6,
                                                                          resolution=500)

# point = wgs84_to_nztm(-45,176)

map_crs = ccrs.TransverseMercator(central_longitude=173.0, central_latitude=0.0, false_easting=1600000, false_northing=10000000, scale_factor=0.9996,globe=ccrs.Globe(ellipse='GRS80'))
data_crs = ccrs.PlateCarree()  # data is in lat/lon coordinates


plt.figure(figsize=(4.5, 5))
ax = plt.axes(projection=map_crs)
ax.axis('off')
plt.xlim(1070000, 1730000)
plt.ylim(4720000, 5550000)
ax.coastlines(resolution='10m', color='black', linewidth=1)
ax.add_feature(feature.LAKES.with_scale('10m'), facecolor='none', edgecolor='k', linewidth=0.5)
plt.pcolormesh(x_centres, y_centres,nztm_dem,cmap=plt.cm.gray_r,vmin=1)#pink_r
# permanent snow layer
# snow_shp = r"C:/Users/conwayjp/OneDrive - NIWA/BS/Manuscripts/NZ permament snow/15-001.002.R1.0_PermanentSnow&Ice_BrunkSirguey/2016_permanent_snow_ice_150k_BrunkSirguey_v1.shp"
# ax.add_geometries(shpreader.Reader(snow_shp).geometries(), map_crs, facecolor="#3B9AB2", edgecolor="#3B9AB2", linewidth=0.5)

for i in range(len(swe_sites['names'])):
    plt.plot(swe_sites['easting'][i], swe_sites['northing'][i], 'or')  # Transform=data_crs
    # plt.annotate(swe_sites['names'][i], (swe_sites['easting'][i], swe_sites['northing'][i]))

# plt.title(var_to_plot)
plt.tight_layout()
plt.savefig(store_folder + '/SWE sites with dem.png',dpi=1200)
print()
