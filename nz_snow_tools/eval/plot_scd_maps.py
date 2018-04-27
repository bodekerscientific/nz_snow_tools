"""
code to plot maps of snow covered area for individual years from summary lists generated by catchment_evalutation.py
"""

from __future__ import division

import numpy as np
import pickle
import matplotlib.pylab as plt

average_scd = False # boolean specifying if all years are to be averaged together
which_model = 'all'  # string identifying the model to be run. options include 'clark2009', 'dsc_snow', or 'all' # future will include 'fsm'
clark2009run = True  # boolean specifying if the run already exists
dsc_snow_opt = 'python'  # string identifying which version of the dsc snow model to use output from 'python' or 'fortran'
catchment = 'Nevis'
output_dem = 'nztm250m'  # identifier for output dem
hydro_years_to_take = range(2001, 2016 + 1)  # [2013 + 1]  # range(2001, 2013 + 1)
modis_sc_threshold = 50  # value of fsca (in percent) that is counted as being snow covered
model_output_folder = 'P:/Projects/DSC-Snow/nz_snow_runs'
plot_folder = 'P:/Projects/DSC-Snow/nz_snow_runs'

ann = pickle.load(open(model_output_folder + '/summary_{}_{}.pkl'.format(catchment, output_dem), 'rb'))
# indexes 0-3 modis, 4-8 model 1 and 9-13 model 2
# ann = [ann_ts_av_sca_m, ann_hydro_days_m, ann_dt_m, ann_scd_m,
# ann_ts_av_sca, ann_ts_av_swe, ann_hydro_days, ann_dt, ann_scd,
# ann_ts_av_sca2, ann_ts_av_swe2, ann_hydro_days2, ann_dt2, ann_scd2]

ann_scd_m = ann[3]
ann_scd = ann[8]
ann_scd2 = ann[13]


plt.tight_layout()
if average_scd ==True:
    modis_scd = np.mean(np.asarray(ann_scd_m), axis=0)
    mod1_scd = np.mean(np.asarray(ann_scd), axis=0)
    if which_model == 'all':
        mod2_scd = np.mean(np.asarray(ann_scd2), axis=0)

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(modis_scd, origin=0, interpolation='none', vmin=0, vmax=365, cmap='viridis')
    plt.colorbar()
    plt.title('modis duration fsca > {}'.format(modis_sc_threshold))

    plt.subplot(1, 3, 2)
    plt.imshow(mod1_scd, origin=0, interpolation='none', vmin=0, vmax=365, cmap='viridis')
    plt.colorbar()
    if which_model != 'all':
        plt.title('{} duration'.format(which_model))
    if which_model == 'all':
        plt.title('clark2009 duration')
        plt.subplot(1, 3, 3)
        plt.imshow(mod2_scd, origin=0, interpolation='none', vmin=0, vmax=365, cmap='viridis')
        plt.colorbar()
        plt.title('dsc_snow duration')
    plt.savefig(plot_folder + '/SCA hy{} to hy{}.png'.format(hydro_years_to_take[0],hydro_years_to_take[-1]), dpi=300)
else:
    for i, hydro_year_to_take in enumerate(hydro_years_to_take):
        modis_scd = np.asarray(ann_scd_m[i])
        mod1_scd = np.asarray(ann_scd[i])
        if which_model == 'all':
            mod2_scd = np.asarray(ann_scd2[i])

        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(modis_scd, origin=0, interpolation='none', vmin=0, vmax=365, cmap='viridis')
        plt.colorbar()
        plt.title('modis duration fsca > {}'.format(modis_sc_threshold))

        plt.subplot(1, 3, 2)
        plt.imshow(mod1_scd, origin=0, interpolation='none', vmin=0, vmax=365, cmap='viridis')
        plt.colorbar()
        if which_model != 'all':
            plt.title('{} duration'.format(which_model))
        if which_model == 'all':
            plt.title('clark2009 duration')
            plt.subplot(1, 3, 3)
            plt.imshow(mod2_scd, origin=0, interpolation='none', vmin=0, vmax=365, cmap='viridis')
            plt.colorbar()
            plt.title('dsc_snow duration')
        plt.savefig(plot_folder + '/SCA hy{}_{}_{}.png'.format(hydro_year_to_take,catchment, output_dem), dpi=300)

