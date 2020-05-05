"""
code to call the snow model for a simple test case using brewster glacier data
"""

import numpy as np
from nz_snow_tools.snow.clark2009_snow_model import snow_main_simple
from nz_snow_tools.util.utils import make_regular_timeseries, nash_sut, mean_bias, rmsd
import matplotlib.pylab as plt
import datetime as dt
import matplotlib.dates as mdates



# load brewster glacier data
inp_dat = np.genfromtxt(
   r'S:\Scratch\Jono\Final Brewster Datasets\updated_met_data\BrewsterGlacier_Oct10_Sep12_mod3.dat')
start_t = 2 - 1 # 9456 = start of doy 130 10th May 2011 9600 = end of 13th May, 18432 = start of 11th Nov 2013,19296 = 1st december 2011
end_t = 3265  # 20783 = end of doy 365, 21264 = end of 10th January 2012
inp_dt = make_regular_timeseries(dt.datetime(2010,10,25,00,30),dt.datetime(2012,9,2,00,00),1800)

inp_doy = inp_dat[start_t:end_t, 2]
inp_hourdec = inp_dat[start_t:end_t, 3]
# make grids of input data
grid_size = 1
grid_id = np.arange(grid_size)
inp_ta = inp_dat[start_t:end_t, 7][:, np.newaxis] * np.ones(grid_size) + 273.16  # 2 metre air temp in C
inp_precip = inp_dat[start_t:end_t, 21][:, np.newaxis] * np.ones(grid_size)  # precip in mm
inp_sw = inp_dat[start_t:end_t, 15][:, np.newaxis] * np.ones(grid_size)
inp_sfc = inp_dat[start_t-1:end_t, 19] # surface height change
inp_sfc -= inp_sfc[0]# reset to 0 at beginning of period


daily_sfc = []
for i in range(47, len(inp_sfc), 48):
    daily_sfc.append(inp_sfc[i])

daily_sfc_melt = np.diff(np.asarray(daily_sfc) * -492.) #492 taken from Cullen et al 2016 Journal of Glaciology

# validation data
seb_dat = np.genfromtxt(
   r'S:\Scratch\Jono\Final Brewster Datasets\SEB_output\cdf - code2p0_MC_meas_noQPS_single_fixed output_fixed_B\modelOUT_br1_headings.txt',skip_header=3)
seb_mb = seb_dat[start_t-1:end_t, -1]
seb_mb -= seb_mb[0] # reset to 0

hourly_seb_melt = np.diff(seb_mb) * -1
#
# # show daily change in SWE
# daily_mb = []
# for i in range(47, len(seb_mb), 48):
#     daily_mb.append(seb_mb[i])


# # read in measured daily SEB change
# mb_dat = np.genfromtxt(
#     r'S:\Scratch\Jono\Final Brewster Datasets\mass_balance_validation\5 MB scatters\mchange.dat')
# # note that the measured MB interprets surface height loss in the winter as mass loss, rather than compaction.
# mb_dt = make_regular_timeseries(dt.datetime(2010,10,26,00,00),dt.datetime(2012,9,2,00,00),86400)
# ts_mb = plt.cumsum(mb_dat[:,0])
# np.where(np.asarray(mb_dt)==dt.datetime(2011,5,13,00,00))
# ts_mb -= ts_mb[199]
#

init_swe = np.ones(inp_ta.shape[1:]) * 10000 # intialise with lots of snow so doesn't melt out#1740  # give initial value of swe as starts in spring
init_d_snow = np.ones(inp_ta.shape[1:]) * 10  # give initial value of days since snow

# configuration dictionary containing model parameters.
config = {}
config['num_secs_output']= 1800
config['tacc'] = 274.16
config['tmelt'] = 274.16

# dsc_snow melt parameters
config['tf'] = 0.05*24  # hamish 0.13
config['rf'] = 0.0094*24 # hamish 0.0075
# albedo parameters
config['dc'] = 11.0
config['tc'] = 10
config['a_ice'] = 0.42
config['a_freshsnow'] = 0.95
config['a_firn'] = 0.62
config['alb_swe_thres'] = 20

# use measured albedo
config['inp_alb'] = inp_dat[start_t:end_t, 16][:, np.newaxis] * np.ones(grid_size)

tmelt_list = np.linspace(-5,5,11) + 273.16
tf_list = np.linspace(0,0.5,11)
rf_list = np.linspace(0,100e-4,11)

ns_array = np.zeros((len(tf_list),len(rf_list)))
mbd_array = np.zeros((len(tf_list),len(rf_list)))
rmsd_array = np.zeros((len(tf_list),len(rf_list)))
h_ns_array = np.zeros((len(tf_list),len(rf_list)))
h_mbd_array = np.zeros((len(tf_list),len(rf_list)))
h_rmsd_array = np.zeros((len(tf_list),len(rf_list)))

for tmelt in tmelt_list:
    config['tmelt'] = tmelt
    # loop to call range of parameters
    for i, tf in enumerate(tf_list):
        for j, rf in enumerate(rf_list):
            config['tf'] = tf * 24  # hamish 0.13
            config['rf'] = rf * 24  # hamish 0.0075
            # call main function once hourly/sub-hourly temp and precip data available.

            st_swe3, st_melt3, st_acc3, st_alb3 = snow_main_simple(inp_ta, inp_precip, inp_doy, inp_hourdec, dtstep=1800, init_swe=init_swe,
                                                       init_d_snow=init_d_snow, inp_sw=inp_sw, which_melt='dsc_snow', **config)


            daily_swe3 = []
            for k in range(47, len(st_swe3[:, 0]), 48):
                daily_swe3.append(st_swe3[k, 0])

            mb_sim = -1 * np.diff(np.asarray(daily_swe3))

            ns_array[i,j]= nash_sut(mb_sim,daily_sfc_melt)
            mbd_array[i,j]= mean_bias(mb_sim,daily_sfc_melt)
            rmsd_array[i,j]= rmsd(mb_sim,daily_sfc_melt)

            h_ns_array[i,j]= nash_sut(-1 *np.diff(st_swe3[:, 0]),hourly_seb_melt)
            h_mbd_array[i,j]= mean_bias(-1 *np.diff(st_swe3[:, 0]),hourly_seb_melt)
            h_rmsd_array[i,j]= rmsd(-1 *np.diff(st_swe3[:, 0]),hourly_seb_melt)

    plt.figure(figsize=[8,3])

    plt.subplot(1,3,1)
    CS = plt.contour(rf_list,tf_list,ns_array)
    plt.clabel(CS,inline=1, fontsize=10)
    plt.title('Nash-Sutcliffe')
    plt.tight_layout()

    plt.subplot(1,3,2)
    CS = plt.contour(rf_list,tf_list,mbd_array)
    plt.clabel(CS,inline=1, fontsize=10)
    plt.title('MBD (sim-obs)')

    plt.subplot(1,3,3)
    CS = plt.contour(rf_list,tf_list,rmsd_array)
    plt.clabel(CS,inline=1, fontsize=10)
    plt.title('RMSD')

    plt.tight_layout()
    plt.savefig('P:/Projects/DSC-Snow/nz_snow_runs/brewster calibration/daily_sfc_fit_metrics2010_tmelt{}.png'.format(tmelt))

    plt.close()
    plt.figure(figsize=[8,3])

    plt.subplot(1,3,1)
    CS = plt.contour(rf_list,tf_list,h_ns_array)
    plt.clabel(CS,inline=1, fontsize=10)
    plt.title('Nash-Sutcliffe')
    plt.tight_layout()

    plt.subplot(1,3,2)
    CS = plt.contour(rf_list,tf_list,h_mbd_array)
    plt.clabel(CS,inline=1, fontsize=10)
    plt.title('MBD (sim-obs)')

    plt.subplot(1,3,3)
    CS = plt.contour(rf_list,tf_list,h_rmsd_array)
    plt.clabel(CS,inline=1, fontsize=10)
    plt.title('RMSD')

    plt.tight_layout()
    plt.savefig('P:/Projects/DSC-Snow/nz_snow_runs/brewster calibration/hourly_SEB_fit_metrics2010_tmelt{}.png'.format(tmelt))
    plt.close()
#repeat for 2011:


# load brewster glacier data
inp_dat = np.genfromtxt(
   r'S:\Scratch\Jono\Final Brewster Datasets\updated_met_data\BrewsterGlacier_Oct10_Sep12_mod3.dat')
start_t = 19296 # 9456 = start of doy 130 10th May 2011 9600 = end of 13th May, 18432 = start of 11th Nov 2013,19296 = 1st december 2011
end_t = 21360  # 20783 = end of doy 365, 21264 = end of 10th January 2012
inp_dt = make_regular_timeseries(dt.datetime(2010,10,25,00,30),dt.datetime(2012,9,2,00,00),1800)

inp_doy = inp_dat[start_t:end_t, 2]
inp_hourdec = inp_dat[start_t:end_t, 3]
# make grids of input data
grid_size = 1
grid_id = np.arange(grid_size)
inp_ta = inp_dat[start_t:end_t, 7][:, np.newaxis] * np.ones(grid_size) + 273.16  # 2 metre air temp in C
inp_precip = inp_dat[start_t:end_t, 21][:, np.newaxis] * np.ones(grid_size)  # precip in mm
inp_sw = inp_dat[start_t:end_t, 15][:, np.newaxis] * np.ones(grid_size)
inp_sfc = inp_dat[start_t-1:end_t, 19] # surface height change
inp_sfc -= inp_sfc[0]# reset to 0 at beginning of period


daily_sfc = []
for i in range(47, len(inp_sfc), 48):
    daily_sfc.append(inp_sfc[i])

daily_sfc_melt = np.diff(np.asarray(daily_sfc) * -492.) #492 taken from Cullen et al 2016 Journal of Glaciology

# validation data
seb_dat = np.genfromtxt(
   r'S:\Scratch\Jono\Final Brewster Datasets\SEB_output\cdf - code2p0_MC_meas_noQPS_single_fixed output_fixed_B\modelOUT_br1_headings.txt',skip_header=3)
seb_mb = seb_dat[start_t-1:end_t, -1]
seb_mb -= seb_mb[0] # reset to 0
hourly_seb_melt = np.diff(seb_mb) * -1
#
# # show daily change in SWE
# daily_mb = []
# for i in range(47, len(seb_mb), 48):
#     daily_mb.append(seb_mb[i])


# # read in measured daily SEB change
# mb_dat = np.genfromtxt(
#     r'S:\Scratch\Jono\Final Brewster Datasets\mass_balance_validation\5 MB scatters\mchange.dat')
# # note that the measured MB interprets surface height loss in the winter as mass loss, rather than compaction.
# mb_dt = make_regular_timeseries(dt.datetime(2010,10,26,00,00),dt.datetime(2012,9,2,00,00),86400)
# ts_mb = plt.cumsum(mb_dat[:,0])
# np.where(np.asarray(mb_dt)==dt.datetime(2011,5,13,00,00))
# ts_mb -= ts_mb[199]
#

init_swe = np.ones(inp_ta.shape[1:]) * 10000# intialise with lots of snow so doesn't melt out#1291  # give initial value of swe as starts in spring 1291
init_d_snow = np.ones(inp_ta.shape[1:]) * 10  # give initial value of days since snow

# configuration dictionary containing model parameters.
config = {}
config['num_secs_output']= 1800
config['tacc'] = 274.16
config['tmelt'] = 274.16

# dsc_snow melt parameters
config['tf'] = 0.05*24  # hamish 0.13
config['rf'] = 0.0094*24 # hamish 0.0075
# albedo parameters
config['dc'] = 11.0
config['tc'] = 10
config['a_ice'] = 0.42
config['a_freshsnow'] = 0.95
config['a_firn'] = 0.62
config['alb_swe_thres'] = 20

# use measured albedo
config['inp_alb'] = inp_dat[start_t:end_t, 16][:, np.newaxis] * np.ones(grid_size)

tmelt_list = np.linspace(-5,5,11) + 273.16
tf_list = np.linspace(0,0.5,11)
rf_list = np.linspace(0,100e-4,11)

ns_array = np.zeros((len(tf_list),len(rf_list)))
mbd_array = np.zeros((len(tf_list),len(rf_list)))
rmsd_array = np.zeros((len(tf_list),len(rf_list)))
h_ns_array = np.zeros((len(tf_list),len(rf_list)))
h_mbd_array = np.zeros((len(tf_list),len(rf_list)))
h_rmsd_array = np.zeros((len(tf_list),len(rf_list)))

for tmelt in tmelt_list:
    config['tmelt'] = tmelt
    # loop to call range of parameters
    for i, tf in enumerate(tf_list):
        for j, rf in enumerate(rf_list):
            config['tf'] = tf * 24  # hamish 0.13
            config['rf'] = rf * 24  # hamish 0.0075
            # call main function once hourly/sub-hourly temp and precip data available.

            st_swe3, st_melt3, st_acc3, st_alb3 = snow_main_simple(inp_ta, inp_precip, inp_doy, inp_hourdec, dtstep=1800, init_swe=init_swe,
                                                       init_d_snow=init_d_snow, inp_sw=inp_sw, which_melt='dsc_snow', **config)


            daily_swe3 = []
            for k in range(47, len(st_swe3[:, 0]), 48):
                daily_swe3.append(st_swe3[k, 0])

            mb_sim = -1 * np.diff(np.asarray(daily_swe3))

            ns_array[i,j]= nash_sut(mb_sim,daily_sfc_melt)
            mbd_array[i,j]= mean_bias(mb_sim,daily_sfc_melt)
            rmsd_array[i,j]= rmsd(mb_sim,daily_sfc_melt)

            h_ns_array[i,j]= nash_sut(-1 *np.diff(st_swe3[:, 0]),hourly_seb_melt)
            h_mbd_array[i,j]= mean_bias(-1 *np.diff(st_swe3[:, 0]),hourly_seb_melt)
            h_rmsd_array[i,j]= rmsd(-1 *np.diff(st_swe3[:, 0]),hourly_seb_melt)

    plt.figure(figsize=[8,3])

    ax = plt.subplot(1,3,1)
    CS = plt.contour(rf_list,tf_list,ns_array)
    plt.clabel(CS,inline=1, fontsize=10)
    plt.title('Nash-Sutcliffe')
    plt.tight_layout()

    plt.subplot(1,3,2)
    CS = plt.contour(rf_list,tf_list,mbd_array)
    plt.clabel(CS,inline=1, fontsize=10)
    plt.title('MBD (sim-obs)')

    plt.subplot(1,3,3)
    CS = plt.contour(rf_list,tf_list,rmsd_array)
    plt.clabel(CS,inline=1, fontsize=10)
    plt.title('RMSD')

    plt.tight_layout()
    plt.savefig('P:/Projects/DSC-Snow/nz_snow_runs/brewster calibration/daily_sfc_fit_metrics2011_tmelt{}.png'.format(tmelt))

    plt.close()
    plt.figure(figsize=[8,3])

    plt.subplot(1,3,1)
    CS = plt.contour(rf_list,tf_list,h_ns_array)
    plt.clabel(CS,inline=1, fontsize=10)
    plt.title('Nash-Sutcliffe')
    plt.tight_layout()

    plt.subplot(1,3,2)
    CS = plt.contour(rf_list,tf_list,h_mbd_array)
    plt.clabel(CS,inline=1, fontsize=10)
    plt.title('MBD (sim-obs)')

    plt.subplot(1,3,3)
    CS = plt.contour(rf_list,tf_list,h_rmsd_array)
    plt.clabel(CS,inline=1, fontsize=10)
    plt.title('RMSD')

    plt.tight_layout()
    plt.savefig('P:/Projects/DSC-Snow/nz_snow_runs/brewster calibration/hourly_SEB_fit_metrics2011_tmelt{}.png'.format(tmelt))
    plt.close()