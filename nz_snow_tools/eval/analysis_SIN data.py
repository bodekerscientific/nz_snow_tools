import numpy as np
from nz_snow_tools.snow.clark2009_snow_model import snow_main_simple
from nz_snow_tools.util.utils import make_regular_timeseries,convert_datetime_julian_day,convert_dt_to_hourdec,nash_sut, mean_bias, rmsd, mean_absolute_error
import matplotlib.pylab as plt
import datetime as dt
import matplotlib.dates as mdates
from nz_snow_tools.util.utils import fill_timeseries
from nz_snow_tools.eval.utils_Ambre import maxmin

#CASTLE MOUNT Clark2009, Albedo and VCSN [2012-2016]
# C_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Castle Mount/CastleMount_npy files/Castle Mount_clark2009_{}.npy"
# A_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Castle Mount/CastleMount_npy files/Castle Mount_dsc_snow-param albedo_{}.npy"
# VC_file = "C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Castle Mount/CastleMount_npy files/Castle Mount_VC_{}.npy"
# VN_file = "C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Castle Mount/CastleMount_npy files/Castle Mount_VN_{}.npy"
# LARKINS Clark2009, Albedo and VCSN [2014-2018]
# C_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Larkins/Larkins_npy files/Larkins_clark2009_{}.npy"
# A_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Larkins/Larkins_npy files/Larkins_dsc_snow-param albedo_{}.npy"
# VC_file = "C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Larkins/Larkins_npy files/Larkins_VC_{}.npy"
# VN_file = "C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Larkins/Larkins_npy files/Larkins_VN_{}.npy"
# MAHANGA Clark2009, Albedo and VCSN [2009-2018]
# C_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mahanga/Mahanga_npy files/Mahanga_clark2009_{}.npy"
# A_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mahanga/Mahanga_npy files/Mahanga_dsc_snow-param albedo_{}.npy"
# VC_file = "C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mahanga/Mahanga_npy files/Mahanga_VC_{}.npy"
# VN_file = "C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mahanga/Mahanga_npy files/Mahanga_VN_{}.npy"
# MUELLER Clark2009, Albedo and VCSN [2011-2018]
C_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mueller/Mueller_npy files/Mueller_clark2009_{}.npy"
A_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mueller/Mueller_npy files/Mueller_dsc_snow-param albedo_{}.npy"
VC_file = "C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mueller/Mueller_npy files/Mueller_VC_{}.npy"
VN_file = "C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mueller/Mueller_npy files/Mueller_VN_{}.npy"
# MURCHISON Clark2009 and Albedo [2009-2018]
# C_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Murchison/Murchison_npy files/Murchison_clark2009_{}.npy"
# A_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Murchison/Murchison_npy files/Murchison_dsc_snow-param albedo_{}.npy"
# PHILISTINE Clark2009, Albedo and VCSN [2011-2018]
# C_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Philistine/Philistine_npy files/Philistine_clark2009_{}.npy"
# A_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Philistine/Philistine_npy files/Philistine_dsc_snow-param albedo_{}.npy"
# VC_file = "C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Philistine/Philistine_npy files/Philistine_VC_{}.npy"
# VN_file = "C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Philistine/Philistine_npy files/Philistine_VN_{}.npy"

import csv
Stname = ['Mueller']
with open("C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Analysis/{}_Max&Min.csv".format(Stname[0]),mode='w', newline='') as maxmin_file:
    maxmin_writer = csv.writer(maxmin_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    fieldnames = ['Data', 'Year', 'SWE max', 'Date max', 'SWE min', 'Date min']
    maxmin_writer.writerow(['Data', 'Year', 'SWE max', 'Date max', 'SWE min', 'Date min'])
    maxmin_writer = csv.DictWriter(maxmin_file, fieldnames=fieldnames)
    for i in range (0,8) :
        Year = 2011 + i
        # load clark2009 model
        inp_clark = np.load(C_file.format(Year),allow_pickle=True)
        inp_time1 = inp_clark[:,0]
        inp_swe1 = np.asarray(inp_clark[:,1],dtype=np.float)
        plot_dt = inp_clark[:, 0] # model stores initial state

        # plt.plot(inp_time1, inp_swe1, color ="red")
        # plt.show()


        # load dsc_param_albedo model
        inp_albedo = np.load(A_file.format(Year),allow_pickle=True)
        inp_time2 = inp_albedo[:,0]
        inp_swe2 = np.asarray(inp_albedo[:,1],dtype=np.float)

        # plt.plot(inp_time2, inp_swe2, color ="aquamarine")
        # plt.show()

        # load VC model
        inp_VC = np.load(VC_file.format(Year),allow_pickle=True)
        inp_time3 = inp_VC[:,0]
        inp_swe3 = np.asarray(inp_VC[:,1],dtype=np.float)

        plt.plot(inp_time3, inp_swe3, color ="darkolivegreen")
        # plt.show()
        # plt.close()

        # load VN model
        if Year <= 2017 :
            inp_VN = np.load(VN_file.format(Year),allow_pickle=True)
            inp_time4 = inp_VN[:,0]
            inp_swe4 = np.asarray(inp_VN[:,1],dtype=np.float)

            plt.plot(inp_time4, inp_swe4, color ="chartreuse")
            # plt.show()
            # plt.close()
        else :
            print('No data')

        # MUELLER SWE csv file
        csv_file = "C:/Users/Bonnamourar/OneDrive - NIWA/CSV SWE/Mueller SWE.csv"
        # MAHANGA SWE csv file
        # csv_file = "C:/Users/Bonnamourar/OneDrive - NIWA/CSV SWE/Mahanga SWE.csv"
        # LARKINS SWE csv file
        # csv_file ="C:/Users/Bonnamourar/OneDrive - NIWA/CSV SWE/Larkins SWE.csv"
        # CASTLE MOUNT SWE csv file
        # csv_file = "C:/Users/Bonnamourar/OneDrive - NIWA/CSV SWE/Castle Mount SWE.csv"
        # MURCHISON SWE csv file
        # csv_file = "C:/Users/Bonnamourar/OneDrive - NIWA/CSV SWE/Murchison SWE.csv"
        # PHILISTINE SWE csv file
        # csv_file = "C:/Users/Bonnamourar/OneDrive - NIWA/CSV SWE/Philistine SWE.csv"

        # load observed data
        inp_datobs = np.genfromtxt(csv_file, delimiter=',',usecols=(1),
                                skip_header=4)*1000
        inp_timeobs = np.genfromtxt(csv_file, usecols=(0),
                                 dtype=(str), delimiter=',', skip_header=4)
        inp_dtobs = np.asarray([dt.datetime.strptime(t, '%d/%m/%Y %H:%M') for t in inp_timeobs])
        ind = np.logical_and(inp_dtobs >= plot_dt[0],inp_dtobs <= plot_dt[-1])

        inp_dtobs_clean, inp_datobs_clean = fill_timeseries(inp_dtobs[ind], inp_datobs[ind], 3600)

        # plt.plot(inp_dtobs_clean,inp_datobs_clean,"o", label = "Observed SWE", color = "blue")
        # plt.show()

        # mean = np.mean(inp_datobs_clean)
        # print('mean observed :',mean)

        # Max and Min values, observed data
        try :
            maximum_observed, minimum_observed, date_max_observed, date_min_observed = maxmin(inp_dtobs_clean, inp_datobs_clean)
            print('Max observed :', maximum_observed,'Min observed :', minimum_observed,'Date max :', date_max_observed,'Date min :', date_min_observed)
        except :
            print('ERROR {}'.format(Year))

        # Max and Min values, clark2009
        try :
            maximum_clark, minimum_clark, date_max_clark, date_min_clark = maxmin(inp_time1, inp_swe1)
            print('Max clark :', maximum_clark,'Min clark :', minimum_clark,'Date max :', date_max_clark,'Date min :', date_min_clark)
        except :
            print('ERROR {}'.format(Year))

        # Max and Min values, albedo
        try :
            maximum_albedo, minimum_albedo, date_max_albedo, date_min_albedo = maxmin(inp_time2, inp_swe2)
            print('Max albedo :', maximum_albedo,'Min albedo :', minimum_albedo,'Date max :', date_max_albedo,'Date min :', date_min_albedo)
        except :
            print('ERROR {}'.format(Year))

        # Max and Min values, VC
        maximum_VC, minimum_VC, date_max_VC, date_min_VC = maxmin(inp_time3, inp_swe3)
        print('Max VC :', maximum_VC,'Min VC :', minimum_VC,'Date max :', date_max_VC,'Date min :', date_min_VC)

        # Max and Min values, VN
        try :
            maximum_VN, minimum_VN, date_max_VN, date_min_VN = maxmin(inp_time4, inp_swe4)
            print('Max VN :', maximum_VN,'Min VN :', minimum_VN,'Date max :', date_max_VN,'Date min :', date_min_VN)
        except :
            print('ERROR {}'.format(Year))
        import csv
        plt.plot(inp_dtobs_clean,inp_datobs_clean)




        # csv file writing
        try :
            maxmin_writer.writerow({'Data' : 'Observed', 'Year': Year, 'SWE max' : maximum_observed, 'Date max' : date_max_observed, 'SWE min' : minimum_observed, 'Date min' : date_min_observed})
        except :
            print('ERROR {} observed'.format(Year))
        try :
            maxmin_writer.writerow({'Data': 'Clark2009', 'Year': Year, 'SWE max': maximum_clark, 'Date max': date_max_clark, 'SWE min': minimum_clark,'Date min': date_min_clark})
        except :
            print('ERROR {} clark'.format(Year))
        try :
            maxmin_writer.writerow({'Data': 'dsc_snow-param albedo', 'Year': Year, 'SWE max': maximum_albedo, 'Date max': date_max_albedo, 'SWE min': minimum_albedo,'Date min': date_min_albedo})
        except :
            print('ERROR {} albedo'.format(Year))
        maxmin_writer.writerow({'Data': 'VC', 'Year': Year, 'SWE max': maximum_VC, 'Date max': date_max_VC,'SWE min': minimum_VC, 'Date min': date_min_VC})
        try :
            maxmin_writer.writerow({'Data': 'VN', 'Year': Year, 'SWE max': maximum_VN, 'Date max': date_max_VN, 'SWE min': minimum_VN,'Date min': date_min_VN})
        except :
            print('ERROR {} VN'.format(Year))