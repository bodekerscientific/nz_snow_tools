"""
import snow data from SIN sites
JOno Conway
"""

import numpy as np
import matplotlib.pylab as plt
import datetime as dt
import pickle

outfile = r'C:\Users\conwayjp\OneDrive - NIWA\projects\DSC Hydro\sin_snow_data\sin data June2019 python36.pkl'
infile = r"C:\Users\conwayjp\OneDrive - NIWA\projects\DSC Hydro\Ambre\SIN calibration timeseries\SIN\SIN Data_June2019_incl density columns.csv"
dat = np.genfromtxt(infile, delimiter=',', skip_header=4)

headers = np.genfromtxt(infile, delimiter=',', skip_footer=110334, dtype=(str))
inp_time = np.genfromtxt(infile, delimiter=',', skip_header=4, usecols=(0), dtype=(str))
inp_dtobs = np.asarray([dt.datetime.strptime(t, '%d/%m/%Y %H:%M') for t in inp_time])

dict_sin_snow = {}

for site in headers[0]:
    dict_sin_snow[site] = {}

vars = [h.split('.')[0] for h in headers[2]]
sites = headers[0]

for i in range(len(sites)):
    if i == 0:
        dict_sin_snow['dt_UTC+12'] = inp_dtobs
    else:
        dict_sin_snow[sites[i]][vars[i]] = dat[:, i]

pickle.dump(dict_sin_snow, open(outfile, 'wb'), protocol=3)

print()
