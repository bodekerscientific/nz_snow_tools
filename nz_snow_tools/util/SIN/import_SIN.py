"""
import data from SIN sites
JOno Conway
"""

import numpy as np
import matplotlib.pylab as plt

infile = r"C:\Users\conwayjp\OneDrive - NIWA\projects\DSC Hydro\Ambre\SIN calibration timeseries\SIN\SIN Data_June2019_incl density columns.csv"
dat = np.genfromtxt(infile, delimiter=',', skip_header=4)

inp_time = np.genfromtxt(infile, delimiter=',', usecols = (1), dtype=(str))

print()
