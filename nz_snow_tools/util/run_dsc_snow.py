# script to run the Fortran version of dsc_snow model and iterate through different namelist options, file names etc

import subprocess
import os
import shutil
import f90nml

# compatibility for windows or linux
projects =  '/shareddrive/Projects' #
temp = '/mnt/temp'
# data2 =  '/mnt/data2'

path_to_ddf_run = "/home/jono/dsc_snow/ddf/ddf_run" # path to fortran executable
inp_id = 'jobst_ucc' # string identifying the input data used (suffix to input netcdf files)
config_id_in = '4' # string identifying suffix to the namelist configuration file used as default
config_id_out = '4' # string identifying suffix to the namelist configuration file used by the model
path_to_namelist = projects + '/DSC-Snow/runs/input/clutha_nztm250m_erebus/ddf_config_{}.txt'.format(config_id_in  )
years = range(2000, 2016 + 1)

for year in years:
    # read in namelist
    nml = f90nml.read(path_to_namelist)
    # modify for given run
    nml['ddf_config']['ClimateSource'] = temp + '/DSC-Snow/input_data_hourly/met_inp_Clutha_nztm250m_{}_{}.nc'.format(year, inp_id)
    nml['ddf_config']['OutputFile'] = projects + '/DSC-Snow/runs/output/clutha_nztm250m_erebus/Clutha_nztm250m_{}_{}_{}.nc'.format(year, inp_id, config_id_out)
    nml['ddf_config']['starttime'][0] = year
    nml['ddf_config']['endtime'][0] = year + 1
    # write namelist
    path_to_output_namelist = projects + '/DSC-Snow/runs/input/clutha_nztm250m_erebus/ddf_config_{}_{}_{}.txt'.format(year, inp_id, config_id_out)
    nml.write(path_to_output_namelist,force=True)

    # # copy the met input onto P drive
    # metinfile = nml['ddf_config']['ClimateSource']
    # shutil.copy(data2 + '/DSC-Snow/input_data_hourly/' + metinfile.split('/')[-1], metinfile)

    # run model
    args = [path_to_ddf_run, path_to_output_namelist]
    subprocess.call(args)

    # move output file to Temp drive
    outfile = nml['ddf_config']['OutputFile']
    shutil.move(outfile, temp + '/DSC-Snow/runs/output/clutha_nztm250m_erebus/' + outfile.split('/')[-1])

    # # remove the met input from P drive
    # os.remove(metinfile)
