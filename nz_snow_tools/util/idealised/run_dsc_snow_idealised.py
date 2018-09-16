# script to run the Fortran version of dsc_snow model and iterate through different namelist options, file names etc

import subprocess
import os
import shutil
import f90nml

# compatibility for windows or linux
projects = '/mnt/shareddrive/Projects'  #
temp = '/mnt/temp'
# data2 =  '/mnt/data2'

origin = 'topleft'  # or 'bottomleft'
path_to_ddf_run = "/home/bs/dsc_snow/ddf/ddf_run"  # path to fortran executable
config_id_in = '5'  # string identifying suffix to the namelist configuration file used as default
config_id_out = '5'  # string identifying suffix to the namelist configuration file used by the model
path_to_namelist = projects + '/DSC-Snow/runs/input/clutha_nztm250m_erebus/ddf_config_{}.txt'.format(config_id_in)
years = [2016]

met_inps = ['flat_2000', 'north_facing', 'south_facing', 'bell_4000', 'real']

for met_inp in met_inps:
    for year in years:
        for catchment in met_inps:
            # read in namelist
            nml = f90nml.read(path_to_namelist)
            # modify for given run
            nml['ddf_config']['TopographyFile'] = projects + '/DSC-Snow/runs/idealised/{}_nztm250m_topo_no_ice_origin{}.nc'.format(catchment, origin)
            nml['ddf_config']['ClimateSource'] = projects + '/DSC-Snow/runs/idealised/met_inp_{}_nztm250m_{}_origin{}.nc'.format(met_inp, year, origin)
            nml['ddf_config']['OutputFile'] = projects + '/DSC-Snow/runs/idealised/snow_out_{}_dem{}_met{}_{}_origin{}.nc'.format(year, catchment, met_inp,
                                                                                                                                  config_id_out, origin)
            nml['ddf_config']['starttime'][0] = year
            nml['ddf_config']['endtime'][0] = year + 1
            # write namelist
            path_to_output_namelist = projects + '/DSC-Snow/runs/idealised/ddf_config_{}_dem{}_met{}_{}_origin{}.txt'.format(year, catchment, met_inp,
                                                                                                                             config_id_out, origin)
            nml.write(path_to_output_namelist, force=True)

            # run model
            args = [path_to_ddf_run, path_to_output_namelist]
            subprocess.call(args)
