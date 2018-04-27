nz_snow_tools
====

A suite of python code to run and evaluate snow models developed as part of the Deep South National Science Challenge "Frozen Water" project.


Installation
============

Install the latest version of the package using pip:

    pip install nz_snow_tools


Getting Started
===============

How to use the package....

The package has been tested in Python 2.7


The snow model is called from run_snow_model (for 2D met input) or run_snow_model_simple (for 1D met input)

A configuration dictionary can be used to change the default parameter values in the snow model. This dictionary specifies the parameter values, which will override the default values specified in the functions.