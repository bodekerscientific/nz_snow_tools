import os

from setuptools import setup, find_packages


PACKAGE_NAME = "nz_snow_tools"
AUTHOR = "Jono Conway"
AUTHOR_EMAIL = "jono@bodekerscientific.com"
DESCRIPTION = "Tools to run and evaluate snow models"


version = None
exec(open('nz_snow_tools/version.py').read())


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name=PACKAGE_NAME,
    version=version,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    license="MIT",
    packages=find_packages(exclude=["tests"]),
    long_description=read('README.rst'),
    url='https://github.com/bodekerscientific/nz_snow_tools',  # use the URL to the github repo
    download_url='https://github.com/bodekerscientific/nz_snow_tools/archive/{}.tar.gz'.format(version),
    install_requires=[
        'matplotlib',
        'netCDF4',
        'numpy',
        'basemap',
        'pyshp',
        'pillow',
        'pyproj'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: System Administrators',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',

        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Atmospheric Science'
    ],
    keywords='snow climate model evaluation',
)
