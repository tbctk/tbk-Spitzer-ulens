Usage
=====

Installation
------------

The Spitzer-ulens package can be installed by downloading the source code from the `github repository`_, then navigating to the installation (Spitzer-ulens) directory and running the command:

::

    python setup.py install
    
..
    TODO: run python setup.py sdist to make a zipped distributable.

Data Download
-------------

This package is designed to work with data as downloaded from the `OGLE Early Warning System`_.

Data Configuration
------------------

The ``data_config`` module contains tools to configure the raw data you have downloaded for use with the Spitzer-ulens package. To start, you will want to run the ``directory_config`` method to generate the directory structure that the package uses, and copy the CBCD and CBUNC into it. This can also be done manually. After calling the ``directory_config`` method, the directory structure should look something like this:

::
    
    Project/
    |--data/
    |  |--[event_name_1]/
    |  |  |--PLD_input/
    |  |  |--spitzer/
    |  |  |  |--images/
    |  |  |  |  |--SPITZER_I1_00000000_0001_0000_1_cbcd.fits
    |  |  |  |  |--SPITZER_I1_00000000_0001_0000_1_cbunc.fits
    |  |  |  |  |--SPITZER_I1_00000000_0002_0000_1_cbcd.fits
    |  |  |  |  |--SPITZER_I1_00000000_0002_0000_1_cbunc.fits
    |  |  |  |  | # You should see all of your CBCD and CBUNC in this folder.

Now you are ready to obtain centroid data from your FITS files. The ``get_centroid_data`` method searches each FITS file header for the AOR key and time stamp of each image, employs the ``astropy.wcs.utils.skycoord_to_pixel`` to get each image's pixel coordinates for the centroid of the microlensing event, and also obtains the file names for the CBCD and CBUNC for each time stamp.

::

    from Spitzer_ulens import data_config
    # Specify the directory where your data will be stored
    src_dir = 'data'
    # Specify the event and telescope names
    event_name = 'obXXXXXX'
    telescope = 'spitzer'
    # Configure the data directory structure
    data_config.directory_config(event_name,telescope,src_dir)
    
    
    


.. `OGLE Early Warning System`_: http://ogle.astrouw.edu.pl/ogle4/ews/ews.html
.. `github repository`_: https://github.com/tbctk/tbk-Spitzer-ulens