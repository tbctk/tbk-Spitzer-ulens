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

The ``data_config`` module contains tools to configure the raw data you have downloaded for use with the Spitzer-ulens package. To start, you will want to create a PLDEventData object for the event you have selected. To create a PLDEventData object, you need to specify the directory where the data is stored, as well as the coordinates of the event. To recursively search the directory for relevant data files, set ``recursive=True`` in the PLDEventData constructor. Additional optional arguments can be seen on the ``data_config`` module page. Ground-based data from the `OGLE Early Warning System`_ can be added using the ``add_OGLE_data`` method. The following code block shows an example usage of the ``data_config`` module. In this example, Spitzer data is located in the 'raw_spitzer_data/' directory, and the OGLE data is in the 'ogle_data.dat' file.

::

    from Spitzer_ulens import data_config
    # Specify the directory where your data will be stored:
    src_dir = 'raw_spitzer_data'
    # Specify the event coordinates:
    coords = ('17:47:31.93','-24:31:21.6')
    # Create the PLDEventData object:
    evt = data_config.PLDEventData(src_dir,coords,recursive=True)
    # Add ground-based data:
    evt.add_OGLE_data('ogle_data.dat')
    
Model Selection
---------------


    


.. `OGLE Early Warning System`_: http://ogle.astrouw.edu.pl/ogle4/ews/ews.html
.. `github repository`_: https://github.com/tbctk/tbk-Spitzer-ulens