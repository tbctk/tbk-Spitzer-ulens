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

The ``data_config`` module contains tools to configure the raw data you have downloaded for use with the Spitzer-ulens package. To start, 

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