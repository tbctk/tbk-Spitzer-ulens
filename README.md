# Spitzer-ulens

Author: Lisa Dang (McGill University)

This is a pipeline for using an adaptation of Pixel Level Decorrelation (based on Deming et al. 2015) to reduce observations obtained as part of the Galactic Distribution of Planets Spitzer Microlens Parallaxes Program (PI A. Gould)

### Usage

1. Download data from https://sha.ipac.caltech.edu/applications/Spitzer/SHA/. For example, to download event 'ob171140', search by 'observer' -> 'Gould, Andrew', filter by target name 'ob171140', and download all data.

2. Run 'data_directory_structure_generator.ipynb' with src_dir set to the directory where you downloaded the data to.

3. Run get_centroids.ipynb. This should produce a file 'centroid.out' in the directory 'data/(event name, e.g. ob171140/spitzer'.

4. Run PLD_Preparation.

5. Run PLD_Decorrelation