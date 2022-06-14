import numpy as np
import os
import pickle

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel
from astropy import units as u
from astropy.coordinates import Angle

def make_path(dst_dir):
    """
    Creates path of specified directory if it does not already exist
    
    Args:
        dst_dir (str): Path to create.
    """
    pathlist = os.path.normpath(dst_dir).split(os.path.sep)
    path = ''
    for d in pathlist:
        path = os.path.join(path,d)
        if os.path.isfile(path):
            raise Exception("Destination path points to a file")
        elif os.path.isdir(path):
            continue
        else:
            os.mkdir(path)
    return

def move_fits_files_rec(src_dir,dst_dir):
    """
    Recursively searches directory structure for FITS files and puts them in a single data folder
    
    Args:
        src_dir (str): Path to folder in which to search for FITS files.
        dst_dir (str): Path to folder where you wish to store the FITS files.
    """
    data = []
    for f in os.listdir(src_dir):
        src = os.path.join(src_dir,f)
        dst = os.path.join(dst_dir,f)
        if os.path.isfile(src):
            if src.endswith('_cbcd.fits') or src.endswith('_cbunc.fits'):
                os.rename(src,dst)
        elif os.path.isdir(src):
            move_fits_files_rec(src,dst_dir)
    return

def directory_config(evt,telescope,src_dir,rd=""):
    """
    Create directory structure for use with PLD-ulens. Working directory must contain your python project that uses PLD-ulens, otherwise you must specify the project's working directory with the 'wd' input.
    
    Args:
        evt (str): Name of the event, e.g. 'ob171140' for OGLE-2017-BLG-1140.
        telescope (str): Name of the telescope, e.g. 'spitzer'.
        src_dir (str): Path to folder in which to search for FITS files.
        rd (str, optional): Destination root project directory. Defaults to current working directory
    """

    dst_dir = os.path.join(rd,'data',evt,telescope,'images')

    make_path(dst_dir)
    move_fits_files_rec(src_dir,dst_dir)

    # Make input folder for PLD_Decorrelation
    dst_dir2 = os.path.join(rd,'data',evt,'PLD_input')
    make_path(dst_dir2)
    return

def get_centroid_data_from_file(filepath,event_coords,origin=1):
    """ 
    Obtains centroid information from a given file path
    
    Args:
        filepath (str): FITS file path
        event_coords (tuple of str): (ra,dec) where ra='hh:mm:ss.ss' and dec='dd:mm:ss.ss'
        origin (int, optional): Indexing of the top-left pixel. Defaults to 1.
    
    Returns:
        label (str): Unique label of this image
        time (float): Time of image taken in Reduced Helioc. Mod. Julian Date (may need to change this)
        xp (float): Pixel x-coordinate of event
        yp (float): Pixel y-coordinate of event
        fname (str): File name, excluding path
        expid (int): The exposure ID, which tells us which dither position this image is from.
    """
    hdu_list = fits.open(filepath)
    myheader = hdu_list[0].header
    hdu_list.close()
    
    aorkey = myheader['AORKEY'] # AOR key of this image
    expid = myheader['EXPID'] # Get the dither number (exposure id)
    label = str(aorkey)
    
    time = myheader['BMJD_OBS']-5e4 # Time of image in Reduced Helioc. Mod. Julian Date (may need to change this)
    
    w = WCS(myheader)
    ra = Angle(event_coords[0],unit='hourangle')
    dec = Angle(event_coords[1],unit='deg')
    coords = SkyCoord(ra,dec)
    (xp,yp) = skycoord_to_pixel(coords,w,mode='wcs',origin=origin) # Convert to pixel coordinates
    
    _,fname = os.path.split(filepath)
    
    return label,time,xp,yp,fname,expid

def read_centroid_data(dirpath,event_coords,timerange=None,**kwargs):
    """
    Reads each corrected BCD (CBCD) file in the provided data root directory.    
    Args:
        dirpath (str): Path to data root directory
        event_coords (tuple of str): (ra,dec) where ra='hh:mm:ss.ss' and dec='dd:mm:ss.ss'
        timerange (float, optional): Time range to use
        **kwargs: Optional labeled parameters
    
    Returns:
        data (ndarray of str): 4 x n array, where each row contains [label,time,xp,yp] and n is the number of FITS images
    """
    data = []
    for filename in os.listdir(dirpath):
        if filename.startswith('SPITZER_I1_') and filename.endswith('_cbcd.fits') and not filename.endswith('0_0000_1_cbcd.fits'):
            label,time,xp,yp,fname,expid = get_centroid_data_from_file(os.path.join(dirpath,filename),event_coords,**kwargs)
            if (timerange is None) or (time > timerange[0] and time < timerange[1]):
                data.append(np.asarray([label,time,xp,yp,fname,expid]))
    return np.asarray(data)

def generate_centroid_file(dirpath,event_coords,destpath='',**kwargs):
    """
    Reads each corrected BCD (CBCD) file in the provided data root directory and writes the output data to output file destpath/centroid.out. The directory structure must be as downloaded from ipac website: dirpath/xxxxx/ch1/bcd/file_name_cbcd.fits
    
    Args:
        dirpath (str): path to data root directory
        event_coords (tuple of str): (ra,dec) where ra='hh:mm:ss.ss' and dec='dd:mm:ss.ss'
        destpath (str, optional): Destination directory for centroid.out output file
        **kwargs: Optional labeled parameters
        
    Returns:
        centroid_data_array (ndarray of str): 4 x n array, where each row contains [label,time,xp,yp] and n is the number of FITS images
    """
    data = read_centroid_data(dirpath,event_coords,**kwargs)
    ind = np.argsort(data[:,1]) # Sort data chronologically
    centroid_data_array = data[ind]
    np.savetxt(os.path.join(destpath,'centroid.out'),centroid_data_array,fmt='%s',delimiter=' ')
    return centroid_data_array

def get_centroid_data(evt,telescope,evt_coords,**kwargs):
    """
    Gets centroid data for the given event, telescope, and coordinates.
    
    Args:
        evt (str): Abridged name of the event (e.g. 'ob171140' for 'OGLE-2017-BLG-1140')
        telescope (str): Name of the telescope (e.g. 'spitzer')
        evt_coords (tuple of str): (ra,dec) where ra='hh:mm:ss.ss' and dec='dd:mm:ss.ss'
        **kwargs: Optional labeled parameters
        
    Returns:
        centroid_data_array (ndarray of str): 4 x n array, where each row contains [label,time,xp,yp] and n is the number of FITS images
    """
    dirpath = 'data/'+evt+'/'+telescope+'/images'
    destpath = 'data/'+evt+'/'+telescope
    data = generate_centroid_file(dirpath,evt_coords,destpath,**kwargs)
    return divide_per_dither(data)

def load_centroid_data(evt,telescope):
    """
    Load centroid data from file for a given event and telescope
    
    Args:
        evt (str): Abridged name of the event (e.g. 'ob171140' for 'OGLE-2017-BLG-1140')
        telescope (str): Name of the telescope (e.g. 'spitzer')
        
    Returns:
        centroid_data_array (ndarray of str): 4 x n array, where each row contains [label,time,xp,yp] and n is the number of FITS images
    """
    filename = 'data/'+evt+'/'+telescope+'/centroid.out'
    data = np.loadtxt(filename,dtype=str)
    return divide_per_dither(data)

def divide_per_dither(centroid_data):
    """
    Divides centroid data into separate arrays for each dither.
    
    Args:
        centroid_data (ndarray): Centroid data, formatted as per output from load_centroid_data() or get_centroid_data().
        
    Returns:
        AORs (ndarray of str): AOR identifier keys
        times (ndarray of float): Times
        xps (ndarray of float): x-pixel coordinates
        yps (ndarray of float): y-pixel coordinates
        cbcd (ndarray of str): Names of CBCD files
        cbunc (ndarray of str): Names of CBUNC files
    """
    dithers = np.unique(centroid_data[:,5])
    print(dithers)
    data_dit = []
    for i in dithers:
        data_dit.append(centroid_data[centroid_data[:,5] == i])
    data_dit = np.array(data_dit,dtype=object)
    AORs = data_dit[:,:,0]
    times = data_dit[:,:,1].astype(np.float)
    xps = data_dit[:,:,2].astype(np.float)
    yps = data_dit[:,:,3].astype(np.float)
    cbcd = data_dit[:,:,4]
    
    cbunc = []
    for dither in cbcd:
        cbunc.append([name[:-9]+'cbunc.fits' for name in dither])
    return AORs,times,xps,yps,cbcd,np.asarray(cbunc)

def target_central_px(XDATA, YDATA):
    """Get the target's central pixel coordinates for each dither. We need to get the flux from the SAME pixels for pixel level decorrelation.
    
    Args:
        XDATA (list): list that contains nb_dithers arrays of x-centroid obtained from astrometry (xread outputs)
        YDATA (list): list that contains nb_dithers arrays of y-centroid obtained from astrometry (xread outputs)
        
    Returns:
        xcent (list of int): x-coordinate of central pixel
        array (list of int): y-coordinate or central pixel
    """
    # number of dithers
    nb_dithers = len(XDATA)

    # get x-coordinate of central pixel (unsure if floor or round is the way to go)
    xmed = []
    for i in range(nb_dithers):
        xmed.append(np.median(XDATA[i]))
    xcent = np.round(np.array(xmed) - 1)

    # get y-coordinate of central pixel (unsure if floor or round is the way to go)
    ymed = []
    for i in range(nb_dithers):
        ymed.append(np.median(YDATA[i]))
    ycent = np.round(np.array(ymed) - 1)

    return xcent, ycent

def target_image_square(evt, xcent, ycent, CBCD, CBUNC, box=5):
    """Using the central pixel coordinate for each dither, this function will return only the box of enclosing the target of interest.
    
    Args:
        evt (str): event name in the format obYYXXXX.
        xcent (1D array): x-coordinate of central pixel
        ycent (1D array): y-coordinate of central pixel
        CBCD (list) : list of arrays of cbcd fits file names for each dither position
        CBUNC (list): list of arrays of cbunc fits file names for each dither position
        box (int): size of the box of interest. Default value is 5.
        
    Returns:
        image (list): list of 3D-array (image flux stacks) per dither positions
        image_err (list): list of 3D-array (image uncertainty stacks) per dither positions
    """
    nb_dithers = len(xcent)
    # create list where the images stacks for each dithers will be stored
    image      = []
    image_err  = []
    # for each dither positions
    for j in range(nb_dithers):
        # get delimitation of PLD box
        half = int((box-1)/2)
        xbeg = int(xcent[j] - half)
        ybeg = int(ycent[j] - half)
        xend = xbeg + box
        yend = ybeg + box
        # create empty array where image stack for each dither will be stored
        img_tmp     = np.empty((len(CBCD[j]), box, box))
        img_tmp_err = np.empty((len(CBUNC[j]), box, box))
        # for each images with dither position j
        for i in range(len(CBCD[j])):
            path     = 'data/'+evt+'/spitzer/images/'+ CBCD[j][i]
            path_err = 'data/'+evt+'/spitzer/images/'+ CBUNC[j][i]
            # open cbcd and cbunc fits file
            hdu      = fits.open(path)
            hdu_err  = fits.open(path_err)
            # record the box of pixels we want
            img_tmp[i]     = hdu[0].data[ybeg:yend, xbeg:xend]
            img_tmp_err[i] = hdu_err[0].data[ybeg:yend, xbeg:xend]

        image.append(img_tmp)
        image_err.append(img_tmp_err)
    return image, image_err