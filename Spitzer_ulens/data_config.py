"""Contains tools for extracting and formatting data for use with PLD.


"""

import os
import re
import pickle
import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel
from astropy import units as u
from astropy.coordinates import Angle

class PLDEventData(object):
    """Used to extract and format Spitzer data for use with PLD.
    
    A PLDEventData object is initialized with the coordinates of the event and a string 
    specifying a path to the source directory for the data. The resulting object will store 
    the (5-by-5, or custom size given by the 'box' parameter) images, in the vicinity of 
    the event coordinates. OGLE data can be added as well, and the flux, errors, etc. can 
    all be obtained.
    
    Attributes:
        cbcd_pattern: String representing a regex pattern for CBCD of the desired format.
        src: List of string source directory paths linked to this PLDEventData object.
        coords: Tuple of strings (ra,dec), containing the RA and declination of the event, 
            where ra is in the format 'hh:mm:ss.ss' and dec in the format 'dd:mm:ss.ss'.
        channel: Integer Spitzer IRAC channel to use, defaults to 1 (3.6 microns).
        time: List of times corresponding to each image, separated by dither position.
        img: List of cropped CBCD, separated by dither position.
        img_err: List of cropped CBUNC, separated by dither position.
        t_g: Ground-based time-series data.
        mag_g: Ground-based magnification data.
        mag_err_g: Ground-based magnification error data.
        flux_g: Calculated ground-based flux data.
        flux_err_g: Calculated ground-based flux error data.
        ndit: Integer number of dither positions for the Spitzer data.
        box: Integer width & height of the images, defaults to 5.
    """
    
    def __init__(self,src,coords,channel=1,recursive=False,box=5,subtract_2450000=False,ogle_data=None):
        """Initializes a PLDEventData object.
        
        Args:
            src (str): Path to source folder containing data.
            coords (tuple of str): The RA and declination of the event, (ra,dec), where 
                ra is in the format 'hh:mm:ss.ss' and dec in the format 'dd:mm:ss.ss'.
            channel (int): IRAC channel to use. Select 1 for 3.6 microns, or 2 for 4.5
                microns. Defaults to 1.
            recursive (bool): If ``True``, searches subdirectories recursively. Defaults
                to ``False``.
            box (int): Pixel width of cropped images. Defaults to 5.
            subtract_2450000 (bool): If ``True``, subtracts 2450000 from all times.
                Defaults to ``False``.
            ogle_data (str): Path to OGLE data file as donwloaded from OGLE Early Warning 
                System.
        """
        
        if box%2 == 0:
            raise Exception("Parameter 'box' must be an odd integer.")
        
        # Setting attributes
        self.cbcd_pattern = '^SPITZER_I%i_[0-9]{6,10}_[0-9]{3}[1-9]_0000_[1-2]_cbcd.fits$'%channel
        #self.cbunc_pattern = '^SPITZER_I%i_[0-9]{6,10}_[0-9]{3}[1-9]_0000_1_cbunc.fits$'%channel
        self.src = [src]
        self.coords = coords
        self.channel = channel
        self.time = []
        self.img = []
        self.img_err = []
        self.flux_s = None
        self.flux_err_s = None
        self.flux_frac = None
        self.flux_scatter = None
        self.t_g = None
        self.mag_g = None
        self.mag_err_g = None
        self.flux_g = None
        self.flux_err_g = None
        self.ndit = 0
        self.box = box
        
        t0 = 0 if subtract_2450000 else 2450000
        
        # Search src directory for fits files
        if recursive:
            centroid_data = self.extract_centroid_data_recursive(src)
        else:
            centroid_data = self.extract_centroid_data(src)
        if len(centroid_data) > 0:
            # Separate into dithers and get images
            expids = np.array(centroid_data)[:,1]
            dithers = np.unique(expids)
            self.ndit = len(dithers)
            arrs = np.array(list(map(list,zip(*centroid_data))))
            for i in dithers:
                ind = np.array(expids)==i
                dithered = arrs[:,ind]
                aorkey,expid,time,xp,yp,cbcd_filepath,cbunc_filepath = dithered
                xp = np.array(xp,dtype=float)
                yp = np.array(yp,dtype=float)
                time = np.array(time,dtype=float)+t0
                x0 = round(np.median(xp)-1)
                y0 = round(np.median(yp)-1)
                img = np.empty((len(aorkey),box,box))
                img_err = np.empty((len(aorkey),box,box))
                for i,k in enumerate(aorkey):
                    img[i] = self.target_image_square(cbcd_filepath[i],x0,y0,box=box)
                    img_err[i] = self.target_image_square(cbunc_filepath[i],x0,y0,box=box)
                self.time.append(time)
                self.img.append(img)
                self.img_err.append(img_err)
        try:
            self.flux_s,self.flux_err_s,self.flux_frac,self.flux_scatter = self.aperture_photometry()
        except:
            pass
        if ogle_data is not None:
            self.add_OGLE_data(ogle_data,subtract_2450000=subtract_2450000)
        
    @staticmethod
    def target_image_square(filepath,xp,yp,box=5):
        """Extract small square centered at event centroid.
        
        Takes as input a path to a FITS file along with a pair of x- and y-coordinates
        and returns a 5-by-5 (or other width specified by ``box``) image centered at
        these selected coordinates.
        
        Args:
            filepath (str): Path to FITS file.
            xp (int): Pixel x-coordinate.
            yp (int): Pixel y-coordinate.
            box (int): Pixel image width, defaults to 5.
        """
        hdu_list = fits.open(filepath)
        full_img = hdu_list[0].data
        hdu_list.close()
        
        half = int((box-1)/2)
        xmin = xp-half
        ymin = yp-half
        xmax = xmin+box
        ymax = ymin+box
        
        return full_img[ymin:ymax,xmin:xmax]
    
    @staticmethod
    def read_fits_file(cbcd_filepath,coords,origin=1):
        """Reads FITS file header to obtain centroid information.
        
        Args:
            cbcd_filepath (str): Path to CBCD FITS file.
            coords (tuple of str): The RA and declination of the event, (ra,dec), where 
                ra is in the format 'hh:mm:ss.ss' and dec in the format 'dd:mm:ss.ss'.
            origin (int): Pixel indexing origin for FITS image, defaults to 1.
            
        Returns:
            aorkey: String representing a unique code the epoch.
            expid: Integer exposure ID, or the dither position number.
            time: Float; time that image was taken, in HJD-2450000.
            xp: Float; exact x-coordinate of event centroid on the image.
            yp: Float; exact y-coordinate of event centroid on the image.
            cbcd_filepath: String path to CBCD fits file.
            cbunc_filepath: String path to corresponding CBUNC FITS error file.
        """
        # Open fits file and obtain header
        hdu_list = fits.open(cbcd_filepath)
        header = hdu_list[0].header
        hdu_list.close()
        # Extract the data we need
        aorkey = header['AORKEY'] # AOR key of this image
        expid = int(header['EXPID']) # Get the dither number (exposure id)

        time = float(header['BMJD_OBS']-5e4) # Time of image in Reduced Helioc. Mod. Julian Date (may need to change this)

        w = WCS(header)
        ra = Angle(coords[0],unit='hourangle')
        dec = Angle(coords[1],unit='deg')
        coords = SkyCoord(ra,dec)
        (xp,yp) = skycoord_to_pixel(coords,w,mode='wcs',origin=origin) # Convert to pixel coordinates
        
        cbunc_filepath = cbcd_filepath[:-9]+'cbunc.fits'
        
        return aorkey,expid,time,xp,yp,cbcd_filepath,cbunc_filepath
    
    def extract_centroid_data_recursive(self,src):
        """Recursively search directory for FITS files and extract centroid data.
        
        Performs a depth-first search of the directory structure rooted at src. Calls
        the ``read_fits_file`` method on every FITS file found that matches the CBCD
        pattern ``cbcd_pattern``, and returns a list of the results.
        
        Args:
            src (str): Path to root directory for search.
        
        Returns:
            List of tuples (aorkey,expid,time,xp,yp,cbcd_filepath,cbunc_filepath) for
            each CBCD FITS file found. See ``read_fits_file`` for more information.
        """
        centroid_data = []
        def rec(src):
            # Inner recursive function to search the file tree and extract data
            for fname in os.listdir(src):
                path = os.path.join(src,fname)
                if os.path.isdir(path):
                    # Recursively search subdirectories
                    rec(path)
                else:
                    if re.match(self.cbcd_pattern,fname,re.I):
                        # Get centroid data
                        centroid_data.append(list(self.read_fits_file(path,self.coords)))
        rec(src)
        return centroid_data
    
    def extract_centroid_data(self,src):
        """Search directory for FITS files and extract centroid data.
        
        Searches ``src`` folder for FITS files. Calls the ``read_fits_file`` method on 
        every FITS file found that matches the CBCD pattern ``cbcd_pattern``, and returns 
        a list of the results.
        
        Args:
            src (str): Path to root directory for search.
        
        Returns:
            List of tuples (aorkey,expid,time,xp,yp,cbcd_filepath,cbunc_filepath) for
            each CBCD FITS file found. See ``read_fits_file`` for more information.
        """
        centroid_data = []
        for fname in os.listdir(src):
            path = os.path.join(src,fname)
            if os.path.isfile(path):
                if re.match(self.cbcd_pattern,fname,re.I):
                    # Get flux data from CBCD
                    centroid_data.append(self.read_fits_file(path,self.coords))
        return centroid_data
    
    def add_OGLE_data(self,datafile,subtract_2450000=False):
        """Add ground-based observations to this PLDEventData.
        
        Takes in data files formatted as per the data from the OGLE Early Warning System,
        where the first three columns represent the time, magnification, and error on the
        magnification.
        
        Args:
            datafile (str): Path to OGLE data file.
            subtract_2450000 (bool): If ``True``, subtracts 2450000 from all times.
                Defaults to ``False``.
        """
        time,mag,mag_err = np.loadtxt(datafile,usecols=(0,1,2)).T
        self.src.append(datafile)
        if subtract_2450000:
            time -= 2450000
        
        self.t_g = time
        self.mag_g = mag
        self.mag_err_g = mag_err
        self.flux_g = 10**(-(mag-18)/2.5)
        self.flux_err_g = np.sqrt((10**(-(mag-18)/2.5)*(-0.4)*np.log(10))**2*mag_err**2)
        
    def save(self,filepath='pld_event_data.pkl',overwrite=False):
        """Saves the PLDEventData instance as a pickle file.
        
        Args:
            filepath (str): Path at which to store the pickled object. Defaults to
                'pld_event_data.pkl'.
            overwrite (bool): If ``True``, will overwrite the file path. Otherwise raises
                an error if the file already exists. Defaults to ``False``.
        """
        if os.path.exists(filepath) and not overwrite:
            raise Exception('Path %s already points to a file.'%filepath)
        else:
            with open(filepath,'wb') as output:
                pickle.dump(self,output,pickle.HIGHEST_PROTOCOL)
    
    @classmethod
    def from_pickle(filepath):
        """Loads a PLDEventData from a pickle file.
        
        Args:
            filepath (str): Path from which to load the object.
        
        Returns:
            A PLDEventData instance loaded from the pickle file.
        """
        with open(filepath, 'rb') as file:
            event = pickle.load(file)
        if isinstance(event,PLDEventData):
            return event
        else:
            raise Exception('Could not parse file at %s to a PLDEventData object.'%filepath)
            
    def aperture_photometry(self):
        """Get sattelite flux data for this event.
        
        A getter method for space-based flux, flux error, fractional flux, and raw scatter on
        the flux. Also calculates these values from the raw images if not already done.
        
        Returns:
            flux: Array of flux data.
            flux_err: Array of flux error data.
            flux_frac: Array of fractional flux data.
            flux_scatter: Float, estimate raw scatter in the flux data.
        """
        if (self.flux_s is not None and 
                self.flux_err_s is not None and 
                self.flux_frac is not None and 
                self.flux_scatter is not None):
            return self.flux_s,self.flux_err_s,self.flux_frac,self.flux_scatter
        else:
            flux = []
            flux_err = []
            flux_frac = []
            for i,img in enumerate(self.img):
                tmp = np.sum(img,axis=(1,2))
                flux.append(tmp)
                flux_err.append(np.sum(self.img_err[i],axis=(1,2)))
                flux_frac.append(img/tmp[:,None,None])
            flux_med = np.median(flux,axis=0)
            flux_scatter = np.std(flux - flux_med)
            return np.array(flux),np.array(flux_err),np.array(flux_frac),np.array(flux_scatter)