import numpy as np
import pickle
from scipy import optimize as opt
from . import plot

class MuLensEvent(object):    
    def __init__(self,name,telescope,AOR,TIMES,XDATA,YDATA,IMG,IMG_E):
        self.name      = name
        self.telescope = telescope
        self.AOR       = AOR
        self.TIMES     = TIMES
        self.XDATA     = XDATA
        self.YDATA     = YDATA
        self.IMG       = IMG
        self.IMG_E     = IMG_E
        self.nb_dit    = len(IMG)
        self.img_size  = np.shape(IMG)[3]
        
    def save(self,filename=None):
        """Saves this MuLensEvent to a pickle file for quick access.
        """
        if filename is None:
            filename = 'data/'+self.name+'/PLD_input/'+self.name+'_'+self.telescope+'.pkl'
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        return
    
    def aperture_photometry(self,min=None,max=None):
        """Summing pixel values over given range (or the entire 2D array if no range input).
        Args:
            IMG (list): list of image stacks for each positions. 
        Returns:
            list: list of 1D flux array for each dither positions.
        """
        # filling list with 1 array per dither positions
        if min is None and max is None:
            PTOT = np.sum(self.IMG,axis=(2,3))
            PTOT_E = np.sqrt(np.sum(np.asarray(self.IMG_E)**2,axis=(2,3)))
            PTOT_BIN = np.median(PTOT,axis=0)
            E_BIN = np.std(PTOT - PTOT_BIN)
        else:        
            PTOT = np.sum(self.IMG[:,:,min:max,min:max],axis=(2,3))
            PTOT_E = np.sqrt(np.sum(np.asarray(self.IMG_E[:,:,min:max,min:max])**2,axis=(2,3)))
            PTOT_BIN = np.median(PTOT,axis=0)
            E_BIN = np.std(PTOT - PTOT_BIN)
        return PTOT,PTOT_E,E_BIN

    def get_PNORM(self, min=None, max=None):
        """Returns P-Hat, i.e. the fraction of total flux recorded by each pixels.
        Args:
            IMG (list): list of image stacks for each positions.
            PTOT (list): list of 1D flux array for each dither positions.
            min (int): minimum bound range if you want to perform aperture photometry on a subset of the image.
            max (int): maximum bound range if you want to perform aperture photometry on a subset of the image.
        Returns:
            list: list of PNORMS stacks for each dither positions.
        """
        PTOT,_,_ = self.aperture_photometry(min,max)
        if min is None and max is None:
            PNORM = np.asarray(self.IMG)/PTOT[:,:,None,None]
        else:
            PNORM = np.asarray(self.IMG[:,:,min:max,min:max])/PTOT[:,:,None,None]
        return PNORM
    
    def chrono_flatten(self,*argv):
        """Returns lst flattened and sorted chronologically as per this event's time vector
        Args:
            lst (list): list to be sorted and flattened, must have same fundamental size as TIME
        Returns:
            1D array: sorted and flattened lists
        """
        t_flat = self.TIMES.flatten()
        ind = np.argsort(t_flat)
        returnv = [t_flat[ind]]
        for lst in argv:
            assert np.shape(lst)[:2] == self.TIMES.shape
            lst_flat = np.reshape(lst,[t_flat.size]+list(np.shape(lst)[2:]))
            returnv.append(lst_flat[ind])
        return tuple(returnv)
    
    def modelfit(self,func,p0,PTOT,makeplots=False,**kwargs):
        time,ptot = self.chrono_flatten(PTOT)
        popt, pcov = opt.curve_fit(f=func,xdata=time,ydata=ptot,p0=p0,**kwargs) 
        perr = np.sqrt(np.diag(pcov)) # assuming uncorrelated
        bestfit = func(time,*popt)
        resi = ptot-bestfit
        if makeplots:
            timeplot = np.linspace(np.min(time)-30, np.max(time)+30, 1000)
            lcguess = func(timeplot,*p0)
            lcoptim = func(timeplot,*popt)
            plot.plot_guess(time, ptot, timeplot, lcguess, lcoptim, guessing=False)
            return popt,perr,bestfit,resi,timeplot,lcoptim
        return popt,perr,bestfit,resi,None,None