"""This module contains the LCModel abstract base class and some useful implementations of it.

LCModel is used as a wrapper class for a light-curve function, with some extra functionalities. 
To use the LCModel abstract base class, users can either choose one of the included implementations 
or create their own. The models included are the SingleLensModel, BinaryLensModel, and 
SingleLensParallaxModel. The SingleLensModel and BinaryLensModel classes are examples of the 
simplest form of an LCModel. They are simply a function, defined by the 'func' method, wrapped by 
the __call__ method, which allows the model to be created and called as follows:

::

    my_1l_model = SingleLensModel()
    flux = my_1l_model(time,*pars)

The SingleLensParallaxModel is an example of a more complex model that takes advantage of the 
structure of the LCModel. A SingleLensParallaxModel must be initialized with its event coordinates 
and a path to its satellite ephemeris file:

::

    my_1lpar_model = SingleLensParallaxModel()
    flux_satellite,flux_ground = my_1lpar_model(time_satellite,time_ground,*pars)
    
Note that the return type can be different for different abstractions of the LCModel, and this is 
up to the user to handle. See the descriptions of the individual classes for more details on their 
methods and applications.

One can create their own user-defined LCModel by defining a class to extend the LCModel abstract 
base class and then overwrite the 'func' method with their own desired function. For example, a 
Gaussian model:

::

    class GaussianModel(LCModel):
        def func(self,time,mean,std):
            return np.exp(-0.5*((time-mean)/std)**2)

One can also overwrite the '__init__' method to add special properties of the model. For example, 
suppose one wanted to create a single-lens model with one fixed parameter:

::

    class SingleLensModel_FixedFb(LCModel):
        def __init__(self,fb):
            self.fb = fb
        
        def func(self,t0,fs,tE):
            slmodel = models.SingleLensModel()
            return slmodel(self.fb,t0,fs,tE)
"""


import numpy as np
from inspect import signature, BoundArguments
from abc import ABC,abstractmethod
import MulensModel as mm
import VBBinaryLensing as vbbl
from Spitzer_ulens import PLD

class LCModel(ABC):
    """Abstract base class representing a callable light-curve model.
    
    As an abstract class, an LCModel cannot be instantiated. Users must extend the LCModel
    class to use its functionalities. All subclasses must overwrite the ``func`` method.
    """
    def __call__(self,*pars,**kwpars):
        return self.func(*pars,**kwpars)
    
    @abstractmethod
    def func(self,*pars,**kwpars):
        """The light-curve function itself. Must be overwritten in subclasses.
        """
        pass
    
    @staticmethod
    def mag2flux(mag,fb,fs):
        """Converts from magnification to flux.
        
        Args:
            mag (array of float): Array of magnification data.
            fb (float): Baseline flux.
            fs (float): Source flux.
            
        Returns:
            Flux data array.
        """
        return mag*fs+fb
    
    def lnprior(self,pars,bounds):
        """Simple box likelihood priors for MCMC fitting.
        
        Checks whether the given parameters fit within the given bounds. For more complex
        prior functions, this method can be overwritten in the subclass.
        
        Args:
            pars (array of float): Parameters to check.
            bounds (tuple of array of float): A tuple of two arrays, each of the same size as
                ``pars``. The first array represents lower bounds for the parameters, the 
                second upper bounds.
        
        Returns:
            0 if the parameters all fit within the bounds, negative infinity otherwise.
            Since this is a log-likelihood, negative infinity represents a zero likelihood.
        """
        if bounds is None:
            return 0
        else:
            if all(pars>bounds[0]) and all(pars<bounds[1]):
                return 0.0
            else: 
                return -np.inf

    def lnlike(self,pars,time,flux,flux_err,flux_frac,flux_scatter):
        """Computes the log-likelihood for this model to fit the given data.
        
        This function provides a simple metric for estimating fit log-likelihoods for non-
        parallax events. The likelihood function is explained in Dang et al.
        
        Pars:
            pars (array of float): Parameter array to be passed into ``func``.
            time (array of float): Time data array.
            flux (array of float): Flux data array.
            flux_err (array of float): Flux error data array.
            flux_frac (array of float): Fractional flux data array.
            flux_scatter (float): Estimate of raw scatter in the flux data.
        
        Returns:
            A float representing an estimate log-likelihood for the model to fit the data,
            not including priors.
        """
        # solving for PLD coefficients analytically
        Y, Astro, Ps, A, C, E, X = PLD.analytic_solution(time,flux,flux_err,flux_frac,pars,self)

        # Generating time series from bestfit params
        fit,sys,corr,resi = PLD.get_bestfit(A, Ps, X, flux, Astro)
        Ndat = len(flux[0])

        like = 0
        # generating model and calculating the difference with the flux
        for i in range(len(time)):
            # diff (don't forget ravel, otherwise you'll have some matrix operation!)
            diff  = flux[i]-fit[i].ravel()
            diff2 = flux[i]-Astro[i].ravel()
            # likelihood = -0.5*chisq
            inerr = 1/flux_err[i]
            inerr2 = 1/flux_scatter
            like  += -0.5*np.sum(diff**2*inerr**2) + np.sum(np.log(inerr)) - Ndat*0.9189385332046727
            like  += -0.5*np.sum(diff2**2*inerr2**2) + Ndat*np.log(inerr2) - Ndat*0.9189385332046727
        return like

    def lnprob(self,pars,bounds,time,flux,flux_err,flux_frac,flux_scatter):
        """Computes the full log-likelihood for this model to fit the given data, including priors.
        
        Combines the ``lnprior`` and ``lnlike`` functions to provide a full picture of the log-
        likelihood to be used in an MCMC.
        
        Pars:
            pars (array of float): Parameter array to be passed into ``func``.
            bounds (tuple of array of float): Parameter bounds to be passed into ``lnprior``.
            time (array of float): Time data array.
            flux (array of float): Flux data array.
            flux_err (array of float): Flux error data array.
            flux_frac (array of float): Fractional flux data array.
            flux_scatter (float): Estimate of raw scatter in the flux data.
        
        Returns:
            A float representing an estimate log-likelihood for the model to fit the data,
            including priors.
        """
        # get lnprior
        lp = self.lnprior(pars,bounds)
        # if guess is out of bound
        if not np.isfinite(lp):
            return -np.inf
        # calculate posterior
        return lp + self.lnlike(pars,time,flux,flux_err,flux_frac,flux_scatter)
    
class SingleLensModel(LCModel):
    """Simple 4-parameter single-lens model abstraction.
    
    """
    
    def func(self,time,tE,t0,fb,fs):
        """See base class.
        
        """
        ts = (time-t0)/(tE/np.sqrt(12))
        flux = fb+fs/(np.sqrt(ts**2 +1))
        return flux
    
class BinaryLensModel(LCModel):
    """VBBinaryLensing-based 9-parameter binary-lens model abstraction.
    
    """
    
    VBBL = vbbl.VBBinaryLensing()
    VBBL.RelTol = 1e-03
    
    def func(self,time,tE,t0,fb,fs,s,q,rho,alpha,u0):
        """See base class.
        
        """
        mag = self.get_mag(time,tE,t0,s,q,rho,alpha,u0)
        flux = self.mag2flux(mag,fb,fs)
        return flux
             
    def get_mag(self,time,tE,t0,s,q,rho,alpha,u0):
        """Solves for the magnification only.
        
        """
        # Position of the center of the source with respect to the center of mass.
        tau = (time - t0)/tE
        # from Seb. Calchi Novati
        y1 = u0*np.sin(alpha) - tau*np.cos(alpha)
        y2 = -u0*np.cos(alpha) - tau*np.sin(alpha)

        # empty array where magnification will be stored
        mag = np.zeros(len(tau))

        # calculate the magnification at each time
        params = [np.log(s), np.log(q), u0, alpha, np.log(rho), np.log(tE), t0]
        mag = self.VBBL.BinaryLightCurve(params, time, y1, y2)
        return np.array(mag)
    
    def get_caustics(self,time,tE,t0,s,q,rho,alpha,u0):
        """Get data for plotting caustics 
        
        """
        tau = (time - t0)/tE
        # from Seb. Calchi Novati
        y1 = u0*np.sin(alpha) - tau*np.cos(alpha)
        y2 = -u0*np.cos(alpha) - tau*np.sin(alpha)
        def iterate_from(item):
            while item is not None:
                yield item
                item = item.next
        
        solutions = self.VBBL.PlotCrit(s, q)

        # generator function iterating over _sols, _curve, or _point objects 
        # making use of the next keyword
        curves = []
        for curve in iterate_from(solutions.first):
            for point in iterate_from(curve.first):
                curves.append((point.x1, point.x2))

        critical_curves = np.array(curves[:int(len(curves)/2)])
        caustic_curves  = np.array(curves[int(len(curves)/2):])
        return caustic_curves, critical_curves, np.transpose([y1.ravel(),y2.ravel()])
    
class SingleLensParallaxModel(LCModel):
    """9-parameter single-lens model abstraction with ground parallax based on MuLensModel package.
    
    """
    
    def __init__(self,coords,ephemeris_file_path):
        self.satellite = mm.SatelliteSkyCoord(ephemeris_file_path)
        if isinstance(coords,tuple):
            self.coords = '%s %s'%coords
        elif isinstance(coords,str):
            self.coords = coords
        else:
            raise Exception('Unrecognized coordinate format')
    
    def func(self,t_s,t_g,tE,t0,u0,pi_E_N,pi_E_E,fb_g,fs_g,fb_s,fs_s):
        """See base class.
        
        """
        mag_s,mag_g = self.get_mag(t_s,t_g,tE,t0,u0,pi_E_N,pi_E_E)
        
        flux_g = self.mag2flux(mag_g,fb_g,fs_g)
        flux_s = self.mag2flux(mag_s,fb_s,fs_s)
        
        return flux_s,flux_g
    
    def get_mag(self,t_s,t_g,tE,t0,u0,pi_E_N,pi_E_E):
        """Get source magnification.
        
        """
        #time_g, time_s, tE, t0, u0, pi_E_N, pi_E_E, coord)):
        params        = {'t_0': t0, 'u_0': u0, 't_E': tE}
        params_pi_E   = {'pi_E_N': pi_E_N, 'pi_E_E': pi_E_E}
        
        #model = mm.Model({**params}, coords=self.coords)
        
        model_parallax = mm.Model({**params, **params_pi_E}, coords = self.coords) 
        model_parallax.parallax(earth_orbital = False, satellite = True)
        
        mag_g = model_parallax.get_magnification(time=t_g)
        mag_s = model_parallax.get_magnification(time=t_s, 
                                               satellite_skycoord=self.satellite.get_satellite_coords(t_s))
        return mag_s,mag_g
    
    def lnprob(self,l_pars,bounds,t_s,t_g,flux_s,flux_s_err,flux_frac,flux_scatter,flux_g,flux_g_err):
        """See base class.
        
        """
        pars = l_pars[:-2]
        l_s,l_g = l_pars[-2:]
        lp = self.lnprior(pars,bounds)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(pars,t_s,t_g,flux_s,flux_s_err,flux_frac,flux_scatter,flux_g,flux_g_err,l_s,l_g)
        
    def lnprior(self,pars,bounds):
        """See base class.
        
        """
        if bounds is None:
            return 0
        else:
            if all(pars>bounds[0]) and all(pars<bounds[1]):
                return 0.0
            else: 
                return -np.inf
        
    def lnlike(self,pars,t_s,t_g,flux_s,flux_s_err,flux_frac,flux_scatter,flux_g,flux_g_err,l_s,l_g):
        """See base class.
        
        """
        Y, Astro, Ps, A, X, flu_g  = PLD.analytic_solution(t_s,flux_s,flux_s_err,flux_frac,pars,self,t_g)
        # Generating time series from bestfit params
        FIT, SYS, CORR, RESI = PLD.get_bestfit(A, Ps, X, flux_s, Astro)
        like = 0
        for i in range(len(t_s)):
            Ndat = len(flux_s[i])
            diff  = flux_s[i]-FIT[i].ravel()
            diff2 = flux_s[i]-Astro[i].ravel()
            # error
            inerr  = 1/(l_s*flux_s_err[i])
            inerr2 = 1/(flux_scatter)
            # likelihood (np.log(np.sqrt(2*np.pi)) = 0.9189385332046727)
            like  += -0.5*np.sum(diff**2*inerr**2) + np.sum(np.log(inerr)) - Ndat*0.9189385332046727
            like  += -0.5*np.sum(diff2**2/flux_scatter**2) + Ndat*np.log(inerr2) - Ndat*0.9189385332046727
        # likelihood for ground-based
        diffg = flux_g-flu_g
        inerrg = 1/(l_g*flux_g_err)
        like += -0.5*np.sum(diffg**2*inerrg**2) + np.sum(np.log(inerrg)) - len(diffg)*0.9189385332046727
        return like