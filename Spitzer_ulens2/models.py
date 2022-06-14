import numpy as np
from inspect import signature, BoundArguments
from abc import ABC,abstractmethod
import MulensModel as mm
from Spitzer_ulens import PLD

class LCModel(ABC):

    def __call__(self,*pars,**kwpars):
        return self.func(*pars,**kwpars)
    
    @abstractmethod
    def func(self,*pars,**kwpars):
        pass
    
    @staticmethod
    def mag2flux(mag,fb,fs):
        return mag*fs+fb
    
    def lnprob(self,pars,bounds):
        lp = self.lnprior(p0,bounds)
        # if guess is out of bound
        if not np.isfinite(lp):
            return -np.inf
        # calculate posterior
        return lp + self.lnlike(p0, func, TIMES, PTOT, PTOT_E, E_BIN, PNORM, PLD_chain)
    
    def lnprior(self,pars,bounds):
        return 0
    
    def lnlike(self,pars):
        return 0
    
class SingleLensParallaxModel(LCModel):
    
    def __init__(self,coords,ephemeris_file_path):
        self.satellite = mm.SatelliteSkyCoord('data/ob171140/spitzer/Spitzer_ephemeris_02.dat')
        if isinstance(coords,tuple):
            self.coords = '%s %s'%coords
        elif isinstance(coords,str):
            self.coords = coords
        else:
            raise Exception('Unrecognized coordinate format')
    
    def func(self,t_s,t_g,tE,t0,u0,pi_E_N,pi_E_E,fb_g,fs_g,fb_s,fs_s):

        mag_s,mag_g = self.get_mag(t_s,t_g,tE,t0,u0,pi_E_N,pi_E_E)
        
        flux_g = self.mag2flux(mag_g,fb_g,fs_g)
        flux_s = self.mag2flux(mag_s,fb_s,fs_s)
        
        return flux_s,flux_g
    
    def get_mag(self,t_s,t_g,tE,t0,u0,pi_E_N,pi_E_E):
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
        pars = l_pars[:-2]
        l_s,l_g = l_pars[-2:]
        lp = self.lnprior(pars,bounds)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(pars,t_s,t_g,flux_s,flux_s_err,flux_frac,flux_scatter,flux_g,flux_g_err,l_s,l_g)
        
    def lnprior(self,pars,bounds):
        if bounds is None:
            return 0
        else:
            if all(pars>bounds[0]) and all(pars<bounds[1]):
                return 0.0
            else: 
                return -np.inf
        
    def lnlike(self,pars,t_s,t_g,flux_s,flux_s_err,flux_frac,flux_scatter,flux_g,flux_g_err,l_s,l_g):
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

class SingleLensModel(LCModel):
    
    def func(self,time,fb,t0,fs,tE):
        ts = (time-t0)/(tE/np.sqrt(12))
        flux = fb+fs/(np.sqrt(ts**2 +1))
        return flux