import numpy as np
from inspect import signature, BoundArguments
from abc import ABC,abstractmethod
import MulensModel as mm

class LCModel(ABC):

    def __call__(self,*pars,**kwpars):
        return self.func(*pars,**kwpars)
    
    @abstractmethod
    def func(self,*pars,**kwpars):
        pass
    
    @staticmethod
    def mag2flux(mag,fb,fs):
        return mag*fs+fb
    
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

class SingleLensModel(LCModel):
    
    def func(self,time,fb,t0,fs,tE):
        ts = (time-t0)/(tE/np.sqrt(12))
        flux = fb+fs/(np.sqrt(ts**2 +1))
        return flux