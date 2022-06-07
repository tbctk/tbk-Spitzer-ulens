import numpy as np
from . import models
from . import MuLensEvent
import emcee
import time as ti
from tqdm import tqdm
import os
from . import mcmc
    
def analytic_solution(t,raw_flux,raw_err,frac_flux,popt,func):
    """As our model is a linear function, we can solve for the best-fit parameters analytically.
    
    Args:
        t (list): Image timestamps in list of shape (n_dit,img_per_dit)
        raw_flux (list): Raw photometry in list of shape (n_dit,img_per_dit)
        raw_err (list): Error on raw photometry in list of shape (n_dit,img_per_dit)
        frac_flux (list): Fractional flux in list of shape (n_dit,img_per_dit,img_size,img_size), e.g. (6,43,5,5) for ob171140
        popt (list): Optimal parameters for curve fit
        func (function): Function to fit data to
        
    Returns:
        Y (list):
        Astro (list):
        Ps (list):
        A (list):
        C (list):
        E (list):
        X (list):
    """
    n_dit,n_data,size,_ = np.shape(frac_flux)
    n_reg = size*size

    Y = raw_flux[:,:,None]
    Astro = func(t.ravel(),*popt).reshape(t.shape)
    Ps = np.reshape(frac_flux,(n_dit,n_data,n_reg))
    A = Astro[:,:,None]*Ps
    C = raw_err**2
    X = np.empty((n_dit,n_reg))
    E = np.empty((n_dit,n_reg,n_reg))

    for i in range(n_dit):
        Cinv = C[i]**-1
        tmp1 = np.linalg.pinv(A[i].T@np.diag(Cinv)@A[i])
        tmp2 = A[i].T@np.diag(Cinv)@Y[i]
        E[i] = tmp1
        X[i] = (tmp1@tmp2).ravel()
 
    return Y, Astro, Ps, A, C, E, X

def get_bestfit(A, Ps, X, raw_flux, Astro):
    """
    Getting the best-fit value from analytical solutions.
    
    Args:
        Y (list):
        Astro (list):
        Ps (list):
        A (list):
        C (list):
        E (list):
        X (list):
        
    Returns:
        FIT (list):
        SYS (list):
        CORR (list):
        RESI (list):
    """
    FIT = (A@X[:,:,None])[:,:,0]   
    SYS = (Ps@X[:,:,None])[:,:,0]
    CORR = raw_flux/SYS
    RESI = CORR-Astro
    
    return FIT, SYS, CORR, RESI

def get_RMS(residual,label='RMS',visual=False):
    """Given some residuals, returns the Root-Mean-Square.
    
    Args:
        RMS (1D array): Array of residuals
        label (string): Label for printing the RMS. Default is 'RMS'.
        
    Returns:
        float : Root-mean-square. 
    """
    resi2 = residual**2
    N = len(residual)
    RMS = np.sqrt((1/N)*np.sum(resi2))
    if visual:
        print(label, ' : ', RMS)
    return RMS

def get_BIC(func,popt,t,raw_flux,raw_err,E_BIN,frac_flux):
    ndit,_,sx,sy = frac_flux.shape
    dudchain = mcmc.PLDCoeffsChain(np.empty(ndit*sx*sy))
    lnlike = mcmc.lnlike(popt_mcmc, func, t, raw_flux, raw_err, E_BIN, frac_flux, dudchain)
    return popt_mcmc.size*np.log(ptot.size)-lnlike