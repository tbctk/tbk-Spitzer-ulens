import numpy as np
from . import models
from . import MuLensEvent
import emcee
import time as ti
from tqdm import tqdm
import os
from . import mcmc
    
def analytic_solution(TIMES,PTOT,PTOT_E,PNORM,popt,func):
    """As our model is a linear function, we can solve for the best-fit parameters analytically.
    
    Args:
        TIMES (list): Image timestamps in list of shape (n_dit,img_per_dit)
        PTOT (list): Raw photometry in list of shape (n_dit,img_per_dit)
        PTOT_E (list): Error on raw photometry in list of shape (n_dit,img_per_dit)
        PNORM (list): Fractional flux in list of shape (n_dit,img_per_dit,img_size,img_size), e.g. (6,43,5,5) for ob171140
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
    n_dit,n_data,size,_ = np.shape(PNORM)
    n_reg = size*size

    Y = PTOT[:,:,None]
    Astro = func(TIMES.ravel(),*popt).reshape(TIMES.shape)
    Ps = np.reshape(PNORM,(n_dit,n_data,n_reg))
    A = Astro[:,:,None]*Ps
    C = PTOT_E**2
    X = np.empty((n_dit,n_reg))
    E = np.empty((n_dit,n_reg,n_reg))

    for i in range(n_dit):
        Cinv = C[i]**-1
        tmp1 = np.linalg.pinv(A[i].T@np.diag(Cinv)@A[i])
        tmp2 = A[i].T@np.diag(Cinv)@Y[i]
        E[i] = tmp1
        X[i] = (tmp1@tmp2).ravel()
 
    return Y, Astro, Ps, A, C, E, X

def get_bestfit(A, Ps, X, PTOT, Astro):
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
    CORR = PTOT/SYS
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

def get_BIC(func,popt,TIMES,PTOT,PTOT_E,E_BIN,PNORM):
    ndit,_,sx,sy = PNORM.shape
    dudchain = mcmc.PLDCoeffsChain(np.empty(ndit*sx*sy))
    lnlike = mcmc.lnlike(popt_mcmc, func, TIMES, PTOT, PTOT_E, E_BIN, PNORM, dudchain)
    return popt_mcmc.size*np.log(ptot.size)-lnlike