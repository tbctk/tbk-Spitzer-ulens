import numpy as np
import time as ti
    
def analytic_solution(time,flux,flux_err,flux_frac,pars,func,time_g=None):
    """Analytical solve for PLD coefficients.
    
    Args:
        time (list): Image timestamps in list of shape (n_dit,img_per_dit).
        flux (list): Raw photometry in list of shape (n_dit,img_per_dit).
        flux_err (list): Error on raw photometry in list of shape (n_dit,img_per_dit).
        flux_frac (list): Fractional flux in list of shape (n_dit,img_per_dit,img_size,img_size), e.g. (6,43,5,5) for ob171140.
        popt (list): Optimal parameters for curve fit.
        func (function): Function to fit data to.
        
    Returns:
        Y: List representing dithered raw flux vector.
        astro: List representing dithered astrophisical signal vector.
        Ps: List representing dithered fractional flux matrix.
        A: List representing dithered action matrix.
        C: List representing dithered covariance matrix.
        E: List representing dithered error estimate??
        X: List representing dithered PLD coefficients.
    """
    n_dit,n_data,size,_ = np.shape(flux_frac)
    n_reg = size*size

    Y = flux[:,:,None]
    if time_g is not None:
        flu,flu_g = func(time.ravel(),time_g,*pars)
        astro = flu.reshape(time.shape)
    else:
        astro = func(time.ravel(),*pars).reshape(time.shape)
    Ps = np.reshape(flux_frac,(n_dit,n_data,n_reg))
    A = astro[:,:,None]*Ps
    C = flux_err**2
    X = np.empty((n_dit,n_reg))
    E = np.empty((n_dit,n_reg,n_reg))

    for i in range(n_dit):
        Cinv = C[i]**-1
        tmp1 = np.linalg.pinv(A[i].T@np.diag(Cinv)@A[i])
        tmp2 = A[i].T@np.diag(Cinv)@Y[i]
        E[i] = tmp1
        X[i] = (tmp1@tmp2).ravel()
    if time_g is not None:
        return astro, Ps, A, X, flu_g
    else:
        return astro, Ps, A, C, E, X

def get_bestfit(X, flux, flux_frac, astro, A=None):
    """
    Getting the best-fit value from analytical solutions.
    
    Args:
        X (list of float): PLD coefficient matrix.
        flux (list of float): Raw flux vector.
        flux_frac (list of float): Fractional flux matrix.
        astro (list of float): Astrophysical signal vector.
        A (list of float): Action matrix. Will be calculated if not provided.
        
    Returns:
        fit: Expected raw flux.
        sys: Detector systematics.
        corr: Corrected data with detector systematics removed.
        resi: Residuals with respect to corrected data.
    """
    n_dit,n_data,size,_ = np.shape(flux_frac)
    n_reg = size*size
    
    Ps = np.reshape(flux_frac,(n_dit,n_data,n_reg))
    if A is None:
        A = astro[:,:,None]*Ps
    
    fit = (A@X[:,:,None])[:,:,0]   
    sys = (Ps@X[:,:,None])[:,:,0]
    corr = flux/sys
    resi = corr-astro
    return fit, sys, corr, resi

def get_RMS(residual,label='RMS',visual=False):
    """Given some residuals, returns the Root-Mean-Square.
    
    Args:
        RMS (1D array): Array of residuals
        label (string): Label for printing the RMS. Default is 'RMS'.
        
    Returns:
        Root-mean-square error as a float.
    """
    resi2 = residual**2
    N = len(residual)
    RMS = np.sqrt((1/N)*np.sum(resi2))
    if visual:
        print(label, ' : ', RMS)
    return RMS