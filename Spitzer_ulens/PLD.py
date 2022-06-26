import numpy as np
import time as ti
    
def analytic_solution(time,flux,flux_err,flux_frac,pars,func,time_g=None):
    """As our model is a linear function, we can solve for the best-fit parameters analytically.
    
    Args:
        time (list): Image timestamps in list of shape (n_dit,img_per_dit)
        flux (list): Raw photometry in list of shape (n_dit,img_per_dit)
        flux_err (list): Error on raw photometry in list of shape (n_dit,img_per_dit)
        flux_frac (list): Fractional flux in list of shape (n_dit,img_per_dit,img_size,img_size), e.g. (6,43,5,5) for ob171140
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
    n_dit,n_data,size,_ = np.shape(flux_frac)
    n_reg = size*size

    Y = flux[:,:,None]
    if time_g is not None:
        flu,flu_g = func(time.ravel(),time_g,*pars)
        Astro = flu.reshape(time.shape)
    else:
        Astro = func(time.ravel(),*pars).reshape(time.shape)
    Ps = np.reshape(flux_frac,(n_dit,n_data,n_reg))
    A = Astro[:,:,None]*Ps
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
        return Y, Astro, Ps, A, X, flu_g
    else:
        return Y, Astro, Ps, A, C, E, X

def get_bestfit(A, Ps, X, flux, Astro):
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
    CORR = flux/SYS
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

#def get_BIC(func,popt,time,flux,flux_err,flux_scatter,flux_frac):
#    ndit,_,sx,sy = flux_frac.shape
#    dudchain = mcmc.PLDCoeffsChain(np.empty(ndit*sx*sy))
#    lnlike = mcmc.lnlike(popt_mcmc, func, time, flux, flux_err, flux_scatter, flux_frac, dudchain)
#    return popt_mcmc.size*np.log(ptot.size)-lnlike