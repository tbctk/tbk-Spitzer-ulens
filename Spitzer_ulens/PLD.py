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
        Y
        Astro
        Ps
        A 
        C
        E
        X
    """
    n_dit,img_per_dit,size,_ = np.shape(PNORM)
    n_data = img_per_dit
    n_reg = size*size

    Y = []        # Data
    Astro = []    # Astrophysical signal
    Ps = []       # Avg flux
    A = []        # Independent variables
    C = []        # Data Covariance Matrix
    X = []        # Best Fit Parameters
    E = []        # Parameter Covariance Matrix (useful to get error on the best-fit parameters)

    for i in range(n_dit):
        Y_i = np.reshape(PTOT[i],(n_data,1))
        Astro_i = np.reshape(func(TIMES[i],*popt),(n_data,1))
        Ps_i = np.reshape(PNORM[i],(n_data,n_reg))
        A_i = Ps_i*Astro_i
        C_i = np.diag(PTOT_E[i]**2)
        Cinv = np.linalg.pinv(C_i)

        tmp1 = np.linalg.pinv(A_i.T@Cinv@A_i)
        tmp2 = A_i.T@Cinv@Y_i
        X_i = tmp1@tmp2

        # appending
        Y.append(Y_i)
        Astro.append(Astro_i[:,0])
        Ps.append(Ps_i)
        A.append(A_i)
        C.append(C_i)
        E.append(tmp1)
        X.append(X_i)

    return Y, Astro, Ps, A, C, E, X

def get_bestfit(A, Ps, X, PTOT, Astro):
    """
    Getting the best-fit value from analytical solutions.
        
    :param A: Independent variable matrix, defined by A[i,j] = Astro[i]*Ps[i,j]
    """
    FIT = []    
    SYS = []
    CORR = []
    RESI = []

    nb_dithers = len(PTOT)

    for i in range(nb_dithers):
        FIT_tmp = np.matmul(A[i], X[i])[:,0]
        SYS_tmp = np.matmul(Ps[i], X[i])[:,0]
        FIT.append(FIT_tmp)
        SYS.append(SYS_tmp)
        CORR.append(PTOT[i]/SYS_tmp)
        RESI.append(PTOT[i]/SYS_tmp - Astro[i])

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