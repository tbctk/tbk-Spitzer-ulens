import numpy as np
from . import models
from . import MuLensEvent
import emcee

def invert_matrix(mat):
    """Inverting matrix. If the matrix is not invertible (because it has a determinant of 0), try np.linalg.pinv which computes the (Moore-Penrose) pseudo-inverse of a matrix. Calculate the generalized inverse of a matrix using its singular-value decomposition (SVD) and including all large singular values.
    Args:
        mat (2D array): matrix to be inverted
    Returns:
        2D array: Inverse of mat, or pseudomatrix if mat is singular.
    """
    inv = np.linalg.inv(mat)
    if False:
        try:
            inv  = np.linalg.inv(mat)
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                print('Error:', err)
                print('Fix: Calculating pseudo-inverse of a matrix instead')
                inv  = np.linalg.pinv(mat)
    return inv

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

    for i in range(nb_dit):
        Y_i = np.reshape(PTOT[i],(n_data,1))
        Astro_i = np.reshape(func(TIMES[i],*popt),(n_data,1))
        Ps_i = np.reshape(PNORM[i],(n_data,n_reg))
        A_i = Ps_i*Astro_i
        C_i = np.diag(PTOT_E[i]**2)
        Cinv = invert_matrix(C_i)

        tmp1 = invert_matrix(A_i.T@Cinv@A_i)
        tmp2 = A_i.T@Cinv@Y_i
        X_i = tmp1@tmp2

        # appending
        Y.append(Y_i)
        Astro.append(Astro_i)
        Ps.append(Ps_i)
        A.append(A_i)
        C.append(C_i)
        E.append(tmp1)
        X.append(X_i)

    return Y, Astro, Ps, A, C, E, X

def get_bestfit(A, Ps, X, PTOT, Astro):
    """Getting the best-fit value from analytical solutions.
    """
    FIT = []    
    SYS = []
    CORR = []
    RESI = []

    nb_dithers = len(PTOT)

    for i in range(nb_dithers):
        FIT_tmp = np.matmul(A[i], X[i])
        SYS_tmp = np.matmul(Ps[i], X[i])
        FIT.append(FIT_tmp)
        SYS.append(SYS_tmp)
        CORR.append(PTOT[i]/SYS_tmp.ravel())
        RESI.append(PTOT[i]/SYS_tmp.ravel() - Astro[i].ravel())

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


def get_MCMC_sampler(p0,modelfunc,TIMES,PTOT,PTOT_E,PNORM,PLD_coeffs):
    # initializing walkers
    ndim, nwalkers = len(p0), 100

    # sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, a = 2,
                                    args=(models.single_lens,TIMES, PTOT, PTOT_E, PNORM, [PLD_coeffs]))
    
    return sampler

    
def lnprior(p0_astro):
    """TO BE GENERALIZED"""
    fb, t0, fs, tE = p0_astro
    if (t0>0.0) and (tE>0.0) and (fs>0.0) and (fb>0.0):
        return 0.0
    else:
        return -np.inf

def lnlike(p0_astro, func, TIMES, PTOT, PTOT_E, PNORM, PLD_coeffs):
    """
    Args:
    Return:
    """
    
    # solving for PLD coefficients analytically
    Y, Astro, Ps, A, C, E, X = analytic_solution(TIMES, PTOT, PTOT_E, PNORM, p0_astro, func)
    
    # saving PLD coeff
    PLD_coeffs[0] = np.concatenate((PLD_coeffs[0],(np.array(X).ravel()).reshape((1,150))))

    # Generating time series from bestfit params
    FIT, SYS, CORR, RESI = get_bestfit(A, Ps, X, PTOT, Astro)
    
    like = 0
    # generating model and calculating the difference with the flux
    for i in range(len(TIMES)):
        # diff (don't forget ravel, otherwise you'll have some matrix operation!)
        diff  = PTOT[i].ravel()-FIT[i].ravel()
        # likelihood
        inerr = 1/PTOT_E[i]
        like  += -0.5*np.sum(diff**2*inerr**2)
    return like
    
def lnprob(p0_astro, func, TIMES, PTOT, PTOT_E, PNORM, PLD_coeffs):
    # get lnprior
    lp = lnprior(p0_astro)
    # if guess is out of bound
    if not np.isfinite(lp):
        return -np.inf
    # calculate posterior
    return lp + lnlike(p0_astro, func, TIMES, PTOT, PTOT_E, PNORM, PLD_coeffs)