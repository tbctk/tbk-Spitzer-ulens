import numpy as np
from . import models
from . import MuLensEvent
import emcee
import time as ti
from tqdm import tqdm
import os

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

    for i in range(n_dit):
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


class PLDCoeffsChain:
    def __init__(self,coeffs):
        self.chain = [np.asarray(coeffs)]
    
    def update_chain(self,coeffs):
        self.chain = np.concatenate((self.chain,[np.asarray(coeffs)]))

def get_MCMC_sampler(p0,modelfunc,TIMES,PTOT,PTOT_E,PNORM,PLD_chain,pool=None):
    # initializing walkers
    ndim, nwalkers = len(p0)-1, 100
    # sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, a = 2, pool=pool,
                                    args=(p0[0],models.single_lens,TIMES, PTOT, PTOT_E, PNORM, PLD_chain))
    
    return sampler

# FIXED fb, 3 MCMC params

def lnprior(p0_astro):
    """TO BE GENERALIZED"""
    t0, fs, tE = p0_astro
    if (t0>0.0) and (tE>0.0) and (fs>0.0):
        return 0.0
    else:
        return -np.inf

def lnlike(p0_astro, fb, func, TIMES, PTOT, PTOT_E, PNORM, PLD_chain):
    """
    Args:
    Return:
    """
    p0 = np.concatenate(([fb],p0_astro))

    # solving for PLD coefficients analytically
    Y, Astro, Ps, A, C, E, X = analytic_solution(TIMES, PTOT, PTOT_E, PNORM, p0, func)
    
    # saving PLD coeff
    PLD_chain.update_chain(np.array(X).ravel())

    # Generating time series from bestfit params
    FIT, SYS, CORR, RESI = get_bestfit(A, Ps, X, PTOT, Astro)
    
    like = 0
    # generating model and calculating the difference with the flux
    for i in range(len(TIMES)):
        # diff (don't forget ravel, otherwise you'll have some matrix operation!)
        diff  = PTOT[i]-FIT[i].ravel()
        # likelihood = -0.5*chisq
        inerr = 1/PTOT_E[i]
        like  += -0.5*np.sum(diff**2*inerr**2)
    return like
    
def lnprob(p0_astro, fb, func, TIMES, PTOT, PTOT_E, PNORM, PLD_chain):
    # get lnprior
    lp = lnprior(p0_astro)
    # if guess is out of bound
    if not np.isfinite(lp):
        return -np.inf
    # calculate posterior
    return lp + lnlike(p0_astro, fb, func, TIMES, PTOT, PTOT_E, PNORM, PLD_chain)

def run_MCMC_analytic(sampler,pos0,visual=True):
    #First burn-in:
    tic = ti.time()
    print('Running burn-in')
    niter=300
    for pos1, prob, state in tqdm(sampler.sample(pos0, iterations=niter),total=niter):
        #print(np.mean(pos1,axis=0))
        pass

    print("Mean burn-in acceptance fraction: {0:.3f}"
                        .format(np.mean(sampler.acceptance_fraction)))
    sampler.reset()
    toc = ti.time()
    print('MCMC runtime = %.2f min\n' % ((toc-tic)/60.))
    #Second burn-in
    #Continue from best spot from last time, and do quick burn-in to get walkers spread out
    tic = ti.time()
    print('Running second burn-in')
    pos2 = pos1[np.argmax(prob)]
    # slightly change position of walkers to prevent them from taking the same path
    pos2 = [pos2*(1+1e-6*np.random.randn(sampler.ndim))+1e-6*np.abs(np.random.randn(sampler.ndim)) for i in range(sampler.nwalkers)]
    niter = 300
    for pos2, prob, state in tqdm(sampler.sample(pos2, iterations=niter),total=niter):
        pass
    print('Mean burn-in acceptance fraction: {0:.3f}'
                        .format(np.median(sampler.acceptance_fraction)))
    sampler.reset()
    toc = ti.time()
    print('MCMC runtime = %.2f min\n' % ((toc-tic)/60.))
    #Run production
    #Run that will be saved
    tic = ti.time()
    # Continue from last positions and run production
    print('Running production')
    niter = 1000
    for pos3, prob, state in tqdm(sampler.sample(pos2, iterations=niter),total=niter):
        pass
    print("Mean acceptance fraction: {0:.3f}"
                        .format(np.mean(sampler.acceptance_fraction)))
    toc = ti.time()
    print('MCMC runtime = %.2f min\n' % ((toc-tic)/60.))
    return sampler.chain,pos3,sampler.lnprobability

# All 154 MCMC params

def lnprior2(p0_astro):
    """TO BE GENERALIZED"""
    fb, t0, fs, tE = p0_astro
    if (t0>0.0) and (tE>0.0) and (0.0<fs<1000) and (0.0<fb<60):
        return 0.0
    else:
        return -np.inf

def lnlike2(p0_astro, p0_PLD, func, TIMES, PTOT, PTOT_E, PNORM):
    """
    Args:
    Return:
    """
    like = 0
    # generating model and calculating the difference with the flux
    for i in range(len(TIMES)):
        astro = func(TIMES[i], *p0_astro)
        detec = np.sum(np.sum(p0_PLD[i,:,:]*PNORM[i], axis=2), axis=1)
        # diff
        model = astro*detec
        diff  = PTOT[i]-model
        # likelihood
        inerr = 1/PTOT_E[i]
        like  += -0.5*np.sum(diff**2*inerr**2)
    return like
    
def lnprob2(p0, func, TIMES, PTOT, PTOT_E, PNORM):
    # unpack will happen here, so that it will be done only once
    nb_dithers = len(TIMES)
    nb_PLDcoef = nb_dithers*25
    # get PLD astro
    p0_astro = p0[:-nb_PLDcoef]
    p0_PLD   = p0[-nb_PLDcoef:]
    p0_PLD   = p0_PLD.reshape(nb_dithers,5,5)
    # get lnprior
    lp = lnprior2(p0_astro)
    # if guess is out of bound
    if not np.isfinite(lp):
        return -np.inf
    # calculate posterior
    return lp + lnlike2(p0_astro, p0_PLD, func, TIMES, PTOT, PTOT_E, PNORM)

def run_MCMC_over(sampler,pos0,visual=True):
    #First burn-in:
    tic = ti.time()
    print('Running burn-in')
    niter=300
    for pos1, prob, state in tqdm(sampler.sample(pos0, iterations=niter),total=niter):
        #print(np.mean(pos1,axis=0))
        pass

    print("Mean burn-in acceptance fraction: {0:.3f}"
                        .format(np.mean(sampler.acceptance_fraction)))
    sampler.reset()
    toc = ti.time()
    print('MCMC runtime = %.2f min\n' % ((toc-tic)/60.))
    #Second burn-in
    #Continue from best spot from last time, and do quick burn-in to get walkers spread out
    tic = ti.time()
    print('Running second burn-in')
    pos2 = pos1[np.argmax(prob)]
    # slightly change position of walkers to prevent them from taking the same path
    pos2 = [pos2*(1+1e-6*np.random.randn(sampler.ndim))+1e-6*np.abs(np.random.randn(sampler.ndim)) for i in range(sampler.nwalkers)]
    niter = 300
    for pos2, prob, state in tqdm(sampler.sample(pos2, iterations=niter),total=niter):
        pass
    print('Mean burn-in acceptance fraction: {0:.3f}'
                        .format(np.median(sampler.acceptance_fraction)))
    sampler.reset()
    toc = ti.time()
    print('MCMC runtime = %.2f min\n' % ((toc-tic)/60.))
    #Run production
    #Run that will be saved
    tic = ti.time()
    # Continue from last positions and run production
    print('Running production')
    niter = 1000
    for pos3, prob, state in tqdm(sampler.sample(pos2, iterations=niter),total=niter):
        pass
    print("Mean acceptance fraction: {0:.3f}"
                        .format(np.mean(sampler.acceptance_fraction)))
    toc = ti.time()
    print('MCMC runtime = %.2f min\n' % ((toc-tic)/60.))
    return sampler.chain,pos3,sampler.lnprobability

def save_chain(evt,chain,posit,lnprob,PLD_coeffs=None):
    #Saving MCMC Results
    savepath  = 'data/'+ evt + '/mega_MCMC/'
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    # path + name for saving important MCMC info
    pathchain = savepath + 'samplerchain.npy'
    pathposit = savepath + 'samplerposit.npy'
    pathlnpro = savepath + 'samplerlnpro.npy'

    # chain of all walkers during the last production steps (nwalkers, nsteps, ndim)
    np.save(pathchain, chain)
    # position of all 100 walkers (nwalkers, ndim)
    np.save(pathposit, posit)
    # lnprob for all position of the walkers (nwalkers, nsteps)
    np.save(pathlnpro, lnprob)
    # save PLD coefficients too 
    if not PLD_coeffs is None:
        pathPLDco = savepath + 'PLD_chain.npy'
        np.save(pathPLDco, PLD_coeffs)
    return