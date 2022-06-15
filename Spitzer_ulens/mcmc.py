import emcee
import numpy as np
from . import PLD
import time as ti
from tqdm import tqdm
import os

def get_MCMC_sampler(p0,modelfunc,TIMES,PTOT,PTOT_E,E_BIN,PNORM,PLD_chain,pool=None,nwalkers=100,bounds=None):
    """
    Produces an emcee MCMC sampler that uses the PLD log-likelihood function.
    
    Args:
        p0 (list of float): Initial parameter guess.
        modelfunc (function): Model function. Must be able to be called via modelfunc(time,*pars).
        TIMES (list of float): Dithered time data from which to evaluate model fit.
        PTOT (list of float): Dithered raw photometric data.
        PTOT_E (list of float): Raw photometric error.
        PNORM (list of float): Dithered fractional flux.
        PLD_Chain (PLDCoeffsChain object): PLD coefficient chain to store PLD coefficients
        pool (multiprocessing.Pool object, optional): Multiprocessing pool, see documentation (TODO)
        nwalkers (int): Number of MCMC walkers to employ, defaults to 100.
        bounds (list, optional): Bounds on model parameters to impose upon MCMC walkers. Must be of shape (n_par,n_par) where the first element is the lower bound, second is upper bound.
        
    Returns:
        sampler (emcee.EnsembleSampler object): MCMC sampler that uses the PLD log-likelihood function.
    """
    ndim = len(p0)
    # sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, a = 2, pool=pool,
                                    args=(modelfunc,TIMES, PTOT, PTOT_E, E_BIN, PNORM, PLD_chain, bounds))
    return sampler

# FIXED fb, 3 MCMC params

def lnprior(p0,bounds):
    """
    Constrain parameters based on prior known bounds. Currently just a basic step function that disallows paramters outside bounds.
    
    Args:
        p0 (list of float): Parameter values for this MCMC iteration.
        bounds (list or None): If None, no bounds are imposed. Otherwise, must be of shape (n_par,n_par) where the first element is the lower bound, second is upper bound.
        
    Returns:
        0 if the parameter values are within the bounds, negative infinity otherwise. This ensures that if the parameters are unphysical, the likelihood will go to negative infinity and the iteration will be discarded.
    """
    if bounds is None:
        return 0
    else:
        if all(p0>bounds[0]) and all(p0<bounds[1]):
            return 0.0
        else: return -np.inf

def lnlike(p0, func, TIMES, PTOT, PTOT_E, E_BIN, PNORM, PLD_chain):
    """
    Computes the log-likelihood for model to fit the given data. Updates the PLD coefficient chain accordingly.
    
    Args:
        p0 (list of float): Initial parameter guess.
        func (function): Model function. Must be able to be called via func(time,*pars).
        TIMES (list of float): Dithered time data from which to evaluate model fit.
        PTOT (list of float): Dithered raw photometric data.
        PTOT_E (list of float): Raw photometric error.
        E_BIN (float): Estimate of raw photometric scatter.
        PNORM (list of float): Dithered fractional flux.
        PLD_Chain (PLDCoeffsChain object): PLD coefficient chain to store PLD coefficients.
    
    Returns:
        like (float): log-likelihood for the model to fit the given data.
    """
    # solving for PLD coefficients analytically
    Y, Astro, Ps, A, C, E, X = PLD.analytic_solution(TIMES, PTOT, PTOT_E, PNORM, p0, func)
    
    # saving PLD coeff
    PLD_chain.update_chain(np.array(X).ravel())

    # Generating time series from bestfit params
    FIT, SYS, CORR, RESI = PLD.get_bestfit(A, Ps, X, PTOT, Astro)
    Ndat = len(PTOT[0])
    
    like = 0
    # generating model and calculating the difference with the flux
    for i in range(len(TIMES)):
        # diff (don't forget ravel, otherwise you'll have some matrix operation!)
        diff  = PTOT[i]-FIT[i].ravel()
        diff2 = PTOT[i]-Astro[i].ravel()
        # likelihood = -0.5*chisq
        inerr = 1/PTOT_E[i]
        inerr2 = 1/E_BIN
        like  += -0.5*np.sum(diff**2*inerr**2) + np.sum(np.log(inerr)) - Ndat*0.9189385332046727
        like  += -0.5*np.sum(diff2**2*inerr2**2) + Ndat*np.log(inerr2) - Ndat*0.9189385332046727
    return like
    
def lnprob(p0, func, TIMES, PTOT, PTOT_E, E_BIN, PNORM, PLD_chain, bounds):
    """
    Computes the log-likelihood for model to fit the given data, constrained by the provided bounds. Updates the PLD coefficient chain accordingly.
    
    Args:
        p0 (list of float): Initial parameter guess.
        func (function): Model function. Must be able to be called via func(time,*pars).
        TIMES (list of float): Dithered time data from which to evaluate model fit.
        PTOT (list of float): Dithered raw photometric data.
        PTOT_E (list of float): Raw photometric error.
        E_BIN (float): Estimate of raw photometric scatter.
        PNORM (list of float): Dithered fractional flux.
        PLD_Chain (PLDCoeffsChain object): PLD coefficient chain to store PLD coefficients.
        bounds (list or None): If None, no bounds are imposed. Otherwise, must be of shape (n_par,n_par) where the first element is the lower bound, second is upper bound.
    
    Returns:
        Float representing log-likelihood for the model to fit the given data, or negative infinity if the parameters fall outside the bounds.
    """
    # get lnprior
    lp = lnprior(p0,bounds)
    # if guess is out of bound
    if not np.isfinite(lp):
        return -np.inf
    # calculate posterior
    return lp + lnlike(p0, func, TIMES, PTOT, PTOT_E, E_BIN, PNORM, PLD_chain)

def run_mcmc(sampler,pos0,nsteps,visual=True,label=''):
    """
    Runs MCMC using the given sampler and starting positions.
    
    Args:
        sampler (emcee.EnsembleSampler object): emcee MCMC sampler as returned by the get_MCMC_sampler function.
        pos0 (list of float): Initial parameter positions for each walker.
        nsteps (int): Number of MCMC steps to take.
        visual (bool): If true, displays some runtime information along with a tqdm progress bar.
        label (str, optional): Label to distinguish this MCMC run. Will be displated if visual is set to `True`.
        
    Returns:
        pos (list of int): Final position of the MCMC walkers.
        prob (list of int): Log-probabilities for positions pos.
        state ():
    """
    if visual:
        tic = ti.time()
        print('Running MCMC '+label+'...')
        
        for pos, prob, state in tqdm(sampler.sample(pos0, iterations=nsteps),total=nsteps):
            pass

        print("Mean burn-in acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
        toc = ti.time()
        print('MCMC runtime = %.2f min\n' % ((toc-tic)/60.))
        return pos,prob,state
    else:
        pos,prob,state = sampler.run_mcmc(pos0,nsteps)
        return pos,prob,state

def save_results(evt,chain,posit,lnprob,PLD_coeffs=None,folder=''):
    """
    Save MCMC chains.
    
    Args:
        evt (str): Abridged event name.
        chain (list of float): Full walker position chain.
        posit (list of float): Final walker positions.
        lnprob (list of float): Walker log-probability chain.
        PLD_coeffs (PLDCoeffsChain object, optional): PLD coefficient chain to be saved.
        folder (str, optional): Extra sorting option to save the results in a mega_MCMC sub-folder.
    """
    #Saving MCMC Results
    savepath = 'data/'+ evt + '/mega_MCMC/'
    savepath = os.path.join(savepath,folder)
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    # path + name for saving important MCMC info
    pathchain = os.path.join(savepath,'samplerchain.npy')
    pathposit = os.path.join(savepath,'samplerposit.npy')
    pathlnpro = os.path.join(savepath,'samplerlnpro.npy')

    # chain of all walkers during the last production steps (nwalkers, nsteps, ndim)
    np.save(pathchain, chain)
    # position of all 100 walkers (nwalkers, ndim)
    np.save(pathposit, posit)
    # lnprob for all position of the walkers (nwalkers, nsteps)
    np.save(pathlnpro, lnprob)
    # save PLD coefficients too 
    if not PLD_coeffs is None:
        pathPLDco = os.path.join(savepath,'PLD_chain.npy')
        np.save(pathPLDco, PLD_coeffs)
    return

def get_MCMC_results(chain,lnprob):
    """
    Get optimal parameters and errors from the MCMC run.
    
    Args:
        chain (list of float): Full walker position chain.
        lnprob (list of float): Walker log-probability chain.
    
    Returns:
        popt (list of float): Mean parameter positions.
        pmax (list of float): Most likely parameter positions.
        std_hi (list of float): Upper standard deviation on the best-fit parameter positions.
        std_hi (list of float): Lower standard deviation on the best-fit parameter positions.
    """
    _,_,npars = chain.shape
    posit = chain.reshape(-1,npars)

    # Get the percentile
    percs = np.percentile(posit, [16, 50, 84],axis=0)
    (MCMC_Results) = np.array(list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*percs))))

    popt = MCMC_Results[:,0]
    std_hi = MCMC_Results[:,1]
    std_lo = MCMC_Results[:,2]

    # Get most probable params
    probs = lnprob.flatten()
    pmax = posit[np.argmax(probs)]
    
    return popt,pmax,std_hi,std_lo

def get_BIC(popt, func, TIMES, PTOT, PTOT_E, E_BIN, PNORM, X):
    """
    Compute the BIC (Bayesian Information Criterion) for the given parameters, using our log-likehihood function and data.
    
    Args:
        popt (list of float): Parameters to compute BIC for.
        func (function): Model function, must be called as per func(times,*popt).
        TIMES (list of float): Dithered time data from which to evaluate model fit.
        PTOT (list of float): Dithered raw photometric data.
        PTOT_E (list of float): Raw photometric error.
        E_BIN (float): Estimate of raw photometric scatter.
        PNORM (list of float): Dithered fractional flux.
        X (list of float): PLD Coefficients or empty array of same size (TODO: REMOVE THIS)
    
    Returns:
        A float representing the BIC for this data, model, and model parameters. The BIC provides a likelihood that is corrected to compensate for complex-model overfitting.
    """
    dudchain = PLDCoeffsChain(np.zeros(np.size(X)))
    ll = lnlike(popt, func, TIMES, PTOT, PTOT_E, E_BIN, PNORM, dudchain)
    return popt_mcmc.size*np.log(PTOT.size)-ll