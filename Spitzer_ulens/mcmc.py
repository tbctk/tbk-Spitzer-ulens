import emcee
import numpy as np
from . import PLD
from . import models
import time as ti
from tqdm import tqdm
import os

class PLDCoeffsChain:
    def __init__(self,coeffs):
        self.chain = [np.asarray(coeffs)]
    
    def update_chain(self,coeffs):
        self.chain = np.concatenate((self.chain,[np.asarray(coeffs)]))

def get_MCMC_sampler(p0,modelfunc,TIMES,PTOT,PTOT_E,E_BIN,PNORM,PLD_chain,pool=None,nwalkers=100,bounds=None):
    ndim = len(p0)
    # sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, a = 2, pool=pool,
                                    args=(modelfunc,TIMES, PTOT, PTOT_E, E_BIN, PNORM, PLD_chain, bounds))
    return sampler

# FIXED fb, 3 MCMC params

def lnprior(p0,bounds):
    """
    Constrain parameters based on prior known bounds. Currently a step function.    
    """
    if bounds is None:
        return 0
    else:
        if all(p0>bounds[0]) and all(p0<bounds[1]):
            return 0.0
        else: return -np.inf

def lnlike(p0, func, TIMES, PTOT, PTOT_E, E_BIN, PNORM, PLD_chain):
    """
    Args:
    Return:
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
    # get lnprior
    lp = lnprior(p0,bounds)
    # if guess is out of bound
    if not np.isfinite(lp):
        return -np.inf
    # calculate posterior
    return lp + lnlike(p0, func, TIMES, PTOT, PTOT_E, E_BIN, PNORM, PLD_chain)

def run_MCMC(sampler,pos0,visual=True,nburnin=300,nprod=1000):
    #First burn-in:
    tic = ti.time()
    print('Running burn-in')
    for pos1, prob, state in tqdm(sampler.sample(pos0, iterations=nburnin),total=nburnin):
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
    for pos2, prob, state in tqdm(sampler.sample(pos2, iterations=nburnin),total=nburnin):
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
    for pos3, prob, state in tqdm(sampler.sample(pos2, iterations=nprod),total=nprod):
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