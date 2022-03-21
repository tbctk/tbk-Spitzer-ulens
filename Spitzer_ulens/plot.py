import numpy as np
import matplotlib.pyplot as plt

def plot_guess(time, ptot, timeplot, lcguess, lcoptim=None, guessing=True):

    fig = plt.figure(figsize = (8,4))
    _ = plt.plot(time, ptot, '.', label='Spitzer Observations', alpha=0.5)
    _ = plt.plot(timeplot, lcguess, '--', label='Initial Guess', alpha=0.4)
    if guessing == False:
        _ = plt.plot(timeplot, lcoptim, label='Optimzed', color='C1', linewidth=3)

    _ = plt.xlabel('Time (JD)', fontsize=15)
    _ = plt.ylabel('MJy/str', fontsize=15)
    _ = plt.xlim(np.min(timeplot), np.max(timeplot))
    _ = plt.legend()

    return fig

def plot_analytic_solution(time, ptot, fit, corr, resi, timeplot, lcoptim):
    # ylim for ptot 
    pmin, pmax = 0, np.max(ptot)+5

    # ylim for residuals
    resi_std   = np.std(resi)
    rmin, rmax = -4*resi_std, 4*resi_std

    fig, axes = plt.subplots(nrows = 3, ncols = 1, sharex = True, figsize = (8, 9))

    axes[0].plot(time, ptot, '.', label = 'Data')
    axes[0].plot(time, fit, '.', label= 'Signal Fit', color='C1', alpha=0.5)
    axes[0].set_ylabel('Photometry (MJy/str)', fontsize=13)
    axes[0].set_ylim(pmin, pmax)
    axes[0].legend()

    axes[1].plot(time, corr, '.', label='Corrected Data', alpha=0.5)
    axes[1].plot(timeplot, lcoptim, label='Astrophysical Model', color='C1', linewidth=3,alpha=0.5)
    axes[1].set_ylabel('Photometry (MJy/str)', fontsize=13)
    axes[1].set_ylim(pmin, pmax)
    axes[1].legend()

    axes[2].plot(time, resi, '.', label='Residuals', alpha=0.5)
    axes[2].axhline(y=0, color='C1', linewidth=3,alpha=0.5)
    axes[2].set_ylabel('Residuals (MJy/str)', fontsize=13)
    axes[2].set_xlabel('Time (JD)', fontsize=13)
    axes[2].set_ylim(rmin, rmax)
    axes[2].legend()

    axes[2].set_xlim(np.min(timeplot), np.max(timeplot))

    fig.subplots_adjust(hspace = 0)

    return fig, axes

def plot_caustics(y1, y2, caustic, critical):
    # getting delimitation of the plot
    ymin = np.min([np.min(y1),np.min(y2)])
    ymax = np.max([np.max(y1),np.max(y2)])
    # plotting
    fig = plt.figure(figsize=(5,5))
    _ = plt.ylim(ymin-0.05,ymax+0.05)
    _ = plt.xlim(ymin-0.05,ymax+0.05)
    _ = plt.title('Lens Plane')
    _ = plt.plot(0.0,0.0,'o',label='Center of Mass', color='C8', markersize=10)
    _ = plt.plot(y1, y2, 'k--', alpha = 0.5, label='trajectory of the source')
    _ = plt.plot(caustic[:, 0], caustic[:, 1], '.', label = 'Caustic', markersize =1)
    _ = plt.plot(critical[:,0], critical[:,1], 'k.', label = 'Critical Curve', markersize=1)
    _ = plt.plot(y1[0],y2[0],'o', color='C1', label='Start of observations')
    _ = plt.plot(y1[-1],y2[-1],'o', color='C2', label='End of observations')
    _ = plt.legend(loc=2, bbox_to_anchor=(1.1, 0.7))
    _ = plt.xlabel('Distance (Einstein Radius)')
    _ = plt.ylabel('Distance (Einstein Radius)')
    return fig



