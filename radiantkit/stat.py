'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import convolve
import scipy.optimize
import scipy.stats
import warnings

def gpartial(V: np.ndarray, d: int, sigma: float) -> np.ndarray:
    '''Calculate the partial derivative of V along dimension d using a filter
    of size sigma. Based on code by Erik Wernersson, PhD.'''

    w = round(8 * sigma + 2)
    if 0 == w % 2:
        w = w + 1
    w = 2 * w + 1
    if 1 == sigma:
        w = 11;

    if sigma == 0:
        dg = [0, -1, 1];
        g = [0, .5, .5];
    else:
        g = stats.norm.pdf(np.linspace(-w/2., w/2., w+1), scale = sigma)
        x = np.linspace(-(w - 1) / 2, (w - 1) / 2, w + 1)
        k0 = 1 / np.sqrt(2 * np.pi * sigma**2.)
        k1 = 1 / (2 * sigma**2)
        dg = -2 * k0 * k1 * x * np.exp(-k1 * x**2.)

    if 3 == len(V.shape):
        if 1 == d:
            V = convolve(V, dg.reshape([1, 1, w+1]), 'same')
        else:
            V = convolve(V, g.reshape([1, 1, w+1]), 'same')
        if 2 == d:
            V = convolve(V, dg.reshape([1, w+1, 1]), 'same')
        else:
            V = convolve(V, g.reshape([1, w+1, 1]), 'same')
        if 3 == d:
            V = convolve(V, dg.reshape([w+1, 1, 1]), 'same')
        else:
            V = convolve(V, g.reshape([w+1, 1, 1]), 'same')
    elif 2 == len(V.shape):
        if 1 == d:
            V = convolve(V, dg.reshape([1, w+1]), 'same')
        else:
            V = convolve(V, g.reshape([1, w+1]), 'same')
        if 2 == d:
            V = convolve(V, dg.reshape([w+1, 1]), 'same')
        else:
            V = convolve(V, g.reshape([w+1, 1]), 'same')

    return V

def gaussian(x: float, k: float, loc: float, scale: float) -> float:
    return k*scipy.stats.norm.pdf(x, loc=loc, scale=scale)

def gaussian_fit(xx: np.ndarray) -> np.ndarray:
    df = scipy.stats.gaussian_kde(xx)
    sd = np.std(xx)
    params = [df(xx).max()*sd/4*np.sqrt(2*np.pi), np.mean(xx), sd]
    with warnings.catch_warnings():
        fitted_params,_ = scipy.optimize.curve_fit(
            gaussian, xx, df(xx), p0=params)
    if all(fitted_params == params): return None
    return fitted_params

def plot_gaussian_fit(xx: np.ndarray, fitted_params:np.ndarray) -> None:
    assert 3 == len(fitted_params)
    df = scipy.stats.gaussian_kde(xx)
    plt.plot(xx, df(xx), '.')
    x2 = np.linspace(xx.min(), xx.max(), 1000)
    plt.plot(x2, gaussian(x2, *fitted_params), 'r')
    plt.show()

def sog(x: float, k1: float, loc1: float, scale1: float,
    k2: float, loc2: float, scale2: float) -> float:
    return gaussian(x, k1, loc1, scale1) + gaussian(x, k2, loc2, scale2)

def sog_fit(xx: np.ndarray) -> np.ndarray:
    df = scipy.stats.gaussian_kde(xx)
    loc2 = np.mean(xx)+2.5*np.std(xx)
    sd1 = np.std(xx)
    params = [df(xx).max()*sd1/4*np.sqrt(2*np.pi), np.mean(xx), sd1,
        df(loc2)[0]*sd1/4*np.sqrt(2*np.pi), loc2, sd1/4]
    with warnings.catch_warnings():
        fitted_params,_ = scipy.optimize.curve_fit(
            sog, xx, df(xx), p0=params)
    if all(fitted_params == params): return None
    return fitted_params

def plot_sog_fit(xx: np.ndarray, fitted_params:np.ndarray) -> None:
    assert 6 == len(fitted_params)
    df = scipy.stats.gaussian_kde(xx)
    plt.plot(xx, df(xx), '.')
    x2 = np.linspace(xx.min(), xx.max(), 1000)
    plt.plot(x2, gaussian(x2, *fitted_params[:3]), 'r')
    plt.plot(x2, gaussian(x2, *fitted_params[3:]), 'g')
    plt.show()

def fwhm(xx: np.ndarray) -> Tuple[float]:
    logging.warning("FWHM not implemented yet. Using full data range.")
    return (xx.min(), xx.max())

def cell_cycle_fit(data: np.ndarray) -> Tuple[Optional[np.ndarray],str]:
    data = np.array([n.volume for n in nuclei])
    fit = (sog_fit(data), 'sog')
    if fit[0] is None:
        fit = (gaussian_fit(data), 'gaussian')
        if fit[0] is None:
            fit = (fwhm(data), 'fwhm')
    return fit

def sog_range_from_fit(fitted_params: Tuple[float],
    fit_type: str, k_sigma: float) -> Tuple[Tuple[float]]:
    assert 6 == len(fitted_params)
    return gaussian_range_from_fit(fitted_params[:3], fit_type, k_sigma)

def gaussian_range_from_fit(fitted_params: Tuple[float],
    fit_type: str, k_sigma: float) -> Tuple[Tuple[float]]:
    assert 3 == len(fitted_params)
    delta = k_sigma*fitted_params[2]
    return (fitted_params[1]-delta, fitted_params[1]+delta)

def range_from_fit(fitted_params: Tuple[float],
    fit_type: str, k_sigma: float) -> Optional[Tuple[Tuple[float]]]:
    if "sog" == fitted_params:
        return sog_range_from_fit(fitted_params, fit_type, k_sigma)
    if "gaussian" == fitted_params:
        return gaussian_range_from_fit(fitted_params, fit_type, k_sigma)
    if "fwhm" == fitted_params:
        return fitted_params
    return None
