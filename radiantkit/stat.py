'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

from enum import Enum
from matplotlib import pyplot as plt  # type: ignore
import numpy as np  # type: ignore
from numpy.polynomial.polynomial import Polynomial  # type: ignore
import pandas as pd  # type: ignore
import scipy as sp  # type: ignore
from typing import Dict, Optional, Tuple
import warnings


class FitType(Enum):
    SOG = "sum_of_gaussians"
    GAUSSIAN = "gaussian"
    FWHM = "full_width_half_maximum"


class DistanceMode(Enum):
    EDT = 'edt'


Interval = Tuple[float, float]
FitResult = Tuple[np.ndarray, FitType]
PolyFitResult = Dict[str, Polynomial]


def gpartial_w(sigma: float) -> int:
    w = round(8 * sigma + 2)
    if 0 == w % 2:
        w = w + 1
    w = 2 * w + 1
    if 1 == sigma:
        w = 11
    return w


def gpartial_g_dg(w: int, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    if sigma == 0:
        dg = np.array([0, -1, 1])
        g = np.array([0, .5, .5])
    else:
        g = sp.stats.norm.pdf(np.linspace(-w/2., w/2., w+1), scale=sigma)
        x = np.linspace(-(w - 1) / 2, (w - 1) / 2, w + 1)
        k0 = 1 / np.sqrt(2 * np.pi * sigma**2.)
        k1 = 1 / (2 * sigma**2)
        dg = -2 * k0 * k1 * x * np.exp(-k1 * x**2.)
    return g, dg


def gpartial_2D(V: np.ndarray, d: int, sigma: float) -> np.ndarray:
    '''Calculate the partial derivative of 2D V along dimension d using a
    filter of size sigma. Based on code by Erik Wernersson, PhD.'''
    w = gpartial_w(sigma)
    g, dg = gpartial_g_dg(w, sigma)
    if 1 == d:
        V = sp.signal.convolve(V, dg.reshape([1, w+1]), 'same')
    else:
        V = sp.signal.convolve(V, g.reshape([1, w+1]), 'same')
    if 2 == d:
        V = sp.signal.convolve(V, dg.reshape([w+1, 1]), 'same')
    else:
        V = sp.signal.convolve(V, g.reshape([w+1, 1]), 'same')
    return V


def gpartial_3D(V: np.ndarray, d: int, sigma: float) -> np.ndarray:
    '''Calculate the partial derivative of a 3D V along dimension d using a
    filter of size sigma. Based on code by Erik Wernersson, PhD.'''
    w = gpartial_w(sigma)
    g, dg = gpartial_g_dg(w, sigma)
    if 1 == d:
        V = sp.signal.convolve(V, dg.reshape([1, 1, w+1]), 'same')
    else:
        V = sp.signal.convolve(V, g.reshape([1, 1, w+1]), 'same')
    if 2 == d:
        V = sp.signal.convolve(V, dg.reshape([1, w+1, 1]), 'same')
    else:
        V = sp.signal.convolve(V, g.reshape([1, w+1, 1]), 'same')
    if 3 == d:
        V = sp.signal.convolve(V, dg.reshape([w+1, 1, 1]), 'same')
    else:
        V = sp.signal.convolve(V, g.reshape([w+1, 1, 1]), 'same')
    return V


def gpartial(V: np.ndarray, d: int, sigma: float) -> np.ndarray:
    '''Calculate the partial derivative of V along dimension d using a filter
    of size sigma. Based on code by Erik Wernersson, PhD.'''
    if 3 == len(V.shape):
        return gpartial_3D(V, d, sigma)
    elif 2 == len(V.shape):
        return gpartial_2D(V, d, sigma)


def gaussian(x: float, k: float, loc: float, scale: float) -> float:
    return k*sp.stats.norm.pdf(x, loc=loc, scale=scale)


def gaussian_fit(xx: np.ndarray) -> Optional[np.ndarray]:
    df = sp.stats.gaussian_kde(xx)
    sd = np.std(xx)
    params = [df(xx).max()*sd/4*np.sqrt(2*np.pi), np.mean(xx), sd]
    with warnings.catch_warnings():
        fitted_params, _ = sp.optimize.curve_fit(
            gaussian, xx, df(xx), p0=params)
    if all(fitted_params == params):
        return None
    return fitted_params


def plot_gaussian_fit(xx: np.ndarray, fitted_params: np.ndarray) -> None:
    assert 3 == len(fitted_params)
    df = sp.stats.gaussian_kde(xx)
    plt.plot(xx, df(xx), '.')
    x2 = np.linspace(xx.min(), xx.max(), 1000)
    plt.plot(x2, gaussian(x2, *fitted_params), 'r')
    plt.show()


def sog(x: float, k1: float, loc1: float, scale1: float,
        k2: float, loc2: float, scale2: float) -> float:
    return gaussian(x, k1, loc1, scale1) + gaussian(x, k2, loc2, scale2)


def sog_fit(xx: np.ndarray) -> Optional[np.ndarray]:
    df = sp.stats.gaussian_kde(xx)
    loc2 = np.mean(xx)+2.5*np.std(xx)
    sd1 = np.std(xx)
    params = [df(xx).max()*sd1/4*np.sqrt(2*np.pi), np.mean(xx), sd1,
              df(loc2)[0]*sd1/4*np.sqrt(2*np.pi), loc2, sd1/4]
    try:
        with warnings.catch_warnings():
            fitted_params, _ = sp.optimize.curve_fit(
                sog, xx, df(xx), p0=params)
    except RuntimeError:
        return None
    if all(fitted_params == params):
        return None
    return fitted_params


def plot_sog_fit(xx: np.ndarray, fitted_params: np.ndarray) -> None:
    assert 6 == len(fitted_params)
    df = sp.stats.gaussian_kde(xx)
    plt.plot(xx, df(xx), '.')
    x2 = np.linspace(xx.min(), xx.max(), 1000)
    plt.plot(x2, gaussian(x2, *fitted_params[:3]), 'r')
    plt.plot(x2, gaussian(x2, *fitted_params[3:]), 'g')
    plt.show()


def fwhm(xx: np.ndarray) -> Tuple[float]:
    raise NotImplementedError


def cell_cycle_fit(data: np.ndarray) -> FitResult:
    fit = (sog_fit(data), FitType.SOG)
    if fit[0] is None:
        fit = (gaussian_fit(data), FitType.GAUSSIAN)
        if fit[0] is None:
            fit = (fwhm(data), FitType.FWHM)
    return fit


def sog_range_from_fit(data: np.ndarray, fitted_params: np.ndarray,
                       fit_type: FitType, k_sigma: float) -> Tuple[float, ...]:
    assert 6 == len(fitted_params)
    return gaussian_range_from_fit(data, fitted_params[:3], fit_type, k_sigma)


def gaussian_range_from_fit(data: np.ndarray, fitted_params: np.ndarray,
                            fit_type: FitType, k_sigma: float
                            ) -> Tuple[float, ...]:
    assert 3 == len(fitted_params)
    delta = k_sigma*fitted_params[2]
    return (max(fitted_params[1]-delta, data.min()),
            min(fitted_params[1]+delta, data.max()))


def range_from_fit(data: np.ndarray, fitted_params: np.ndarray,
                   fit_type: FitType, k_sigma: float
                   ) -> Tuple[float, ...]:
    if FitType.SOG == fit_type:
        return sog_range_from_fit(data, fitted_params, fit_type, k_sigma)
    if FitType.GAUSSIAN == fit_type:
        return gaussian_range_from_fit(data, fitted_params, fit_type, k_sigma)
    if FitType.FWHM == fit_type:
        return fitted_params
    raise ValueError


def quantile_from_counts(values: np.ndarray, counts: np.ndarray,
                         p: float, cumsummed: bool = False) -> float:
    '''Hyndman, R. J. and Fan, Y. (1996),
    “Sample quantiles in statistical packages,”
    The American Statistician, 50(4), 361 - 365.'''
    assert p >= 0 and p <= 1
    if not cumsummed:
        counts = np.cumsum(counts)
    x = len(values)*p+.5
    if int(x) == x:
        loc = (counts >= x).argmax()
        return values[loc]
    else:
        x1 = values[(counts >= np.floor(x)).argmax()]
        x2 = values[(counts >= np.ceil(x)).argmax()]
        return (x1+x2)/2


def radius_interval_to_area(rInterval: Interval) -> Interval:
    return (np.round(np.pi*np.square(rInterval[0]), 6),
            np.round(np.pi*np.square(rInterval[1]), 6))


def radius_interval_to_volume(rInterval: Interval) -> Interval:
    return (np.round(4/3*np.pi*np.power(rInterval[0], 3), 6),
            np.round(4/3*np.pi*np.power(rInterval[1], 3), 6))


def radius_interval_to_size(rInterval: Interval, n_axes: int = 3) -> Interval:
    if 2 == n_axes:
        return radius_interval_to_area(rInterval)
    else:
        return radius_interval_to_volume(rInterval)


def array_cells_distance_to_point(a: np.ndarray, P: np.ndarray,
                                  aspect: Optional[np.ndarray] = None,
                                  mode: DistanceMode = DistanceMode.EDT):
    assert len(a.shape) == len(P)
    if aspect is not None:
        assert len(a.shape) == len(aspect)
    else:
        aspect = np.ones(len(a.shape))
    if mode is DistanceMode.EDT:
        coords = np.array(np.where(np.ones(a.shape))).transpose()
        return np.sqrt((((coords - P)*aspect)**2).sum(1)).reshape(a.shape)
    else:
        raise ValueError


def radial_fit(x: np.ndarray, y: np.ndarray,
               nbins: int = 200, deg: int = 5
               ) -> Tuple[PolyFitResult, pd.DataFrame]:
    bins = np.linspace(x.min(), x.max(), nbins)
    bin_IDs = np.digitize(x, bins)-1

    x_IDs = list(set(bin_IDs))
    x_mids = bins[x_IDs] + np.diff(bins).min().round(12)/2
    yy_stubs = []
    for bi in x_IDs:
        yy_stubs.append(np.hstack([
            np.quantile(y[bi == bin_IDs], (.25, .5, .75)),
            np.mean(y[bi == bin_IDs])]))
    yy = np.vstack(yy_stubs)

    return (dict(
        q1=Polynomial.fit(x_mids, yy[:, 0], deg),
        median=Polynomial.fit(x_mids, yy[:, 1], deg),
        mean=Polynomial.fit(x_mids, yy[:, 3], deg),
        q3=Polynomial.fit(x_mids, yy[:, 2], deg),
        ), pd.DataFrame.from_dict(dict(
            x=x_mids,
            q1_raw=yy[:, 0],
            median_raw=yy[:, 1],
            mean_raw=yy[:, 3],
            q3_raw=yy[:, 2])))


class RootType(Enum):
    MAXIMA = "max"
    MINIMA = "min"
    BOTH = "both"


def select_maxima_roots(roots: np.ndarray, poly: Polynomial, npoints: int):
    delta = np.diff(poly.domain)/npoints
    return roots[np.logical_and(poly(roots-delta) > 0, poly(roots+delta) < 0)]


def select_minima_roots(roots: np.ndarray, poly: Polynomial, npoints: int):
    delta = np.diff(poly.domain)/npoints
    return roots[np.logical_and(poly(roots-delta) < 0, poly(roots+delta) > 0)]


def get_polynomial_real_roots(
        poly: Polynomial, mode: RootType = RootType.MAXIMA,
        npoints: int = 1000, inDomain: bool = True, inWindow: bool = False
        ) -> np.ndarray:
    assert mode in RootType

    roots = poly.roots()
    roots = roots[np.logical_not(np.iscomplex(roots))]

    if inWindow:
        roots = roots[roots >= poly.window[0]]
        roots = roots[roots <= poly.window[1]]
    if inDomain:
        roots = roots[roots >= poly.domain[0]]
        roots = roots[roots <= poly.domain[1]]

    if RootType.MAXIMA == mode:
        roots = select_maxima_roots(roots, poly, npoints)
    elif RootType.MINIMA == mode:
        roots = select_minima_roots(roots, poly, npoints)

    return np.real(roots)


def get_radial_profile_roots(
        profile: Polynomial, npoints: int = 1000
        ) -> Tuple[np.ndarray, np.ndarray]:

    roots_der1 = get_polynomial_real_roots(
        profile.deriv(), RootType.MAXIMA)
    roots_der2 = get_polynomial_real_roots(
        profile.deriv().deriv(), RootType.MINIMA)

    if 0 == len(roots_der1):
        x, y = profile.linspace(npoints)
        roots_der1 = np.array([x[np.argmax(y)]])
        roots_der1 = roots_der1[roots_der1 >= profile.domain[0]]
        roots_der1 = roots_der1[roots_der1 <= profile.domain[1]]
    else:
        roots_der1 = np.array([roots_der1.min()])
    if 0 == len(roots_der2):
        x, y = profile.linspace(npoints)
        roots_der2 = np.array([x[np.argmin(y)]])
        roots_der2 = roots_der2[roots_der2 >= profile.domain[0]]
        roots_der2 = roots_der2[roots_der2 <= profile.domain[1]]

    if 0 == len(roots_der1):
        roots_der2 = roots_der2[0] if 0 != len(roots_der2) else np.nan
        return (np.nan, roots_der2)

    if 0 != len(roots_der2) and not all(roots_der2 < roots_der1):
        return (roots_der1[0],
                roots_der2[np.argmax(roots_der2 >= roots_der1)])

    return (roots_der1[0], np.nan)
