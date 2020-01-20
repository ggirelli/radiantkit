'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import numpy as np
from scipy.signal import convolve
from scipy import stats

def gpartial(V, d, sigma):
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
