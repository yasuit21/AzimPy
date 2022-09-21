"""
Restore default parameters and functions.

This file is a part of `AzimPy`.
This project is licensed under the MIT License.
------------------------------------------------------------------------------
MIT License

Copyright (c) 2022 Yasunori Sawaki

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
------------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.colors as mcolors
from matplotlib import rc, rcParams

kuiper_level_dict = dict(zip(
    [0.15, 0.1, 0.05, 0.025, 0.01],
    [1.537, 1.62, 1.747, 1.862, 2.001]
))

colormap_cc = mcolors.ListedColormap(
    ("gray","lightsalmon","darkorange","red"), N=256
)
kwargs_plot_for_legend = {
    "mean": dict(
        label='Circular mean $\mu$',
        c='darkgreen',
        linewidth=1.5,
        linestyle='dashed'
    ),
    "median": dict(
        label='Circular median $\mu_m$',
        c='b',
        linewidth=2.0,
        linestyle='dotted',
    ),
    "vonmises": dict(
        label='von Mises distribution',
        c='darkgreen',
        linestyle='solid'
    ),
}

dict_OBStypes = {
    360: "BB",
    1: "1s",
    20: "20s",
    120: "120s",
}


## Functions

def get_cbar_bound_norm(min_CC=0.5):

    bound_cc = np.array([min_CC,0.7,0.9,1.0])
    norm_cc = mcolors.BoundaryNorm(bound_cc, ncolors=256, extend='min')

    return bound_cc, norm_cc


def set_rcparams():
    
    rc('figure', figsize=[8,6], facecolor='w', dpi=150, )
    rc('figure.constrained_layout', use=True)
    ## savefig
    rc(
        'savefig', format='png',
        # dpi=plt.rcParams['figure.dpi'], 
        edgecolor=rcParams['figure.edgecolor'],
        facecolor=rcParams['figure.facecolor'],
        bbox='tight', transparent=False,
    )
    rc('font', family='sans-serif', size='12')
    rc('text', usetex=False)

    rc('axes', grid=True, linewidth=1.0, axisbelow=True)
    rc('axes.grid', axis='both')

    rc('lines', linestyle='-', linewidth=1.0, marker=None)

    rc('grid', linewidth=0.5, linestyle='--', alpha=0.8)

    rc('xtick', direction='out', bottom=True, top=True, labelsize=12)
    rc('xtick.major', width=1.0, size=5)
    rc('xtick.minor', visible=True, size=2.5)
    rc('ytick', direction='out', left=True, right=True, labelsize=12)
    rc('ytick.major', width=1.0, size=5)
    rc('ytick.minor', visible=True, size=2.5)

    rc(
        'legend', markerscale=1, 
        frameon=True, fancybox=False, framealpha=1, #edgecolor='k'
    )