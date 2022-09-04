"""
Functions for circular statistics including `R-CIRCULAR` modules.

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

import sys
import pkgutil
import warnings
import gc

import numpy as np
from astropy.stats import circstats #, kuiper

## `rpy2` calling R langulage from python
## used to perform a Kuiper test  

if VALID_RPY2 := 'rpy2' in {p.name for p in set(pkgutil.iter_modules())}: #False
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.packages import importr
        VALID_RPY2 = True
        circular = importr("circular")
    except ValueError as e:
        VALID_RPY2 = False
        warnings.warn(
            f"`rpy2` is not set up. Kuiper test will not be performed. \n{e}"
        )
else:
    # except ModuleNotFoundError:
    warnings.warn(
        "`rpy2` is not installed so the Kuiper test will not be performed."
    )
    
    
def circdist(x, y, deg=True):
    """
    Calculate the difference between two angels `x-y`.
    Calculated values are in (180,180] if `deg=True`, 
    in (-pi, pi] if `deg=False`.
    
    Parameters:
    -----------
    x, y: int, float
        Input angles to calculate the difference `x-y`.
    deg: bool, default to True
        If True, input and output values are in degrees.
        If False, values are in radian.
    """
    
    halfperiod = 180. if deg else np.pi
    angle_difference = (x-y) % (2*halfperiod)
    pi_minus_angle_diff = halfperiod - angle_difference
    sign = np.sign(pi_minus_angle_diff)
    
    return angle_difference + sign*(1-sign)*halfperiod


def angle_stats(data, weights=None, period=360):
    """Circular statistics - Average and variance.
    Using module in `astropy.stats.circstats`
    https://docs.astropy.org/en/stable/stats/circ.html
    
    Reference : https://qiita.com/kn1cht/items/6be4f9b7ff2da940ca68
    """
    
    data_nonan = np.deg2rad(data[~np.isnan(data)])
    
    ## Average
    μ = np.rad2deg(circstats.circmean(data_nonan, weights=weights)) % 360
    
    ## Variance
    V = circstats.circvar(data_nonan, weights=weights)
    
    ## Standard deviation
    # σ = np.rad2deg(circstats.circstd(data_nonan))
    
    ## Unbiased deviation
    
    
    return μ, V, #σ


def circMedian(data, increment=0.1, weights=None, deg=True):
    """Calculate the circular median.
    Modified from Fisher (1995).
    """
    
    halfperiod = 180. if deg else np.pi
    period = 2 * halfperiod
    
    circmean, _ = angle_stats(data, weights=weights, period=period)
    
    angrid = np.arange(0, period, increment)
    cost = [
        np.average(np.abs(circdist(data, phi, deg=deg))*[weights,1][weights is None])
        for phi in angrid
    ]
    min_cost = min(cost)
    
    median, _ = angle_stats(
        angrid[[_jj for _jj, v in enumerate(cost) if v == min_cost]],
        weights=None, period=period
    )
    
    if abs(circdist(median, circmean)) >= (period/4):
        median = (median + halfperiod) % period
        
    return median



def kuiper_test(ar_orient, level=0.05):
    ro.r.assign("arr_orient", numpy2ri.py2rpy(np.deg2rad(ar_orient)))
    ro.r.assign("alpha", level)  # 
    out = ro.r("""
    arr_orient <- circular(arr_orient)
    kuiper.test(arr_orient, alpha=alpha)
    """)
#     alpha <- 0.05
    gc.collect()

    return out