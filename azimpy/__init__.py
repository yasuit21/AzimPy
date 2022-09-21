"""
AzimPy: A Python Module to estimate horizontal orientations of Ocean-Bottom Seismometers
========================================================================================

AzimPy is an open-source module to estimate horizontal orientation of ocean-bottom seismometer. 
This module uses Rayleigh-wave polarization method (e.g., Stachnik+ 2012; Doran & Laske 2017). 
One of the main classes `OrientOBS` was inherited from `obspy.clients.fdsn.Client`, which 
allows us to search for earthquake catalog as a web client and compute Rayleigh-wave 
polarizations. This module also provides other classes and functions to perform statistical
analysis of circular data and to plot the estimated azimuth with uncertainty.



[Example]
>>> import obspy as ob
>>> from azimpy import OrientOBS, read_paz

>>> obs = OrientOBS(base_url='USGS', timezone=9)
>>> obs.query_events(
... starttime=ob.UTCDateTime('20200401000000'),
... endtime=ob.UTCDateTime('20201001000000'),
... minmagnitude=6.0,
... maxdepth=100,
... orderby='time-asc',
)

>>> obs.find_stream(
... '/path/to/datadir',
... output_path='/path/to/output',
... polezero_fpath='/path/to/polezero/hoge.paz',
... fileformat=f'*.*.%y%m%d%H%M.sac',
... freqmin=1./40, freqmax=1./20,
... max_workers=4,
... vel_surface=4.0,
... time_before_arrival=-20.0,
... time_after_arrival=600.0,
... distmin=5., distmax=120.,
)

[References]
- Stachnik, J.C., Sheehan, A.F., Zietlow, D.W., et al., 2012. 
    Determination of New Zealand ocean bottom seismometer orientation 
    via Rayleigh-wave polarization. Seismol. Res. Lett., 83, 
    704–713. https://doi.org/10.1785/0220110128 
- Doran, A.K. & Laske, G., 2017. Ocean‐bottom deismometer 
    instrument orientations via automated Rayleigh‐wave 
    arrival‐angle measurements. Bull. Seismol. Soc. Am., 107, 
    691–708. https://doi.org/10.1785/0120160165 
- Takagi, R., Uchida, N., Nakayama, T., et al., 2019. 
    Estimation of the orientations of the S‐net cabled 
    ocean‐bottom sensors. Seismol. Res. Lett., 90, 
    2175–2187. https://doi.org/10.1785/0220190093 


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


from .orientation import OrientOBS
from .plot import OrientSingle, OrientAnalysis, plotCC
from .utils import read_chtbl, read_paz
from .params import set_rcparams, dict_OBStypes

## update rcParams
set_rcparams()