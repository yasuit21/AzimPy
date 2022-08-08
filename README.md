# AzimPy
Estimate horizontal orientation of ocean-bottom seismograph

[`AzimPy`](https://github.com/yasuit21/AzimPy) is an open-source module for estimating the horizontal orientation of ocean-bottom seismographs. 
This module uses Rayleigh-wave polarization method (e.g., Stachnik+ 2012; Doran & Laske 2017). 
One of the main classes `OrientOBS` is inherited from [`obspy.clients.fdsn.Client`](https://docs.obspy.org/packages/autogen/obspy.clients.fdsn.client.Client.html), which allows us to search for earthquake catalog as a web client and compute Rayleigh-wave polarizations. 
This module also provides other classes and functions for statistical analysis of circular data and plotting the estimated azimuths with uncertainties.


## Usage

```python
import obspy as ob
from azimpy import OrientOBS, read_paz

## Initialize web client
obs = OrientOBS(base_url='USGS', timezone=9)

## Query earthquake event catalog
obs.query_events(
    starttime=ob.UTCDateTime('20200401000000'),
    endtime=ob.UTCDateTime('20201001000000'),
    minmagnitude=6.0,
    maxdepth=100,
    orderby='time-asc',
)

## Read channel table of seismic station
obs.read_chtbl('/path/to/channeltable.txt')

## Compute Rayleigh-wave polarization for each event
## Raw SAC data should be located in '/path/to/datadir'
obs.find_stream(
    '/path/to/datadir',
    output_path='/path/to/output',
    polezero_fpath='/path/to/polezero/hoge.paz',
    fileformat=f'*.*.%y%m%d%H%M.sac',
    freqmin=1./40, freqmax=1./20,
    max_workers=4,
    vel_surface=4.0,
    time_before_arrival=-20.0,
    time_after_arrival=600.0,
    distmin=5., distmax=120.,
)
```

## Installation

### [Recommended] Using `conda` environment and `pip` locally

```
$ conda create -n azimpy-test python=3.9 pip ipython
$ conda activate azimpy-test
(azimpy-test) $ git clone https://github.com/yasuit21/AzimPy.git
(azimpy-test) $ cd AzimPy
(azimpy-test) $ python -m pip install .
```

### Optional installation : [`rpy2`](https://rpy2.github.io/)

Note that the installation takes time.
```
(azimpy-test) $ pip install rpy2
(azimpy-test) $ conda install r-essentials r-base r-circular
```

Then, set environental variables.
```
export R_HOME=/path/to/envs/azimpy-test/lib/R
export R_USER=/path/to/envs/azimpy-test/lib/python3.9/site-packages/rpy2
```

### Use case
<!-- ### Cite -->
<!-- If you use this package to present the results, please cite the  -->
- Sawaki, Y., Yamashita, Y., Ohyanagi, S., Garcia, E.S.M., Ito, A., Sugioka, H., Shinohara, M., and Ito, Y., 
    in revision at *Geophys. J. Int.*

### References
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
    
    
## Note
- The supported format is only `SAC`, but you may use the other formats.
- The observed data files must be located in one directory, where `OrientOBS.find_stream()` will try to search for necessary input files. 
- The author has tested this package in `Linux` environments (`CentOS 7` and `WSL Ubuntu 20.04`), so it might be incompatible when installed in `Windows`.
- `rpy2` is an optional wrapper to run `circular` in `R` language, which performs `Kuiper test`.
- 
    