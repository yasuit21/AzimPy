# AzimPy
Estimate horizontal orientation of ocean-bottom seismograph

[`AzimPy`](https://github.com/yasuit21/AzimPy) is an open-source module to estimate horizontal orientation of ocean-bottom seismometer. 
This module uses Rayleigh-wave polarization method (e.g., Stachnik+ 2012; Doran & Laske 2017). 
One of the main classes `OrientOBS` was inherited from [`obspy.clients.fdsn.Client`](https://docs.obspy.org/packages/autogen/obspy.clients.fdsn.client.Client.html), which allows us to search for earthquake catalog as a web client and compute Rayleigh-wave polarizations. 
This module also provides other classes and functions to perform statistical analysis of circular data and to plot the estimated azimuth with uncertainty.


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
    
### Use case
- Sawaki, Y., Yamashita, Y., Ohyanagi, S., Garcia, E.S.M., Ito, A., Sugioka, H., Shinohara, M.,
    and Ito, Y., in revision, Seafloor Depth Controls Seismograph Orientation Uncertainty, *Geophys. J. Int.*