"""
Estimate OBS sensor orientations by the Rayleigh-wave polarization method.
- Client class for downloading seismograms and computing CC
- A class for circular statistics (single station)

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

from cmath import pi
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta #, timezone
from functools import partial, reduce 
from concurrent import futures
from typing import Union, List, Tuple
import operator
import pickle
import copy
import warnings
import tempfile
# import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, PercentFormatter
import matplotlib.dates as dates
import pandas as pd
from scipy import signal, stats
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

import obspy as ob
import obspy.signal
from obspy.clients.fdsn import Client
from astropy.stats import circstats

from .params import (
    kuiper_level_dict, 
    colormap_cc,
    kwargs_plot_for_legend,
    get_cbar_bound_norm,
)
from .utils import read_chtbl, read_paz
from .rstats import (
    circdist, 
    angle_stats,
    circMedian,
    VALID_RPY2,
)
if VALID_RPY2:
    from .rstats import kuiper_test


###############################################################################
##### Main classes


class OrientOBS(Client):
    """A class to estimate horizontal orientations of 
    Ocean-Bottom Seismometers, inherited from 
    `obspy.clients.fdsn.Client`.
    
    Attributes:
    -----------
    timezone: float = 9
        Set the timezone of observed data.
        -> UTC+`timezone` [h]
        Default to 9, which corresponds to JST.
    base_url: str = 'IRIS'
        Institution name with earthquake catalog.
    See `Client.__init__.__doc__`(https://docs.obspy.org/packages/autogen/obspy.clients.fdsn.client.Client.html)
    for other arguments.
    
    Rayleigh-wave polarization method (e.g., Stachnik+ 2012) 
    is impremented in this class.
    
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
    ... output_path='/path/to/output/station',
    ... polezero_fpath='/path/to/polezero/hoge.paz',
    ... fileformat=f'*.*.%y%m%d%H%M.sac',
    ... freqmin=1./40, freqmax=1./20,
    ... max_workers=4,
    ... vel_surface=4.0,
    ... time_before_arrival=-20.0,
    ... time_after_arrival=600.0,
    ... distmin=5., distmax=120.,
    )

    Then, the output dataframe will be pickled as `station_020_040.pickle` 
    under `/path/to/output/station` directory.
    

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
    """
    
    def __init__(
        self, 
        base_url='IRIS',
        timezone=9,
        **webclient, 
    ):
        """Inits OrientOBS with arguments for 
        obspy.clients.fdsn.Client.
        """
        
        super().__init__(base_url, **webclient)
        self._timezone = timezone

    @property
    def timezone(self):
        return self._timezone
        
    def read_chtbl(self, filepath):
        """A method to read a station channel table.
        
        Parameter:
        -----------
        filepath: str
            Path to the file of the channel table.
            
        """
        
        self.chtbl = read_chtbl(filepath)
        
        return self.chtbl
        
        
    def query_events(
        self, 
        starttime:datetime, endtime:datetime, 
        minmagnitude=6.0,
        maxdepth=150.0,
        **kw_events,
    ):
        '''Query earthquake events from catalog.
        
        Parameters:
        -----------
        starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
            Limit to events on or after the specified start time.
        endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
            Limit to events on or before the specified end time.
        minmagnitude: float, optional
            Limit to events with a magnitude larger than the specified minimum.
        maxdepth: float, optional
            Limit to events with depth, in kilometers, smaller than the specified maximum.
        
        
        See below for other arguments in `obspy.clients.fdsn.Client.get_events`
        
        :type minlatitude: float, optional
        :param minlatitude: Limit to events with a latitude larger than the
            specified minimum.
        :type maxlatitude: float, optional
        :param maxlatitude: Limit to events with a latitude smaller than the
            specified maximum.
        :type minlongitude: float, optional
        :param minlongitude: Limit to events with a longitude larger than the
            specified minimum.
        :type maxlongitude: float, optional
        :param maxlongitude: Limit to events with a longitude smaller than the
            specified maximum.
        :type latitude: float, optional
        :param latitude: Specify the latitude to be used for a radius search.
        :type longitude: float, optional
        :param longitude: Specify the longitude to be used for a radius
            search.
        :type minradius: float, optional
        :param minradius: Limit to events within the specified minimum number
            of degrees from the geographic point defined by the latitude and
            longitude parameters.
        :type maxradius: float, optional
        :param maxradius: Limit to events within the specified maximum number
            of degrees from the geographic point defined by the latitude and
            longitude parameters.
        :type mindepth: float, optional
        :param mindepth: Limit to events with depth, in kilometers, larger than
            the specified minimum.        
        :type maxmagnitude: float, optional
        :param maxmagnitude: Limit to events with a magnitude smaller than the
            specified maximum.
        :type magnitudetype: str, optional
        :param magnitudetype: Specify a magnitude type to use for testing the
            minimum and maximum limits.
        :type includeallorigins: bool, optional
        :param includeallorigins: Specify if all origins for the event should
            be included, default is data center dependent but is suggested to
            be the preferred origin only.
        :type includeallmagnitudes: bool, optional
        :param includeallmagnitudes: Specify if all magnitudes for the event
            should be included, default is data center dependent but is
            suggested to be the preferred magnitude only.
        :type includearrivals: bool, optional
        :param includearrivals: Specify if phase arrivals should be included.
        :type eventid: str (or int, dependent on data center), optional
        :param eventid: Select a specific event by ID; event identifiers are
            data center specific.
        :type limit: int, optional
        :param limit: Limit the results to the specified number of events.
        :type offset: int, optional
        :param offset: Return results starting at the event count specified,
            starting at 1.
        :type orderby: str, optional
        :param orderby: Order the result by time or magnitude with the
            following possibilities:

            * time: order by origin descending time
            * time-asc: order by origin ascending time
            * magnitude: order by descending magnitude
            * magnitude-asc: order by ascending magnitude

        :type catalog: str, optional
        :param catalog: Limit to events from a specified catalog
        :type contributor: str, optional
        :param contributor: Limit to events contributed by a specified
            contributor.
        :type updatedafter: :class:`~obspy.core.utcdatetime.UTCDateTime`,
            optional
        :param updatedafter: Limit to events updated after the specified time.
        :type filename: str or file
        :param filename: If given, the downloaded data will be saved there
            instead of being parsed to an ObsPy object. Thus it will contain
            the raw data from the webservices.
        '''
        
        self.starttime = _check_datetime(starttime)
        self.endtime = _check_datetime(endtime)  
        
        self.events = self.get_events(
            starttime=self.starttime, endtime=self.endtime,
            minmagnitude=minmagnitude,
            maxdepth=maxdepth,
            **kw_events
        )
        
        ## tmp file for events
        with tempfile.NamedTemporaryFile(prefix='tmp_') as tmpf:
            
            self.events.write(tmpf, format='ZMAP')

            ## Create event DataFrame
            df_events = pd.read_csv(
                tmpf.name,
                delim_whitespace=True, header=None, 
                names=['longitude','latitude',2,3,4,'mag','depth',7,8,9]
            )

            timeseries = reduce(
                operator.add, 
                [
                    ser[1].astype('int').map('{:02}'.format) 
                    for ser in df_events[[2,3,4,7,8,9]].iteritems()
                ]
            )
            timeseries = pd.DataFrame(
                pd.to_datetime(timeseries, format='%Y%m%d%H%M%S'),
                columns=['origin_time']
            )
            ## Convert from UTC to the local timezone of observed data
            timeseries += timedelta(hours=self._timezone)  

            self.df_events = pd.concat(
                [timeseries, df_events[['latitude','longitude','depth','mag']]], 
                axis=1,
            )

            del df_events, timeseries 
        
        
    def find_stream(
        self, 
        parent_path, 
        output_path, 
        polezero_fpath,
        fileformat='*.*.%y%m%d%H%M.sac',
        udcomp='U',
        hcomps=('H1','H2'),
        filelength='60m', 
        freqmin=0.02, freqmax=0.04,
        distmin=5., distmax=120.,
        vel_surface=4.0, 
        time_before_arrival=-20.,
        time_after_arrival=600.,
        dphi=0.25,
        cc_asterisk=True,
        outformat='dataframe',
        max_workers=4,
        ignore_BadEpicDist_warning=False,
        filter_kw=dict(corners=2,zerophase=True),
        decimate_kw=dict(factor=10,strict_length=True),
        taper_kw=dict(max_percentage=0.1,type='cosine'),
    ):
        """Find stream to pick up earthquakes from event calalog.
        This method performs Rayleigh wave polarization analysis for each event.
        
        Note that all of the SAC files must be in `parent_path` directory.
        Assume that component names are "*U", '*H1', and '*H2'.
        
        Parameters:
        -----------
        parent_path: 
            Directory path in which all observed data are stored.
            Note that all of the SAC files must be in the directory.
        output_path:
            Directory path in which to store output results.
        polezero_fpath:
            The path to the polezero file to deconvolve instrumental response.
            If set to None, the instrumental response will not be deconvolved.
        fileformat: str = `*.*.%y%m%d%H%M.sac`
            Fileformat of input waveforms (three components) in `parent_path`. 
            You can use `datetime` format with asterisks `*`. For instance,
            - `ABC03.*.%y%m%d%H%M.sac`
                can load `ABC03.U.1408100800.sac`, `ABC03.H1.1408100800.sac`,
                and `ABC03.H2.1408100800.sac`.
            - `%Y%m%d%H0000_ABC03_w*.sac`
                can load `20140810080000_ABC03_wU.sac`, 
                `20140810080000_ABC03_wH1.sac`,
                and `20140810080000_ABC03_wH2.sac`.
            - `%Y%m%d/%H%M.ABC03.*`
                can load `0800.ABC03.U`, `0800.ABC03.H1`,
                and `0800.ABC03.H2` in `parent_path`/`20140810`.
        udcomp: str = `U`
            Component name of vertical seismogram
        hcomps: list = (`H1`,`H2`)
            Component names of original horizontal seismogram.
            `H1` is 90-degree counterclockwise of `H2`.
        filelength: str, defalult to `60m`
            The length of time window in minutes for each raw waveform file. 
            Select one from [`60m`,`1m`]
        freqmin: float = 0.02
            Minimum frequency of the bandpass filter
        freqmax: float = 0.04
            Maximum frequency of the bandpass filter
        distmin: float = 5.
            Minumum epicentral distance in degrees for earthquake events
        distmax: float = 120.
            Maximum epicentral distance in degrees for earthquake events 
        vel_surface: = 4.0
            Assumed travel velocity of Rayleigh waves in km/s 
        time_before_arrival: float = -20.
            Starttime of the time window before estimated Rayleigh-wave arrival 
        time_after_arrival: float = 600.
            Endtime of the time window before estimated Rayleigh-wave arrival 
        dphi: float = 0.25
            Azimuthal increment to rotate horizontal comps in degrees
        cc_asterisk: bool, default to True
            If False, the general cross-correlation will be
            used to determine the orientation.
        outformat str = 'dataframe'
            Output format of the result 
        max_workers: int = 4
            Number of threads in computing
        ignore_BadEpicDist_warning: bool
            If ignoring a warning regarding `BadEpicDist`. 
        filter_kw: dict, dict(corners=2,zerophase=True)
            A kwarg passed to `Stream.filter()`.
            Corner frequencies for the bandpass are already passed.
        decimate_kw: dict, dict(factor=10,strict_length=True)
            A kwarg passed to `Stream.decimate()`.
        taper_kw: dict, dict(max_percentage=0.1,type='cosine')
            A kwarg passed to `Stream.taper()`

        Returns:
        -----------
        df_output: `pandas.DataFrames`
            A dataframe of the results. This output dataframe will also be
            pickled in `output_path` directory as a
            `{output_path.name}_{min_period}_{max_period}.pickle'
        """
        
        self.parent_path = parent_path
        output_path.mkdir(exist_ok=True, parents=True)
        
        ## Events output
        self.events.write(output_path/f'events_{output_path.name}.log','ZMAP')
        
        self.filelength = filelength
        
        # self.freq_range = dict(freqmin=freqmin, freqmax=freqmax)
        self.freqmin = freqmin
        self.freqmax = freqmax
        self.epicdist = dict(min=distmin, max=distmax)
        
        self.vel_surface = vel_surface
        self.time_before_arrival = time_before_arrival
        self.time_after_arrival = time_after_arrival
        
        suffix = fileformat.split(".")[-1]
        stream_stats = ob.read(
            str([*self.parent_path.glob(f'*.{suffix}')][0])
        )[0].stats
        # chtbl = self.chtbl.loc[stream_stats.station]
        
        output = np.zeros([len(self.events), 5]) * np.nan
        
        ## output for stream
        stdir = output_path/'stream'/f'{round(1./freqmax):03d}_{round(1./freqmin):03d}'
        stdir.mkdir(exist_ok=True, parents=True)
        
        
        estimate_azimuth_partial = partial(
            self._estimate_azimuth_for_each_event,
            polezero_fpath=polezero_fpath,
            udcomp=udcomp,
            hcomps=hcomps,
            dphi=dphi,
            cc_asterisk=cc_asterisk,
            filter_kw=filter_kw,
            decimate_kw=decimate_kw,
            taper_kw=taper_kw,
            # stream_stats=stream_stats,
            # chtbl=chtbl, 
            stdir=stdir,
        )

        with tqdm(self.events, unit='event') as pbar:

            future_list = []
            # with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                for i_eve, event in enumerate(pbar):

                    # pbar.set_description(new_starttime.strftime('ST %Y%m%d %H:%M:%S'))
                    
                    try:
                        params, list_datetime_for_datafile = self._calc_params(
                            event, stream_stats
                        )
                    except BadEpicDist:
                        if not ignore_BadEpicDist_warning:
                            warnings.warn('Epicentral distance out of range. Ignore this event.')
                        continue
                        
                    stream = ob.Stream()

                    for filetime in list_datetime_for_datafile:
                        # fileformat = f"{self.parent_path.name}.*.{filetime}.sac"
                        filename_obsdata = filetime.strftime(fileformat)
                        filepath = self.parent_path/filename_obsdata
                        ## Check if the number of files are 1+ or 0 
                        ## HACK: Path.exists() cannot be used because * is included
                        if len(set(self.parent_path.glob(filename_obsdata))):
                            stream += ob.read(filepath)

                    ## Merge the loaded traces into each channel (component)
                    stream.merge()
                    
                    ## HACK: If including a pressure gauge, it is excluded from `stream`
                    for tr in stream.select(channel='PG')+stream.select(channel='DP'):
                        stream.remove(tr)
                    
                    ## check for masked array
                    if not len(stream):
                        continue
                    if_masked = True   # to avoid unreferenced error
                    for tr in stream:
                        if if_masked := bool(np.ma.count_masked(tr.data)):
                            break
                    if if_masked:
                        continue
                        
                    future = executor.submit(
                        estimate_azimuth_partial, 
                        i_eve=i_eve,
                        event=event,
                        params=params,
                        stream=stream,
                    )
                    future_list.append(future)
                        
                for future in futures.as_completed(future_list):
                    try:
                        i_eve, output_event = future.result()
                    except BadStreamError as e:
                        continue
                    else:
                        output[i_eve] = output_event
                    finally:
                        pbar.update(1)
        
        ## Output results of CC    
        filename = f'{output_path.name}_{round(1./freqmax):03d}_{round(1./freqmin):03d}.pickle'
        
        if outformat=='dataframe':
            ## `output` concat with events
            df_output = pd.concat(
                [
                    pd.DataFrame(
                        output,
                        columns=('orientation','CC','CC*','baz','delta')
                    ),
                    self.df_events
                ], axis=1
            )
            
            filepath = output_path/filename
            if filepath.exists():
                os.remove(filepath)
            df_output.to_pickle(filepath)

            return df_output
            
        elif outformat=='ndarray':

            with (output_path/filename).open('wb') as f:
                pickle.dump(output, f)
                
        else:
            return output
             
    def _estimate_azimuth_for_each_event(
        self, 
        i_eve, 
        event, 
        params, 
        stream,
        polezero_fpath,
        udcomp,
        hcomps,
        dphi,
        cc_asterisk,
        filter_kw,
        decimate_kw,
        taper_kw,
        # stream_stats, 
        # chtbl, 
        stdir
    ):
        
        for tr in stream:
            tr.stats.update({
                'origintime': event.origins[0].time + timedelta(hours=self._timezone),
                'depth': event.origins[0].depth/1000,
                'mag': event.magnitudes[0].mag,
                'gcarc': params['epicdist'],
                'back_azimuth': params['baz'],
                'surface_velocity': self.vel_surface,
                'time_window': (self.time_before_arrival, self.time_after_arrival),
                't1': params['t1'],
                't2': params['t2'],
            })

        stream.detrend(type='linear')
        stream.filter(
            'bandpass', 
            freqmin=self.freqmin, freqmax=self.freqmax,
            **filter_kw,
        )
        stream.decimate(**decimate_kw)            
        stream.taper(**taper_kw)

        ## sensitivity correction
        ## Check if properly manupulated
        if polezero_fpath is not None: 
            stream.simulate(
                paz_remove=read_paz(polezero_fpath), #self.polezero_dict.get(int(period))
                pre_filt=(self.freqmin*0.5, self.freqmin, self.freqmax, self.freqmax*2.0),
                sacsim=True,
            )
            
        for tr in stream:
            if hcomps[0] in tr.stats.channel:
                tr.stats.channel = 'N'
            if hcomps[1] in tr.stats.channel:
                tr.stats.channel = 'E'

        stored_stream = stream.copy()
        stream.trim(starttime=params['t1'], endtime=params['t2'])

        ## XXX causing unreference errors
        ## if data period out of range...
        try:
            tr_U = stream.select(channel=f'*{udcomp}').copy()[0]
            U_Hilbert = -np.imag(signal.hilbert(tr_U))  # do not overwrite
        except (IndexError, ValueError):
            raise BadStreamError()
        else:
            S_uu = np.sum(U_Hilbert*U_Hilbert)

        orientation_ranges = np.arange(0, 360, dphi)
        cc = np.zeros([len(orientation_ranges), 2]) * np.nan

        for i_phi, ϕ in enumerate(orientation_ranges):
            baz_apparent = (params['baz'] - ϕ) % 360

            st_rot = stream.select(channel='[N,E]').copy()
            st_rot.rotate(method='NE->RT', back_azimuth=baz_apparent)

            tr_R = st_rot.select(channel='R')[0]

            S_rr = np.sum(tr_R.data*tr_R.data)
            S_ur = np.sum(tr_R.data*U_Hilbert)

            C_ru = S_ur / np.sqrt(S_uu*S_rr)
            D_ru = S_ur / S_uu

            cc[i_phi] = np.array([C_ru, D_ru])

        ## Search for the optimal orientation
        ## If cc_asterisk == False, maximum `C_ru` will be searched.
        argmax = np.argmax(cc[:,int(cc_asterisk)])  
        phi_optimal = orientation_ranges[argmax]
        CCs_optimal = cc[argmax]

        output_event = np.array([
            phi_optimal, *CCs_optimal, 
            params['baz'], params['epicdist']
        ])

        ## Save the rotated stream with the max CC* value
        ## XXX HACK: output SAC has different data points...
        ## 
        stored_stream.trim(
            starttime=stored_stream[0].stats.origintime,
            endtime=stored_stream[0].stats.t2+timedelta(minutes=2)
        )
        st_rot = stored_stream.select(channel='[N,E]').copy()
        st_rot.rotate(
            method='NE->RT', 
            back_azimuth=(params['baz']-phi_optimal)%360
        )
        for tr in st_rot:
            tr.stats.update({
                'back_azimuth': params['baz'], # corrected on Mar 17, 2022
                'rotation_from_north': phi_optimal,
                'CC': CCs_optimal[0],
                'CC2': CCs_optimal[1],
            })
            if 'T' in tr.stats.channel:
                tr.stats['rotation_from_north'] = (tr.stats['rotation_from_north']+90)%360


        outstream = st_rot + stored_stream.select(channel='*U').copy()

        ## Save
        outstream.write(
            str(stdir/f"{tr.stats['origintime'].strftime('%Y%m%d%H%M')}.pickle"),
            format='PICKLE'
        )
        
        return i_eve, output_event
            
    def _calc_params(self, event, stats):
        origin = event.origins[0]
        otime = origin.time
        
        if self._timezone:
            otime += timedelta(hours=self._timezone)  # JST timezone
        
        epi_dist, _, baz = ob.geodetics.gps2dist_azimuth(
            origin.latitude, origin.longitude,
            stats.sac.stla, stats.sac.stlo,
        )
        epi_dist *= 0.001
        epi_dist_deg = ob.geodetics.kilometer2degrees(epi_dist)
                
        if (epi_dist_deg<self.epicdist['min']) or (epi_dist_deg>self.epicdist['max']):
            raise BadEpicDist('Epicentral distance is out of range.')
        
        estimated_arrival = otime + epi_dist/self.vel_surface
        t1 = estimated_arrival + self.time_before_arrival
        t2 = estimated_arrival + self.time_after_arrival
        
        if self.filelength == "1m":
            list_datetime_for_datafile = dates.num2date(
                dates.drange(
                    self._reshape_datetime(t1-timedelta(minutes=2)), 
                    self._reshape_datetime(t2+timedelta(minutes=2)), 
                    timedelta(minutes=1)
                )
            )
        else:
            ## List of filenames with time
            ## Starttime: t0_reshaped; Endtime: t2_reshaped
            ## Counts every one hour
            t0_reshaped = self._reshape_datetime(otime)
            t2_reshaped = self._reshape_datetime(t2+timedelta(minutes=2))
            hr_diff = round(t2_reshaped-t0_reshaped) // 3600
            list_datetime_for_datafile = [
                t0_reshaped + timedelta(hours=dh) 
                for dh in range(1+hr_diff)
            ]
        
        params = dict(
            arrival=estimated_arrival,
            t1=t1, t2=t2,
            baz=baz, epicdist=epi_dist_deg,
        )
        
        return params, list_datetime_for_datafile 
    
    
    def _reshape_datetime(self, dtime):
        
        new_dtime = copy.deepcopy(dtime)
        new_dtime.second = 0
        new_dtime.microsecond = 0
        
        if self.filelength == "60m":
            new_dtime.minute = 0
        elif self.filelength == "1m":
            pass
        else:
            raise ValueError('`filelength` must be `60m` or `1m`.')
            
        return new_dtime

    
    
class OrientSingle():
    """Perform circular statistics for each station.

    Attributes:
    -----------
    df_orient: pd.DataFrame
        A result dataframe by `OrientOBS`
    stationname: str
        A station name for the result dataframe
    if_selection: bool
        Whether to perform bootstrap resampling
    min_CC: float = 0.5
        Minimum cross-correlation values 
        to be considered for analysis
    weight_CC: bool = True
        Whether to weight CC values
    K: bool = 5.0
        Data within `K` times median absolute deviation
    bootstrap_iteration: int = 5000
        Iterration numbers for bootstrap resampling
    bootstrap_fraction_in_percent:float = 100.
        How much percent of numbers of data is used for 
        bootstrap resampling.
    alpha_CI: float = 0.05
        `(1-alpha_CI)*100`% confidence intervals for
        orientation uncertainty 
    kuiper_level:float = 0.05
        The threshold p value in the Kuiper test.

    [Example]
    The example uses a single station `stationA1`.

    >>> import pandas as pd
    >>> from azimpy import OrientSingle, read_chtbl

    ## The output dataframe of orientations
    >>> df_orient = pd.read_pickle(
    ... '/path/to/output/stationA1/stationA1_020_040.pickle')

    ## Init OrientSingle for circular statistics
    >>> orientsingle_raw = OrientSingle(
    ... df_orient, 'stationA1', 
    ... if_selection=False,  # w/o bootstrap analysis
    ... min_CC=0.5, weight_CC=True)
    >>> orientsingle_boot = OrientSingle(
    ... df_orient, 'stationA1', 
    ... if_selection=True,  # perform bootstrap analysis
    ... min_CC=0.5, weight_CC=True, K=5.0,
    ... bootstrap_iteration=5000, alpha_CI=0.05)

    ## Init a figure with subfigures
    >>> fig = plt.figure(figsize=[8,4])
    >>> subfigs = fig.subfigures(nrows=1, ncols=2).flatten()

    ## Plot for `orientsingle_raw`
    >>> orientsingle_raw.plot(
    ... polar=True, 
    ... fig=subfigs[0], in_parentheses='BB',
    ... add_cbar=True)
    >>> subfigs[0].legend(loc=1, bbox_to_anchor=(1,1.15), fontsize='small')

    ## Plot for `orientsingle_boot`
    >>> orientsingle_boot.plot(
    ... fig=subfigs[1], in_parentheses='BB')
    >>> subfigs[1].legend(loc=1, bbox_to_anchor=(1,1.15), fontsize='small')
    """

    def __init__(
        self, 
        df_orient:pd.DataFrame, 
        stationname, 
        if_selection:bool,
        min_CC=0.5, 
        weight_CC=True, 
        K=5.0, 
        bootstrap_iteration:int =5000, 
        bootstrap_fraction_in_percent:float =100.,
        alpha_CI=0.05,
        kuiper_level=0.05,
        in_parentheses=None
    ):
        """Inits OrientSingle"""

        ## Internal params
        self._min_num_eq = 6
        self._goodstation = True
        
        ## Init params
        self.name = stationname
        self.if_selection = if_selection
        self.min_CC = min_CC
        self.alpha_CI = alpha_CI
        self.circmean = np.nan
        self.median = np.nan
        self.MAD = np.nan
        self.kappa = np.nan
        self._mean_vonMises = np.nan
        self.CI1_vonMises = np.nan
        self.p1 = np.nan
        self.p2 = np.nan
        self.CI = np.nan
        self.std1 = np.nan
        self.arcmean = np.nan
        self.std2 = np.nan
        self.std3 = np.nan
        self.CMAD = np.nan
        self._KUIPER_THRESHOLD = kuiper_level_dict[kuiper_level]  #(15%,10%,5%,2.5%,1%)
        self.in_parentheses = in_parentheses

        if self.if_selection:
            if (10. <= bootstrap_fraction_in_percent <= 100.):
                bootstrap_fraction_in_percent *= 0.01
            else:
                raise ValueError('`bootstrap_fraction_in_percent` must be in 10%')

            self.bootstrap_iteration = bootstrap_iteration
            df_orient = df_orient.query(f"CC>={self.min_CC}")

        self.num_eq = len(df_orient)

        if self.num_eq == 0:
            raise IndexError('No data has been extracted')

        self.df_orient = df_orient
        
        
        try:  ## _selfError
            self._perform_circular(
                weight_CC, 
                K, 
                bootstrap_fraction_in_percent,
                kuiper_level,
            )
        except _selfError:
            pass
        
    def __str__(self):
        _str = f'{self.name}, mean={self.circmean:5.1f}, median={self.median:5.1f}, std={self.std1:5.1f}'
        return _str

    def _perform_circular(
        self,
        weight_CC, 
        K, 
        bootstrap_fraction_in_percent,
        kuiper_level,
    ):
        """Internal method to perform circular statistics.
        """
        
        ## Selection 1
        if self.num_eq < self._min_num_eq:
            self._goodstation = False
            raise _selfError(f'Number of EQs smaller than {self._min_num_eq}')

        data_all = self.df_orient[['orientation','CC','CC*']].values
        data_all = data_all[~np.isnan(data_all[:,0])]
        self.ar_orient = data_all[:,0]
        self.ar_cc = data_all[:,1]


        ## Rayleigh test - Calculate p-value
        self.p_Rayleigh = circstats.rayleightest(np.deg2rad(self.ar_orient))
        if self.p_Rayleigh >= 0.01:
            PvalueError(f'Rayleigh test returned large p-value ({self.p_Rayleigh:.2e}>=0.01), and this is excluded')

        ## Average, Variance
        self.circmean, self.circvar = angle_stats(self.ar_orient, weights=self.ar_cc**2)
        
        ## Kuiper test 
        if VALID_RPY2 and (not self.if_selection):
            self.Kuiper_statistic = np.nan
            
            try:
                out = kuiper_test(self.ar_orient, level=kuiper_level)
                # print(stationname, *[out1[0] for out1 in out])
                self.Kuiper_statistic = out[1][0]
                if self.Kuiper_statistic < self._KUIPER_THRESHOLD:
                    PvalueError(
                        'Kuiper test (5%) did not reject MULL hypothesis '
                        +f'({self.Kuiper_statistic:.3f}<{self._KUIPER_THRESHOLD}), and this is excluded'
                    )

            except (AttributeError, ValueError) as e:
                pass

        ## Weighted circular median for all data points
        self.median = circMedian(self.ar_orient, weights=self.ar_cc**2)

        ## Median Absolute Deviation
        self.MAD = np.median(np.abs(circdist(self.ar_orient,self.median)))


        boot_weight = [
            np.ones(len(self.ar_orient)),  ## if not perform bootstrap
            self.ar_cc
        ][weight_CC]

        ## Selection 2
        if self.if_selection:

            ## Extract ϕ in [-K*MAD, K*SMAD]
            SMAD = K * self.MAD
            deviation = circdist(self.ar_orient, self.median)
            self.ar_orient = self.ar_orient[(np.abs(deviation)<=SMAD)]

            self.num_eq = len(self.df_orient)
            if self.num_eq < self._min_num_eq:
                self._goodstation = False
                raise _selfError(f'Number of EQs smaller than {self._min_num_eq}')


            ## Bootstrap resampling and calculate sample mean
            datasize = round(len(self.ar_orient)*bootstrap_fraction_in_percent)
            
            if weight_CC:
                self.ar_orient = np.array([
                    _weight_circmean(
                        data_all[:,:2][np.random.randint(len(self.ar_orient),size=datasize)]
                    ) for _ in range(self.bootstrap_iteration)
                ])

                self.ar_orient, boot_weight = self.ar_orient.T
                self.ar_orient %= 360

            else:
                self.ar_orient = np.array([
                    np.rad2deg(
                        circstats.circmean(
                            np.deg2rad(np.random.choice(self.ar_orient, size=len(self.ar_orient))),
                        )
                    ) % 360 for _ in range(self.bootstrap_iteration)
                ])

            ## Average, Variance
            self.circmean, self.circvar = angle_stats(self.ar_orient, weights=boot_weight) 

            ## Median 
            self.median = circMedian(self.ar_orient, weights=boot_weight)
            
            ## 2.5% (p1) 97.5% (p2) percentiles
            p1, p2 = np.quantile(
                circdist(self.ar_orient, self.circmean),
                q=[self.alpha_CI/2, 1-self.alpha_CI/2]
            )
            self.CI = p2 - p1
            self.p1, self.p2 = (np.array([p1, p2]) + self.circmean) % 360

            
            ## Kuiper test to check vonmises dist.
            # _, self._fpp = kuiper(
            #     np.deg2rad((self.ar_orient-180)), 
            #     partial(stats.vonmises.cdf, kappa=self.kappa, loc=np.deg2rad(self.circmean-180))
            # )
            # if self._fpp < 0.05:
            #     PvalueError(f'Kuiper test denied vonmises distribution ({self._fpp:.2e}<0.05), and this is excluded')
            
        
        ## Circular Mean Absolute Deviation
        self.CMAD = np.inf
        for _dphi in np.arange(0,360,0.5):
            # vtmp = np.sum(np.abs(circdist(self.ar_orient, _dphi))*boot_weight)
            vtmp = np.sum((circdist(self.ar_orient, _dphi))**2*boot_weight)
            if vtmp < self.CMAD:
                self.CMAD = vtmp
                self.arcmean = _dphi
        for _dphi in self.arcmean+np.arange(-5,5,0.02):
            _dphi %= 360
            vtmp = np.sum((circdist(self.ar_orient, _dphi))**2*boot_weight)
            if vtmp < self.CMAD:
                self.CMAD = vtmp
                self.arcmean = _dphi
        self.CMAD /= np.sum(boot_weight)

        ## Fitting to von Mises distribution - (kappa, mu, scale)
        self.kappa, self._mean_vonMises, _ = stats.vonmises.fit(
            np.deg2rad(self.ar_orient-180), 
            floc=np.deg2rad(self.circmean-180), 
            fscale=1
        )

        self.std1 = np.rad2deg(np.sqrt(-2*np.log(1-self.circvar)))
        # self.CI2_vonMises = np.rad2deg(alpha_CI[1]/np.sqrt((1-self.circvar)*self.kappa))
        self.CI1_vonMises = np.rad2deg(stats.vonmises.ppf(1.-self.alpha_CI/2, kappa=self.kappa))
        self.std2 = np.std(circdist(self.ar_orient, self.arcmean, deg=True))
        self.std3 = takagi_error(self.ar_orient, boot_weight)

    def write_obj(self, filepath):
        """Write out the object as a pickled file.

        Parameter:
        -----------
        filepath: str, pathlib.Path
            Path to the output file.
            e.g., `/path/to/stationA1_020_040_boot.pickle`
        """

        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_obj(filepath):
        """Load the existing OrientSingle object as a pickled file.

        Parameter:
        -----------
        filepath: str, pathlib.Path
            Path to the output file.
            e.g., `/path/to/stationA1_020_040_boot.pickle`
        
        Returns: azimpy.OrientSingle
        """
        orientsingle = pd.read_pickle(filepath)
        
        return orientsingle

    
    def plot(
        self, polar=True, fig=None,
        in_parentheses=None,
        add_cbar=False, ax_colorbar=None,
    ):
        """Plot the estimated results for this station 
        w/ or w/o bootstrap.
        Note that `plt.show()` may be required when viewing the outputs.

        Parameters:
        -----------
        polar: bool, defaults to True
            Whether to make a polar plot.
            If `if_selection` == False, `polar` is set to False.
        fig: `matplotlib.figure.Figure` or `matplotlib.figure.SubFigure`, None
            A figure on which the results are drawn.
            If None, a new figure will be created.
        in_parentheses: str, None
            Some texts in parentheses with the station name.
        add_cbar: bool, False
            If True, a colorbar for CC values are drawn.
        ax_colorbar: `matplotlib.axes.Axes`, None
            Axes for plotting a colorbar when add_cbar == True.
            If None, a new axes will be created.

        Returns:
        -----------
        `matplotlib.figure.Figure` or `matplotlib.figure.SubFigure`
            A figure object with the results from a single station.
        
        [Example]
        The example uses a single station `stationA1`.

        >>> import pandas as pd
        >>> from azimpy import OrientSingle, read_chtbl

        ## The output dataframe of orientations
        >>> df_orient = pd.read_pickle(
        ... '/path/to/output/stationA1/stationA1_020_040.pickle')

        ## Init OrientSingle for circular statistics
        >>> orientsingle_raw = OrientSingle(
        ... df_orient, 'stationA1', 
        ... if_selection=False,  # w/o bootstrap analysis
        ... min_CC=0.5, weight_CC=True)
        >>> orientsingle_boot = OrientSingle(
        ... df_orient, 'stationA1', 
        ... if_selection=True,  # perform bootstrap analysis
        ... min_CC=0.5, weight_CC=True, K=5.0,
        ... bootstrap_iteration=5000, alpha_CI=0.05)

        ## Init a figure with subfigures
        >>> fig = plt.figure(figsize=[8,4])
        >>> subfigs = fig.subfigures(nrows=1, ncols=2).flatten()

        ## Plot for `orientsingle_raw`
        >>> orientsingle_raw.plot(
        ... polar=True, 
        ... fig=subfigs[0], in_parentheses='BB',
        ... add_cbar=True)
        >>> subfigs[0].legend(loc=1, bbox_to_anchor=(1,1.15), fontsize='small')

        ## Plot for `orientsingle_boot`
        >>> orientsingle_boot.plot(
        ... fig=subfigs[1], in_parentheses='BB')
        >>> subfigs[1].legend(loc=1, bbox_to_anchor=(1,1.15), fontsize='small')
        """

        if self.if_selection:
            polar = False

        if in_parentheses is None:
            in_parentheses = self.in_parentheses

        if fig is None:
            fig = plt.figure(
                figsize=[
                    np.array([3,2]),
                    np.array([3,3.2]),
                ][polar]
            )

        if self.if_selection:
            theta_range = (-10, 10)
            bin_interval = 0.5
            rorigin = -0.3
        else:
            theta_range = (0, 360)
            bin_interval = 5
            rorigin = 0.

        arr_theta_axis = np.linspace(-np.pi, np.pi, 1000)
        bound_cc, norm_cc = get_cbar_bound_norm(min_CC=self.min_CC)

        #######################################################################
        ##### Main part for plot

        ax = fig.subplots(subplot_kw=dict(polar=polar))

        ax.set_title(
            f'{self.name}'+[f' ({in_parentheses})',''][int(in_parentheses is None)], 
            # fontdict={'fontsize':13}
        )
        ax.set(
            xlabel=[
                "$H_1$ azimuth [$\degree$]",
                "Azimuthal deviation from circular mean [$\degree$]"
            ][self.if_selection]
        )

        ## Different types of plot
        if polar:
            ## Polar plot
            _counts, _bins = np.histogram(
                np.deg2rad(self.ar_orient-self.if_selection*self.circmean),
                bins=np.deg2rad(np.arange(theta_range[0], theta_range[1]+0.01, bin_interval)),
            )
            
            freqmax = np.max(_counts)/bin_interval/np.sum(_counts)*100
            rticks = np.arange(1,np.ceil(freqmax)+0.1) * 0.01
                
            ax.set(
                theta_zero_location="N", theta_direction=-1, 
                xticks=np.deg2rad([
                    np.arange(0,360,30),
                    np.arange(theta_range[0],theta_range[1]+1, 5)
                ][self.if_selection]),
                rorigin=rorigin,
                ylim=(0,np.sqrt((freqmax+1.5)*0.01)),
                yticks=np.sqrt(rticks), 
                yticklabels=[f'{rtk*100:.0f}%' for rtk in rticks],
                thetamin=theta_range[0], thetamax=theta_range[1]
            )
            ax.tick_params('x', labelsize='small')
            ax.tick_params('y', labelsize='x-small')
            ax.xaxis.set_minor_locator(AutoMinorLocator())

            ax.bar(
                _bins[:-1], height=np.sqrt(_counts/np.sum(_counts)/bin_interval), 
                width=np.deg2rad(bin_interval), alpha=0.8,
            )
            
            if not self.if_selection:
                line_mean = ax.axvline(
                    x=np.deg2rad(self.circmean), 
                    zorder=5, **kwargs_plot_for_legend["mean"]
                )
                
                # ## von Mises dist.
                # line1, = ax.plot(
                #     arr_theta_axis+np.pi, 
                #     ax.get_rmax() * np.sqrt(stats.vonmises.pdf(
                #         arr_theta_axis-np.deg2rad(self.circmean-180), kappa=self.kappa
                #     )),
                #     c='darkgreen', zorder=5,
                # )
                
            line_median = ax.axvline(
                x=np.deg2rad(self.median-self.if_selection*self.circmean), 
                # label='Circular median $\mu_m$', c='b', linewidth=2.0, linestyle='dotted', 
                zorder=5,
                **kwargs_plot_for_legend["median"]
            )
                    
        else:
            ## Standard plot

            ax.yaxis.get_major_locator().set_params(integer=True)
            # if j == 2:
            #     ax.tick_params(labelright=True, labelleft=False)
            
            _counts, bins, _ = ax.hist(
                [self.ar_orient, circdist(self.ar_orient, self.circmean)][self.if_selection],
#                     self.ar_orient-self.if_selection*self.circmean, 
                range=theta_range, 
                bins=round(np.sum(np.abs(theta_range))//bin_interval), 
                density=self.if_selection, alpha=0.8,
            )
            
            
            ax.set(
                xlim=theta_range, 
                xticks=[
                    np.arange(0,361,90),
                    np.arange(theta_range[0],theta_range[1]+1, 5)
                ][self.if_selection],
                ylabel=['Counts',None][self.if_selection]
                # ylim=(0,)
            )  
            
            # ax.tick_params(labelsize=11)
            
            if self.if_selection: 
                ax.xaxis.set_minor_locator(AutoMinorLocator(5))
                ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
                
                ## von Mises dist.
                try:
                    vonmises_dist = stats.vonmises.pdf(arr_theta_axis, kappa=self.kappa)
                    line1, = ax.plot(
                        np.rad2deg(arr_theta_axis)+circdist(
                            np.rad2deg(self._mean_vonMises)+180, 
                            self.circmean
                        ), 
                        ax.get_ylim()[-1]*vonmises_dist/vonmises_dist.max(),
                        zorder=5, **kwargs_plot_for_legend["vonmises"]
                    )
                except AttributeError:
                    pass

                ax.set(
                    ylim=(0,np.max([0.1,ax.get_ylim()[-1]])),
                )
            else:
                ax.xaxis.set_minor_locator(AutoMinorLocator(6))
                
                line_mean = ax.axvline(
                    x=self.circmean, 
                    zorder=5, **kwargs_plot_for_legend["mean"]
                )
            #     ## von Mises dist.
            #     line1, = ax.plot(
            #         np.rad2deg(arr_theta_axis)+180, 
            #         ax.get_ylim()[-1] * stats.vonmises.pdf(
            #             arr_theta_axis-np.deg2rad(self.circmean-180), kappa=self.kappa
            #         ), 
            #         c='darkgreen', zorder=5,
            #     )
                
            line_median = ax.axvline(
                x=self.median-self.if_selection*self.circmean, 
                zorder=5, **kwargs_plot_for_legend["median"]
            )

        ## Texts for `ax`
        ax.text(
            *[(0.15,1.06),(0.15,1.08)][polar], 
            f"$\mu$: {self.circmean:5.1f}$\degree$\n"+
            f"$\mu_m$: {self.median:5.1f}$\degree$", 
            fontsize='x-small', va='bottom', ha='right', transform=ax.transAxes,
        )        
        
        ## Different plot or texts dependeing on `if_selection`
        if self.if_selection:
            ax.text(
                1.20, 1.06, 
                f"{round((1-self.alpha_CI)*100)}%CI: {self.CI1_vonMises:4.1f}$\degree$\n"+
                f"$\kappa$: {self.kappa:.2E} ", 
                fontsize='x-small', va='bottom', ha='right', transform=ax.transAxes,
            )
        else:
            ## Scatter: (azimuth, CC)
            if polar:
                
                ## modified due to upgrade of matplotlib ver. 3.3
#                     _mappable = ax.scatter(
#                         np.deg2rad(self.ar_orient), 
#                         self.ar_cc*ax.get_rmax(), 
#                         s=100, c=self.ar_cc, alpha=0.8,
#                         marker='.', linewidth=0.6, 
#                         edgecolor='face', facecolor='none', 
#                         norm=norm_cc, cmap=colormap_cc, zorder=4., 
#                     )
                _mappable = ax.scatter(
                    np.deg2rad(self.ar_orient), 
                    self.ar_cc*ax.get_rmax(), 
                    s=30, alpha=0.8,
                    marker='o', linewidth=0.6, 
                    edgecolor=colormap_cc(norm_cc(self.ar_cc)), 
                    facecolor='none', 
                    zorder=4., 
                )
            else:
                _mappable = ax.scatter(
                    self.ar_orient, 
                    self.ar_cc*ax.get_ylim()[1], 
                    s=100, c=self.ar_cc, alpha=0.8,
                    marker='.', linewidth=0.6, 
                    edgecolor='face', facecolor='none', 
                    norm=norm_cc, cmap=colormap_cc, zorder=4., 
                )
            _mappable.set_facecolor('none')
            
            ax.text(
                *[(1.06,1.06),(1.15,1.08)][polar],
                f"MAD:{self.MAD:5.1f}$\degree$",
                fontsize='x-small', va='bottom', ha='right', transform=ax.transAxes,
            )

        if add_cbar and (not self.if_selection) and polar:
            if ax_colorbar is None:
                ax_colorbar = fig.add_axes([0.0, 1.0, 0.3, 0.03])
            
            fig.colorbar(
                plt.cm.ScalarMappable(cmap=colormap_cc, norm=norm_cc),
                cax=ax_colorbar, ticks=bound_cc, pad=0.01,
                spacing='proportional', orientation='horizontal',
            )
            ax_colorbar.set_title('${C}_{\~{Z}R}$',fontsize='small')
            ax_colorbar.tick_params(
                which='both', direction='in', labelsize='x-small', 
                top=True, bottom=False, labeltop=True, labelbottom=False
            )

        return fig

    
###############################################################################

## Exception classess

class BadEpicDist(Exception):
    pass

class BadStreamError(Exception):
    pass

class _selfError(Exception):
    pass

class PvalueError(Exception):
    pass


## Minor functions

def _check_datetime(dt):
    try:
        out = ob.UTCDateTime(dt)
        return out
    except TypeError:
        raise Exception('Invalid datetime input.')
        
        
def takagi_error(arr_in_deg, arr_cc):
    """Computation of circular uncertainty by Takagi+ (2019).
    
    - Takagi, R., Uchida, N., Nakayama, T., et al., 2019. 
        Estimation of the orientations of the S‐net cabled 
        ocean‐bottom sensors. Seismol. Res. Lett., 90, 
        2175–2187. https://doi.org/10.1785/0220190093 
    
    """
    n_takagi = np.cos(np.deg2rad(arr_in_deg))
    e_takagi = np.sin(np.deg2rad(arr_in_deg))

    vn_ave = np.sum(n_takagi*arr_cc) / np.sum(arr_cc)
    ve_ave = np.sum(e_takagi*arr_cc) / np.sum(arr_cc)

    n_var = np.sum(arr_cc*(n_takagi-vn_ave)**2) / (np.sum(arr_cc))
    e_var = np.sum(arr_cc*(e_takagi-ve_ave)**2) / (np.sum(arr_cc))

    return np.rad2deg(np.sqrt((ve_ave**2*n_var+vn_ave**2*e_var)/(vn_ave**2+ve_ave**2)**2))


def _weight_circmean(data):
    """
    Parameters:
    -----------
    data: ndarray shape=(N,2)
    """
    _circmean = circstats.circmean(
        np.deg2rad(data[:,0]),
        weights=data[:,1]**2
    )
    
    return [np.rad2deg(_circmean), np.sum(data[:,1]**2)]