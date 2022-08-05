## Estimate OBS sensor orientations by a Rayleigh-wave polarization method
## 20210616 Yasunori Sawaki
## 20220204 Longer time window for output streams  
## 20220206 Add Kuiper test for `OrientSingle`  V-test
## 20220316 Modify instrument, Correct BAZ header in RT comps. 
## 20220731 `orientation.py` for main classes


import sys
import os
from pathlib import Path
# import shutil
import time
from datetime import datetime, timedelta #, timezone
# from collections import Counter, defaultdict
# from decimal import Decimal
# from itertools import product
from functools import partial, reduce 
from concurrent import futures
from typing import Union, List, Tuple
import operator
import pickle
# import string
# import re
import copy
import warnings
import tempfile
# import argparse

import numpy as np
import matplotlib.dates as dates
import pandas as pd
from scipy import (
    signal, interpolate, constants, linalg,
    stats, special# optimize, integrate, 
)
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

import obspy as ob
import obspy.signal
from obspy.clients.fdsn import Client
# from obspy import taup
from obspy.signal.invsim import corn_freq_2_paz

from astropy.stats import circstats

from .rstats import (
    circdist, 
    angle_stats,
    circMedian,
    VALID_RPY2,
)
from .params import kuiper_level_dict
from .utils import read_chtbl, read_paz

if VALID_RPY2:
    from .rstats import kuiper_test



###############################################################################
##### Main classes


class OrientOBS(Client):
    
    def __init__(
        self, 
        base_url='IRIS',
        timezone=9,
        **webclient, 
    ):
        f"""A class to estimate horizontal orientations of 
        Ocean-Bottom Seismometers, inherited from 
        `obspy.clients.fdsn.Client`.
        
        Parameters:
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
        
        >>> obs.read_chtbl('/path/to/channeltable.txt')
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

        """
        
        super().__init__(base_url, **webclient)
        self.timezone = timezone
        
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
            ## UTC->JST
            timeseries += timedelta(hours=self.timezone)  

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
        taper_type='cosine',
        outformat='dataframe',
        max_workers=2,
    ):
        '''Find stream to pick up earthquakes from event calalog.
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
        taper_type='cosine'
        outformat str = 'dataframe'
            Output format of the result 
        max_workers: int = 2
            Number of processes in parallel computing
            
        '''
        
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
        
        
        stream_stats = ob.read(
            str([*self.parent_path.glob(f'*.{fileformat.split(".")[-1]}')][0])
        )[0].stats
        chtbl = self.chtbl.loc[stream_stats.station]
        
        output = np.zeros([len(self.events), 5]) * np.nan
        
        ## output for stream
        stdir = output_path/'stream'/f'{int(1./freqmax):03d}_{int(1./freqmin):03d}'
        stdir.mkdir(exist_ok=True, parents=True)
        
        
        estimate_azimuth_partial = partial(
            self._estimate_azimuth_for_each_event,
            polezero_fpath=polezero_fpath,
            dphi=dphi,
            cc_asterisk=cc_asterisk,
            taper_type=taper_type,
            stream_stats=stream_stats,
            chtbl=chtbl, 
            stdir=stdir,
        )
        
        # if max_workers > 1:
            
        nw_submit = max_workers

        with tqdm(self.events, unit='event') as pbar:

            gen = iter(range(nw_submit)) 
            future_list = []
            with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                for i_eve, event in enumerate(pbar):

                    # pbar.set_description(new_starttime.strftime('ST %Y%m%d %H:%M:%S'))
                    
                    try:
                        params, list_datetime_for_datafile = self._calc_params(
                            event, stream_stats
                        )
                    except BadEpicDist:
                        warnings.warn('Epicentral distance out of range.\n')
                        continue
                        
                    stream = ob.Stream()

                    for filetime in list_datetime_for_datafile:
                        # fileformat = f"{self.parent_path.name}.*.{filetime}.sac"
                        filename_obsdata = filetime.strftime(fileformat)
                        try:
                            stream += ob.read(str(self.parent_path/filename_obsdata))
                        except Exception:
                            warnings.warn('Event period is out of range. Ignore this event.')
                            continue

                    ## Merge the loaded traces into each channel (component)
                    stream.merge()
                    
                    ## If including a pressure gauge, it is excluded from `stream`
                    for tr in stream.select(channel='PG')+stream.select(channel='DP'):
                        stream.remove(tr)

                        
                    future = executor.submit(
                        estimate_azimuth_partial, 
                        i_eve=i_eve,
                        event=event,
                        params=params,
                        stream=stream,
                    )
                    future_list.append(future)

                    try:
                        next(gen)
                    except StopIteration as e:
                        
                        for future in future_list:
                            i_eve, output_event = future.result()
                            output[i_eve] = output_event

                        future_list = []

                        gen = iter(range(nw_submit))
                        time.sleep(1)
                
                if future_list:
                    for future in future_list:
                        i_eve, output_event = future.result()
                        output[i_eve] = output_event
        
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
            
            df_output.to_pickle(output_path/filename)
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
        dphi,
        cc_asterisk,
        taper_type,
        stream_stats, 
        chtbl, 
        stdir
    ):
        
        for tr in stream:
            tr.stats.update({
                'origintime': event.origins[0].time + timedelta(hours=self.timezone),
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
            #**self.freq_range,
            freqmin=self.freqmin, freqmax=self.freqmax,
            corners=2, zerophase=True,
        )

        stream.decimate(factor=10, strict_length=True)            
        stream.taper(max_percentage=0.1, type=taper_type)

        ## sensitivity correction
        ## Check if properly manupulated   
        stream.simulate(
            paz_remove=read_paz(polezero_fpath), #self.polezero_dict.get(int(period))
            pre_filt=(self.freqmin*0.5, self.freqmin, self.freqmax, self.freqmax*2.0),
            sacsim=True,
        )
            
        for tr in stream:
            if 'H1' in tr.stats.channel:
                tr.stats.channel = 'N'
            if 'H2' in tr.stats.channel:
                tr.stats.channel = 'E'

        stored_stream = stream.copy()
        stream.trim(starttime=params['t1'], endtime=params['t2'])

        try:
            tr_U = stream.select(channel='*U').copy()[0]
            U_Hilbert = -np.imag(signal.hilbert(tr_U))  # do not overwrite
        except IndexError:  ## if data period out of range
            print(params['t1'])
            # continue
        except ValueError as e:
            print(params['t1'])
            # continue

        S_uu = np.sum(U_Hilbert*U_Hilbert)

        orientation_ranges = np.arange(0, 360, dphi)
        cc = np.zeros([len(orientation_ranges), 2]) * np.nan

        for i_phi, ϕ in enumerate(orientation_ranges):
            α = (params['baz'] - ϕ) % 360

            st_rot = stream.select(channel='[N,E]').copy()
            st_rot.rotate(method='NE->RT', back_azimuth=α)

            tr_R = st_rot.select(channel='R')[0]

            S_rr = np.sum(tr_R.data*tr_R.data)
            S_ur = np.sum(tr_R.data*U_Hilbert)

            C_ru = S_ur / np.sqrt(S_uu*S_rr)
            D_ru = S_ur / S_uu

            cc[i_phi] = np.array([C_ru, D_ru])

        ## Search for the optimal orientation
        ## If cc_asterisk == False, maximum `C_ru` will be searched.
        argmax = np.argmax(cc[:,int(cc_asterisk)])  


        output_event = np.array([
            orientation_ranges[argmax], *cc[argmax], 
            params['baz'], params['epicdist']
        ])

        ## Save the rotated stream with the max CC* value
        stored_stream.trim(starttime=stored_stream[0].stats.origintime)
        st_rot = stored_stream.select(channel='[N,E]').copy()
        st_rot.rotate(
            method='NE->RT', 
            back_azimuth=(params['baz']-orientation_ranges[argmax])%360
        )
        for tr in st_rot:
            tr.stats.update({
                'back_azimuth': params['baz'], # corrected on Mar 17, 2022
                'rotation_from_north': orientation_ranges[argmax],
                'CC': cc[argmax][0],
                'CC2': cc[argmax][1],
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
        
        if self.timezone:
            otime += timedelta(hours=self.timezone)  # JST timezone
        
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
            list_datetime_for_datafile = [
                self._reshape_datetime(t1),
                self._reshape_datetime(t2)
            ]
            if list_datetime_for_datafile[1] == list_datetime_for_datafile[0]:
                list_datetime_for_datafile.pop()
        
        ## Using `set` can ignore duplicates
        ## Note that this STRONGLY DEPENDS ON FILE FORMAT!
        # filelist = {
        #     t.strftime('%Y%m%d%H%M')[2:] 
        #     for t in list_datetime_for_datafile
        # }
        
        params = dict(
            arrival=estimated_arrival,
            t1=t1, t2=t2,
            baz=baz, epicdist=epi_dist_deg,
        )
        
        return params, list_datetime_for_datafile 
#         return estimated_arrival, t1, t2, baz, filelist
    
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
    def __init__(
        self, df_orient:pd.DataFrame, stationname, if_selection:bool,
        min_CC=0.5, weight_CC=True, K=5.0, bootstrap_iteration=2000, alpha_CI=0.05,
        kuiper_level=0.05,
    ):
        ## Internal params
        self._min_num_eq = 6
        self._goodstation = True
        
        ## Init params
        self.name = stationname
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
        
        
        try:  ## _OrientSingleError

            ## Selection 1
            if if_selection:
                df_orient = df_orient[df_orient['CC']>=min_CC]

            if len(df_orient) == 0:
                raise IndexError('No data has been extracted')

            self.df_orient = df_orient
            self.num_eq = len(self.df_orient)
            if self.num_eq < self._min_num_eq:
                self._goodstation = False
                raise _OrientSingleError(f'Number of EQs smaller than {self._min_num_eq}')

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
            if VALID_RPY2 and (not if_selection):
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
                
            
            # ## V test - 
            # # Performs the Rayleigh test of uniformity 
            # # where the alternative hypothesis H1 is assumed to have a known mean angle.
            # # https://docs.astropy.org/en/stable/api/astropy.stats.vtest.html
            # self.p_Vtest = circstats.vtest(np.deg2rad(self.ar_orient), np.deg2rad(self.circmean))
            # if self.p_Vtest >= 0.01:
            #     PvalueError(f'V test returned large p-value ({self.p_Vtest:.2e}>=0.01), and this is excluded')

            ## Weighted circular median for all data points
            self.median = circMedian(self.ar_orient, weights=self.ar_cc**2)

            ## Median Absolute Deviation
            self.MAD = np.median(np.abs(circdist(self.ar_orient,self.median)))


            boot_weight = [
                np.ones(len(self.ar_orient)),  ## if not perform bootstrap
                self.ar_cc
            ][weight_CC]

            ## Selection 2
            if if_selection:

                ## Extract ϕ in [-K*MAD, K*SMAD]
                SMAD = K * self.MAD
                deviation = circdist(self.ar_orient, self.median)
                self.ar_orient = self.ar_orient[(np.abs(deviation)<=SMAD)]

                self.num_eq = len(self.df_orient)
                if self.num_eq < self._min_num_eq:
                    self._goodstation = False
                    raise _OrientSingleError(f'Number of EQs smaller than {self._min_num_eq}')
                #deviation = (self.ar_orient - self.circmean + 180) % 360 - 180  # [-180,180)
                #self.ar_orient = self.ar_orient[(deviation>=-SMAD) & (deviation<=SMAD)]


                ## Bootstrap resampling and calculate sample mean
                datasize = len(self.ar_orient) # max((len(self.ar_orient)*), ) #
                
                if weight_CC:
                    self.ar_orient = np.array([
                        _weight_circmean(
                            data_all[:,:2][np.random.randint(len(self.ar_orient),size=datasize)]
                        ) for _ in range(bootstrap_iteration)
                    ])

                    self.ar_orient, boot_weight = self.ar_orient.T
                    self.ar_orient %= 360

                else:
                    self.ar_orient = np.array([
                        np.rad2deg(
                            circstats.circmean(
                                np.deg2rad(np.random.choice(self.ar_orient, size=len(self.ar_orient))),
                            )
                        ) % 360 for _ in range(bootstrap_iteration)
                    ])

                ## Average, Variance
                self.circmean, self.circvar = angle_stats(self.ar_orient, weights=boot_weight) 

                ## Median 
                self.median = circMedian(self.ar_orient, weights=boot_weight)
                
                ## 2.5% (p1) 97.5% (p2) percentiles
                p1, p2 = np.quantile(
                    circdist(self.ar_orient, self.circmean),
                    q=[alpha_CI/2, 1-alpha_CI/2]
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
            self.CI1_vonMises = np.rad2deg(stats.vonmises.ppf(1.-alpha_CI/2, kappa=self.kappa))
            self.std2 = np.std(circdist(self.ar_orient, self.arcmean, deg=True))
            self.std3 = takagi_error(self.ar_orient, boot_weight)
        
        except _OrientSingleError:
            pass
        
    def __str__(self):
        _str = f'{self.name}, mean={self.circmean:5.1f}, median={self.median:5.1f}, std={self.std1:5.1f}'
        return _str
    
    
    
    
###############################################################################

## Exception classess

class BadEpicDist(Exception):
    pass

class _OrientSingleError(Exception):
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