"""
A class for circular statistics (multiple stations) and plotting results

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

from pathlib import Path
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal, stats

from .orientation import OrientSingle, PvalueError
from .rstats import circdist
from .params import (
    colormap_cc,
    kwargs_plot_for_legend,
    dict_OBStypes,
    get_cbar_bound_norm,
)


class OrientAnalysis():
    """Perform circular statistics for orientations.
    A class for grouping `OrientSingle` objects.

    Attributes:
    -----------
    if_selection: bool
        Whether to perform bootstrap resampling
    df_chtbl: pd.DataFrame
        A result dataframe by `OrientOBS`
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
    The example uses multiple stations whose names are `stationAX`.

    >>> import pandas as pd
    >>> from azimpy import OrientAnalysis, read_chtbl

    ## Init params
    >>> min_CC = 0.5
    >>> alpha_CI = 0.05  ## 100(1-a)% CI
    >>> bootstrap_iteration = 5000

    >>> stationList = ['stationA1','stationA2','stationA3','stationA4']

    ## Channeltable including above stations' info
    >>> df_chtbl = read_chtbl('/path/to/channeltable.txt')
    >>> df_chtbl = df_chtbl.query('comp.str.endswith("U")')

    ## Init OrientAnalysis for circular statistics
    >>> oa_raw = OrientAnalysis(
    ... if_selection=False,  # w/o bootstrap analysis
    ... df_chtbl=df_chtbl, min_CC=min_CC)
    >>> oa_boot = OrientAnalysis(
    ... if_selection=True,  # perform bootstrap analysis
    ... df_chtbl=df_chtbl, 
    ... min_CC=min_CC, alpha_CI=alpha_CI, 
    ... bootstrap_iteration=bootstrap_iteration)

    >>> for stationName in stationList:
    ...     period = df_chtbl.at[stationName,'period']
    ...     df_orient = pd.read_pickle(
    ...         f'/path/to/output/{stationName}/{stationName}_020_040.pickle')
    ...     ## Add the dataframe in `oa_raw`
    ...     ## This is actually passed to `OrientSingle`
    ...     oa_raw.add_station(
    ...         df_orient, stationName, 
    ...         period=period)
    ...     ## Add the dataframe in `oa_boot`
    ...     oa_boot.add_station(
    ...         df_orient, stationName, 
    ...         period=period)
    
    >>> ## Make a plot
    >>> fig1 = oa_raw.plot()
    >>> fig2 = oa_boot.plot()
    """

    def __init__(
        self, 
        if_selection=True, 
        df_chtbl=None, 
        min_CC=0.5, 
        weight_CC=True,
        bootstrap_iteration:int =5000, 
        bootstrap_fraction_in_percent:float =100.,
        alpha_CI=0.05, 
        K=5.0,        
        kuiper_level=0.05,
        only_good_stations=True,
        dict_OBStypes=dict_OBStypes,
    ):
        """Inits OrientAnalysis"""
        self.stations = []
        self.if_selection = if_selection
        self.df_chtbl = df_chtbl

        self._kw_orientsingle = dict(
            min_CC=min_CC, 
            weight_CC=weight_CC, 
            K=K, 
            bootstrap_iteration=bootstrap_iteration, 
            bootstrap_fraction_in_percent=bootstrap_fraction_in_percent,
            alpha_CI=alpha_CI,
            kuiper_level=kuiper_level,
        )
        
        self.min_CC = min_CC
        self.alpha_CI = (alpha_CI, stats.norm.ppf(1-alpha_CI/2))
        self.only_good_stations = only_good_stations
        self.dict_OBStypes = dict_OBStypes
        
    def __len__(self):
        _number_of_good_stations = len([
            sta for sta in self.stations 
            if (not self.only_good_stations) or sta._goodstation
        ])
        return _number_of_good_stations
    
    def __repr__(self):
        _str = (
            f'{len(self)} stations ({["un",""][self.if_selection]}"selected") are listed:' 
            + ''.join(['\n\t'+str(sta) for sta in self.stations if (not self.only_good_stations) or sta._goodstation])
        )
        return _str
        
    def add_station(self, df_orient=None, stationname=None, orientsingle_path=None, period=None):
        """Perform circular statistics for each station.
        The object will be appended in the list `.stations`.

        Parameters:
        -----------
        df_orient: pd.DataFrame
            A result dataframe by `OrientOBS`
        stationname: str
            A station name for the result dataframe
        orientsingle_path: str, pathlib.Path
            Give an orientsingle object instead of using df_orient.
            If not None, `df_orient` and `stationname` are ignored.
        """

        in_parentheses = self.dict_OBStypes.get(period)

        if orientsingle_path is not None:
            orientsingle = pd.read_pickle(orientsingle_path)
        else:
            if (df_orient is None) or (stationname is None):
                raise ValueError('Both df_orient and stationname must be given.')
        
            try:
                orientsingle = OrientSingle(
                    df_orient, stationname, 
                    self.if_selection, 
                    in_parentheses=in_parentheses,
                    **self._kw_orientsingle,
                )
            except IndexError:
                pass
            except PvalueError:
                pass

        self.stations.append(orientsingle)

        
    def plot(self, polar=True, fig=None):
        """Plot the estimated results w/ or w/o bootstrap.
        Note that `plt.show()` may be required when viewing the outputs.

        Parameters:
        -----------
        polar: bool, defaults to True
            Whether to make a polar plot.
            If `if_selection` == False, `polar` is set to False.
        fig: `matplotlib.figure.Figure` or `matplotlib.figure.SubFigure`, None
            A figure on which the results are drawn.
            If None, a new figure will be created.

        Returns:
        -----------
        `matplotlib.figure.Figure` or `matplotlib.figure.SubFigure`
            A figure object with the results from multiple stations.
        """
        
        ncols = 3 #if polar else 2
        nrows = -(-len(self)//ncols)
        
        if self.if_selection:
            polar = False
        
        if not fig:
            unit_figsize = [
                np.array([3,2]),
                np.array([3,3.2]),
            ][polar]
            fig = plt.figure(
                figsize=np.array([ncols,nrows])*unit_figsize
            )
    
        subfigs = fig.subfigures(nrows=nrows, ncols=ncols)
        
        # arr_theta_axis = np.linspace(-np.pi, np.pi, 1000)
        label_ij = [0] * 2
        
        bound_cc, norm_cc = get_cbar_bound_norm(min_CC=self.min_CC)
        
        list_of_orientsingle = [
            sta for sta in self.stations 
            if (not self.only_good_stations) or sta._goodstation
        ]
        ## Plot for each subfigure           
        for ij, subfig in np.ndenumerate(subfigs):
            
            try:
                i, j = ij
                k = ncols * i + j
            except ValueError:
                j, = ij
                k = j
                
            try:
                orientsingle = list_of_orientsingle[k]
            except IndexError:
                # ax.axis('off')
                continue
                
            orientsingle.plot(
                polar=polar,
                fig=subfig,
                # in_parentheses=obs_type,
                add_cbar=False, 
                ax_colorbar=None,
            )

            ax = subfig.axes[0]
            ax.set(xlabel='')

        
        ## Setting of entire figure
        fig.supxlabel([
            "$H_1$ azimuth [$\degree$]",
            "Azimuthal deviation from circular mean [$\degree$]"
        ][self.if_selection])
        if not polar:
            fig.supylabel('Density')
        
        ## Plot legend and colorbar
        
        # Legend
        subfig = subfigs[-1,-1]
        if len(self) % ncols != 0: 
            ax = subfig.subplots()
            [spine.set_visible(False) for spine in ax.spines.values()]
            [getattr(ax.axes,f"get_{xy}axis")().set_visible(False) for xy in 'xy']
            
            if self.if_selection:
                labels = ["median","vonmises"]
            else:
                labels = ["mean","median"]
            for label in labels:
                ax.plot([],[],**kwargs_plot_for_legend[label])
        
            subfig.legend(
                loc="upper center",
                bbox_to_anchor=(0.55,0.98),
                fontsize='small'
            )    

        else:
            subfig.legend(
                loc="upper center",
                bbox_to_anchor=(0.6,-0.0),
                fontsize='small'
            ) 

        # colorbar
        if not self.if_selection:
            if len(self) % ncols != 0:
                ax_colorbar = subfig.add_axes([0.3,0.56,0.5,0.04])
            else:
                subfig = subfigs[-1,0]
                ax_colorbar = subfig.add_axes([0.6,-0.1,0.5,0.04])
            
            subfig.colorbar(
                plt.cm.ScalarMappable(cmap=colormap_cc, norm=norm_cc),
                cax=ax_colorbar, ticks=bound_cc, pad=0.01,
                spacing='proportional', orientation='horizontal',
                #label='${C}_{UR}$'
            )
            ax_colorbar.set_title('${C}_{\~{Z}R}$',fontsize='small')
            ax_colorbar.tick_params(
                which='both', labelsize='small', direction='in', 
                top=True, bottom=False, labeltop=True, labelbottom=False
            )

        return fig
    
    def write(self, filename=None, networkname=None, format='pickle'):
        """Output DataFrame of results and a save pickle file (optional).
        
        Parameters:
        -----------
        filename: str, pathlib.Path
            A name of the file to save. 
            If None, just return Dataframe (default to None).
        networkname: str
            A name of network in which you perform orientation analysis.
        format: str, default to 'pickle'
            A file format. You can choose it below: 
            - ['pickle','csv','json']
        """
        
        df_analysis = pd.DataFrame(
            np.array([
                [
                    sta.name, networkname, sta.circmean, 
                    sta.median, sta.MAD, sta.num_eq,
                    sta.kappa, sta.CI1_vonMises, sta.CI, sta.p1, sta.p2, 
                    sta.std1, sta.arcmean, sta.std2, sta.std3, #sta.CMAD
                ] 
                for sta in self.stations
            ]),
            columns=(
                'station','network','circular mean',
                'circular median', 'MAD','numeq',
                'kappa', 'Half 95%CI', '95%CI', 
                '2.5% percentile', '97.5% percentile', 
                'CSE', 'arc mean', 'arc SE', 'SE_Takagi_et_al', #'CMAD',
            )
        ).astype({
            'circular mean':float, 'circular median':float, 
            'MAD':float, 'numeq':int,
            'kappa':float, '95%CI':float, 
            '2.5% percentile':float, '97.5% percentile':float,
            'Half 95%CI':float, 'CSE':float, 'arc mean':float, 'arc SE':float, 
            'SE_Takagi_et_al':float, #'CMAD':float, 
        }).round({
            'circular mean':2, 'circular median':2, 'MAD':2, 'kappa':1,
            '95%CI':2, '2.5% percentile':2, '97.5% percentile':2,
            'Half 95%CI':2, 'CSE':2, 'arc mean':2, 'arc SE':2, 
            'SE_Takagi_et_al':2
        }).set_index('station')
        #df_analysis.kappa = df_analysis.kappa.map(lambda x: '{0:.3e}'.format(x))

        list_station = [sta.name for sta in self.stations]
        df_analysis = pd.concat(
            [
                df_analysis,
                self.df_chtbl[
                    ['period','latitude','longitude','elevation']
                ].query(f'station in {list_station}')
            ], 
            axis=1
        )  #.reset_index()
        
        if filename is not None:
            if format == 'pickle':
                df_analysis.to_pickle(filename)
            elif format == 'json':
                df_analysis.to_json(filename)
            else:  ## format == 'csv'
                df_analysis.to_csv(filename)
        
        return df_analysis
    

def plotCC(df, center_lonlat, min_CC=0.5, figtitle=None, show=False, **fig_kw):
    """Plot event distribution with cross correlation.
    Specify a result dataframe to the argument `df`.
    
    `cartopy` is required.
    
    Parameters:
    -----------
    df: `pandas.DataFrame`
        Specify a DataFrame with the orientation result for each event
    center_lonlat: list
        A list of longitude and latitude for an AzimuthalEquidistant plot.
        For instance, `center_lonlat=[132.0,31.0]` is the central location with
        the longitude of 132.0°E and the latitude of 31.0°N.
    min_CC: float, default to 0.5
        A minimum value of cross correlation.
    figtitle: str, None
        A figure title at the top
    show: bool, default to False
        plt.show()
    fig_kw: kwargs passed to `plt.figure()`
    """
    
    import cartopy.crs as ccrs
    
    df = df.sort_values('CC').dropna(how='any')  ## cartopy would plot even NaN!
    
    fig = plt.figure(**fig_kw)
    fig.suptitle(figtitle)
    
    ## Colormap settings
    bound_cc, norm_cc = get_cbar_bound_norm(min_CC=min_CC)
    
    ## BAZ vs. CC
    ax_CC = fig.add_subplot(2,2,2, projection="polar")
    ax_CC.set(
        theta_zero_location="N", theta_direction=-1, rlabel_position=330, 
        ylim=(0,1), yticks=(0.5, 0.7, 0.9, ), 
        xlabel="Event back azimuth [$\degree$]", #ylabel="$C_{\~{Z}R}$",
    )

    ax_CC.scatter(
        np.deg2rad(df.baz), df.CC,
        c=colormap_cc(norm_cc(df.CC)), alpha=0.85, edgecolors='none',
    )
    
    ## BAZ vs. Orientation
    ax_or = fig.add_subplot(2,2,(1,3))
    ax_or.set(
        xlim=(0,360),ylim=(0,360),
        xticks=range(0,361,45), yticks=range(0,361,45), 
        xlabel="Event back azimuth [$\degree$]", ylabel="Estimated $H_1$ azimth [$\degree$]"
    )
    
    ax_or.scatter(
        df.baz, df.orientation,
        c=colormap_cc(norm_cc(df.CC)), alpha=0.85, edgecolors='none',
    )
    
    ## Map view
    # mag2rad = lambda x: 80*(x-5.8)
    pc = ccrs.PlateCarree()
    proj_ortho = ccrs.AzimuthalEquidistant(
        central_latitude=center_lonlat[1],
        central_longitude=center_lonlat[0],
    )
    ax_map = fig.add_subplot(2,2,4, projection=proj_ortho)
    
    ax_map.set_global() 
    ax_map.coastlines(resolution='110m', lw=0.35)
    
    ax_map.scatter(
        df.longitude, df.latitude,
        c=colormap_cc(norm_cc(df.CC)), edgecolors='none',
#         s=mag2rad(df.mag), linewidths=0.0, edgecolors='face', alpha=0.7,
        alpha=0.85,
        transform=pc, zorder=10
    )
    
    ## Colorbar
    ax_colorbar = fig.add_axes([0.07, 0.965, 0.20, 0.02]) # fig.add_subplot(gs_events[-4,0:3])
    colorbar = fig.colorbar(
        mappable=plt.cm.ScalarMappable(cmap=colormap_cc, norm=norm_cc), 
        ticks=bound_cc,
        cax=ax_colorbar, 
        # label='${C}_{\~{Z}R}$', 
        orientation='horizontal',
        spacing='proportional', 
        pad=0.01, 
    )
    ax_colorbar.set_title('${C}_{\~{Z}R}$',fontsize='small')
    ax_colorbar.tick_params(
        which='both', direction='in', labelsize='small', 
        top=True, bottom=False, labeltop=True, labelbottom=False
    )
    
#     colorbar.set_label('${C}_{UR}$', loc='top', fontsize=12)
    # ax_colorbar.xaxis.set_ticks_position("top")

    if show:
        plt.show()
    else:
        return fig    