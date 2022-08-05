
import sys
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
from functools import reduce, partial

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from matplotlib.ticker import AutoMinorLocator, PercentFormatter
# import matplotlib.path as path
# import matplotlib.patches as patches
import matplotlib.colors as mcolors
import pandas as pd
from scipy import signal, stats

from .orientation import OrientSingle, PvalueError
from .rstats import circdist
from .params import set_rcparams


## update rcParams
set_rcparams()


class OrientAnalysis():
    def __init__(
        self, 
        if_selection=True, 
        df_chtbl=None, 
        min_CC=0.5, 
        alpha_CI=0.05, 
        K=5.0,
        only_good_stations=True,
    ):
        
        self.stations = []
        self.if_selection = if_selection
        self.df_chtbl = df_chtbl
        
        self.min_CC = min_CC
        self.alpha_CI = (alpha_CI, stats.norm.ppf(1-alpha_CI/2))
        self.K = K
        self.only_good_stations = only_good_stations
        
    def __len__(self):
        return len([sta for sta in self.stations if (not self.only_good_stations) or sta._goodstation])
    
    def __repr__(self):
        _str = (
            f'{len(self)} stations ({["un",""][self.if_selection]}"selected") are listed:' 
            + ''.join(['\n\t'+str(sta) for sta in self.stations if (not self.only_good_stations) or sta._goodstation])
        )
        return _str
        
    def add_station(self, df_orient, stationname, **kwargs):
        
        try:
            orientsingle = OrientSingle(
                df_orient, stationname, 
                self.if_selection, 
                min_CC=self.min_CC,
                alpha_CI=self.alpha_CI[0],
                K=self.K,
                **kwargs,
            )
            self.stations.append(orientsingle)

        except IndexError:
            pass
        except PvalueError:
            pass
        
    def plot(self, polar=True, fig=None):
        
        ncols = 3 #if polar else 2
        
        if self.if_selection:
            polar = False
        
        if not fig:
            fig = plt.figure(
                figsize=np.array([ncols,-(-len(self)//ncols)])*[
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
        
        if polar:
            axs = fig.subplots(nrows=-(-len(self)//ncols), ncols=ncols, subplot_kw=dict(projection='polar'))
        else: 
            axs = fig.subplots(nrows=-(-len(self)//ncols), ncols=ncols)
        
        xxx = np.linspace(-np.pi, np.pi, 1000)
        label_ij = [0] * 2
        
        colormap_cc = mcolors.ListedColormap(
            ("gray","lightsalmon","darkorange","red"), N=256
        )
        bound_cc = np.array([self.min_CC,0.7,0.9, 1.0])
        norm_cc = mcolors.BoundaryNorm(bound_cc, ncolors=256, extend='min')
        
        ## Plot for each axes           
        for ij, ax in np.ndenumerate(axs):
            
            try:
                i, j = ij
                k = ncols * i + j
            except ValueError:
                j, = ij
                k = j
                
            try:
                orientsingle = [sta for sta in self.stations if (not self.only_good_stations) or sta._goodstation][k]
            except IndexError:
                ax.axis('off')
                continue
                
            _period = self.df_chtbl.at[orientsingle.name,'period']
            if _period == 360:
                obs_type = 'BB'
            elif _period == 20:
                obs_type = '20s'
            elif _period == 120:
                obs_type = '120s'
            else:
                obs_type = '1s' 
                
            ax.set_title(f'{orientsingle.name} ({obs_type})', fontdict={'fontsize':13})
            
            if polar:

                _counts, _bins = np.histogram(
                    np.deg2rad(orientsingle.ar_orient-self.if_selection*orientsingle.circmean),
                    bins=np.deg2rad(np.arange(theta_range[0], theta_range[1]+0.01, bin_interval)),
                )
                
                freqmax = -(-np.max(_counts)/np.sum(_counts)//0.01)
                
                if self.if_selection:
                    rticks = np.arange(1,freqmax,freqmax//3) * 0.01
                else:
                    rticks = np.hstack([
                        np.arange(1,4,1),
                        np.arange(5,freqmax+6,5)
                    ]) * 0.01
                    
                ax.set(
                    theta_zero_location="N", theta_direction=-1, 
                    xticks=np.deg2rad([
                        np.arange(0,360,30),
                        np.arange(theta_range[0],theta_range[1]+1, 5)
                    ][self.if_selection]),
                    rorigin=rorigin,
                    ylim=(0,np.sqrt((freqmax+6)*0.01)),
                    yticks=np.sqrt(rticks), yticklabels=[f'{rtk*100:.0f}%' for rtk in rticks],
                    thetamin=theta_range[0], thetamax=theta_range[1]
                )
                ax.tick_params('x', labelsize=11)
                ax.tick_params('y', labelsize=9)
                ax.xaxis.set_minor_locator(AutoMinorLocator())

                ax.bar(
                    _bins[:-1], height=np.sqrt(_counts/np.sum(_counts)), 
                    width=np.deg2rad(bin_interval), alpha=0.8,
                )
                
                if not self.if_selection:
                    line_mean = ax.axvline(
                        x=np.deg2rad(orientsingle.circmean), 
                        c='darkgreen', linewidth=1.5, linestyle='dashed',
                        zorder=5,
                    )
                    
                    ## von Mises dist.
                    line1, = ax.plot(
                        xxx+np.pi, 
                        ax.get_rmax() * np.sqrt(stats.vonmises.pdf(
                            xxx-np.deg2rad(orientsingle.circmean-180), kappa=orientsingle.kappa
                        )),
                        c='darkgreen', zorder=5,
                    )
                    
                line_median = ax.axvline(
                    x=np.deg2rad(orientsingle.median-self.if_selection*orientsingle.circmean), 
                    label='Median', c='b', linewidth=2.0, linestyle='dotted', zorder=5,
                )
                
                               
            else:
                
                ax.yaxis.get_major_locator().set_params(integer=True)
                if j == 2:
                    ax.tick_params(labelright=True, labelleft=False)
                
                _counts, bins, _ = ax.hist(
                    [orientsingle.ar_orient, circdist(orientsingle.ar_orient, orientsingle.circmean)][self.if_selection],
#                     orientsingle.ar_orient-self.if_selection*orientsingle.circmean, 
                    range=theta_range, bins=int(sum(np.abs(theta_range))//bin_interval), 
                    density=self.if_selection,  alpha=0.8,
                )
                
                freqmax = -(-np.max(_counts)/np.sum(_counts)//0.01)
                
                ax.set(
                    xlim=theta_range, 
                    xticks=[
                        np.arange(0,361,90),
                        np.arange(theta_range[0],theta_range[1]+1, 5)
                    ][self.if_selection],
                    #ylim=(0,)
                )  
                
                ax.tick_params(labelsize=11)
                ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=False))
                
                if self.if_selection:
                    
                    ax.xaxis.set_minor_locator(AutoMinorLocator(6))
                    
                    ## von Mises dist.
                    try:
                        vonmises_dist = stats.vonmises.pdf(xxx, kappa=orientsingle.kappa)
                        line1, = ax.plot(
                            np.rad2deg(xxx)+circdist(
                                np.rad2deg(orientsingle._mean_vonMises)+180, 
                                orientsingle.circmean
                            ), 
                            ax.get_ylim()[-1] * vonmises_dist / vonmises_dist.max(), 
                            c='darkgreen', zorder=5,
                        )
                    except AttributeError:
                        pass
                else:
                    
                    ax.xaxis.set_minor_locator(AutoMinorLocator(6))
                    
                    line_mean = ax.axvline(
                        x=orientsingle.circmean, 
                        c='darkgreen', linewidth=1.5, linestyle='dashed',
                        zorder=5,
                    )
                    ## von Mises dist.
                    line1, = ax.plot(
                        np.rad2deg(xxx)+180, 
                        ax.get_ylim()[-1] * stats.vonmises.pdf(
                            xxx-np.deg2rad(orientsingle.circmean-180), kappa=orientsingle.kappa
                        ), 
                        c='darkgreen', zorder=5,
                    )
                    
                line_median = ax.axvline(
                    x=orientsingle.median-self.if_selection*orientsingle.circmean, 
                    label='Median', c='b', linewidth=2.0, linestyle='dotted', zorder=5,
                )

        
            ## Texts for `ax`
            ax.text(
                *[(-0.08,1.06),(-0.15,1.08)][polar], 
                f" $\mu$:  {orientsingle.circmean:5.1f}$\degree$\n"+
                f"$\mu_m$: {orientsingle.median:5.1f}$\degree$", 
                fontsize=8, va='bottom', ha='left', transform=ax.transAxes,
            )
            
            
            
            if self.if_selection:
                
                ax.text(
                    1.13, 1.06, 
                    f"{int((1-self.alpha_CI[0])*100)}%CI: {orientsingle.CI1_vonMises:4.1f}$\degree$\n"+
                    f"$\kappa$: {orientsingle.kappa:.2E} ", 
                    fontsize=8, va='bottom', ha='right', transform=ax.transAxes,
                )
            
            else:
                
                ## Scatter: (azimuth, CC)
                if polar:
                    
                    ## modified due to upgrade of matplotlib ver. 3.3
#                     _mappable = ax.scatter(
#                         np.deg2rad(orientsingle.ar_orient), 
#                         orientsingle.ar_cc*ax.get_rmax(), 
#                         s=100, c=orientsingle.ar_cc, alpha=0.8,
#                         marker='.', linewidth=0.6, 
#                         edgecolor='face', facecolor='none', 
#                         norm=norm_cc, cmap=colormap_cc, zorder=4., 
#                     )
                    _mappable = ax.scatter(
                        np.deg2rad(orientsingle.ar_orient), 
                        orientsingle.ar_cc*ax.get_rmax(), 
                        s=30, alpha=0.8,
                        marker='o', linewidth=0.6, 
                        edgecolor=colormap_cc(norm_cc(orientsingle.ar_cc)), 
                        facecolor='none', 
                        zorder=4., 
                    )
                else:
                    _mappable = ax.scatter(
                        orientsingle.ar_orient, 
                        orientsingle.ar_cc*ax.get_ylim()[1], 
                        s=100, c=orientsingle.ar_cc, alpha=0.8,
                        marker='.', linewidth=0.6, 
                        edgecolor='face', facecolor='none', 
                        norm=norm_cc, cmap=colormap_cc, zorder=4., 
                    )
                _mappable.set_facecolor('none')
                
                ax.text(
                    *[(1.06,1.06),(1.15,1.08)][polar],
                    f"MAD:{orientsingle.MAD:5.1f}$\degree$",
                    fontsize=8, va='bottom', ha='right', transform=ax.transAxes,
                )
                              
        
        ## Setting of entire figure
        fig.text(
            0.5, 0.0, 
            ["Orientation azimuth [$\degree$]","Azimuthal deviation from circular mean [$\degree$]"][self.if_selection],
            ha='center', va='top'
        )
        if not polar:
            fig.text(0.0, 0.5, 'Density', ha='right', va='center', rotation=90)  #'Frequency'
            fig.text(1.0, 0.5, 'Density', ha='left', va='center', rotation=-90)
        
        if self.if_selection:
            ## Legend 
            fig.legend(
                (line_median, line1), 
                ('Circular median $\mu_m$','von Mises distribution'), 
                bbox_to_anchor=(1.,1.02), 
                loc=1, fontsize=10
            )    
        else:            
            ax_colorbar = fig.add_axes([
                [0.08, 0.96, 0.2, 0.015],
                [0.08, 0.978, 0.2, 0.012]
            ][polar])
            fig.colorbar(
                plt.cm.ScalarMappable(cmap=colormap_cc, norm=norm_cc),
                cax=ax_colorbar, ticks=bound_cc, pad=0.01,
                spacing='proportional', orientation='horizontal',
                #label='${C}_{UR}$'
            )
            ax_colorbar.set_title('${C}_{UR}$',fontsize=11)
            ax_colorbar.tick_params(
                which='both', labelsize=10, direction='in', 
                top=True, bottom=False, labeltop=True, labelbottom=False
            )
            
            ## Legend 
            fig.legend(
                (line_median, line_mean, line1), 
                ('Circular median $\mu_m$','Circular mean $\mu$','von Mises distribution'), 
                bbox_to_anchor=[(1.,1.07),(1.,1.08)][polar], 
                loc=1, fontsize=10
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
                'kappa', 'Half 95%CI', '95%CI', '2.5% percentile', '97.5% percentile', 
                'CSE', 'arc mean', 'arc SE', 'SE_Takagi_et_al', #'CMAD',
            )
        ).astype({
            'circular mean':float, 'circular median':float, 
            'MAD':float, 'numeq':int,
            'kappa':float, '95%CI':float, '2.5% percentile':float, '97.5% percentile':float,
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
        
        if filename:
            if format == 'pickle':
                df_analysis.to_pickle(filename)
            elif format == 'json':
                df_analysis.to_json(filename)
            else:  ## format == 'csv'
                df_analysis.to_csv(filename)
        
        return df_analysis
    

def plotCC(df, figtitle=None, center_lonlat=(132,32), min_CC=0.5, show=False, **fig_kw):
    """Plot event distribution with cross correlation.
    Specify a result dataframe to the argument `df`.
    
    `cartopy` is required.
    
    Parameters:
    -----------
    df: `pandas.DataFrame`
        Specify a DataFrame with the orientation result for each event
    figtitle: str
        A figure title at the top
    center_lonlat: list
        A list of longitude and latitude for an AzimuthalEquidistant plot
    min_CC: float, default to 0.5
        A minimum of cross correlation
    show: bool, default to False
        plt.show()
    fig_kw: kwargs passed to `plt.figure()`
    """
    
    import cartopy.crs as ccrs
    
    df = df.sort_values('CC').dropna(how='any')  ## cartopy would plot even NaN!
    
    fig = plt.figure(**fig_kw)
    fig.suptitle(figtitle)
    
    ## Colormap settings
    colormap_cc = mcolors.ListedColormap(
        ("gray","lightsalmon","darkorange","red"), N=256
    )
    bound_cc = np.array([min_CC,0.7,0.9,1.0])
    norm_cc = mcolors.BoundaryNorm(bound_cc, ncolors=256, extend='min')
    
    ## BAZ vs. CC
    ax_CC = fig.add_subplot(2,2,2, projection="polar")
    ax_CC.set(
        theta_zero_location="N", theta_direction=-1, rlabel_position=330, 
        ylim=(0,1), yticks=(0.5, 0.7, 0.9, ), 
        xlabel="BAZ [degrees]", #ylabel="$C_{\hat{Z}R}$",
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
        xlabel="BAZ [degrees]", ylabel="Estimated orientation [degrees]"
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
    ax_colorbar = fig.add_axes([0.07, 0.955, 0.20, 0.01]) # fig.add_subplot(gs_events[-4,0:3])
    colorbar = fig.colorbar(
        mappable=plt.cm.ScalarMappable(cmap=colormap_cc, norm=norm_cc), 
        ticks=bound_cc,
        cax=ax_colorbar, 
        label='${C}_{UR}$', 
        orientation='horizontal',
        spacing='proportional', 
        pad=0.01, 
    )
    
#     colorbar.set_label('${C}_{UR}$', loc='top', fontsize=12)
    ax_colorbar.xaxis.set_ticks_position("top")

    if show:
        plt.show()
    else:
        return fig    