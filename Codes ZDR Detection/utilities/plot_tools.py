#! /usr/bin/env python

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from contextlib import contextmanager
import wradlib

@contextmanager
def penv(nrows=1, ncols=1, subplots_dict=dict(squeeze=False), close=True, show=True, _inchs_per_col=6, _inchs_per_row=4):
    """
    An advanced standard environment for plotting commands.
    
    Parameters:
        nrows:            int, number of rows. Default: 1
        ncols:            int, number of columns. Default: 1
        subplots_dict:    dict, used to override plt.subplots parameters. Note: 'squeeze' is set here to False by default.
        close:            bool, if set to True all other figures are closed before. Default: True
        show:             bool, if set to True plt.show is executed afterwards. Default: True
    
    Simple usage:
        with penv():
            # plotting commands, e.g.:
            plt.plot([1,2,3], [0,1,0]);

        This is equivalent to
        plt.close('all');
        plt.plot([1,2,3], [0,1,0]);
        plt.show();
    
    Advanced usage:
        with penv(nrows=2, ncols=1) as (fig, ax_arr):
            ax_arr[0,0].plot([1,2,3], [0,1,0]);

        This is equivalent to
        plt.close('all');
        fig, ax_arr = plt.subplots(nrows=2, ncols=1)
        fig.set_figwidth(ncols * 6);
        fig.set_figheight(nrows * 4);
        ax_arr[0,0].plot([1,2,3], [0,1,0]);
        plt.show();
    """
    if close:
        plt.close('all');
    #
    _subplots_dict = dict(
        squeeze = False,
    )
    _subplots_dict.update(subplots_dict)
    #
    fig, ax_arr = plt.subplots(nrows, ncols, **_subplots_dict)
    fig.set_figwidth(ncols * _inchs_per_col);
    fig.set_figheight(nrows * _inchs_per_row);
    yield (fig, ax_arr)
    if show:
        plt.show();
    # prevent memory leak
    if close:
        fig.clf()
        plt.close(fig)

#

def _make_ppi_plot(ppi_data, metadata, scan_name, var, ax, lut=15, do_not_mask_lowest_values=False):
    """
    Makes a PPI plot for radar data.
    
    Arguments:
        ppi_data:                   array, data to be plotted as PPI. Azimuth is expected as first, and range as second index.
        metadata:                   dict, contains the metadata of the radar-sweep.
        scan_name:                  str, name of the scan (determines the elevation).
        var:                        str, name of variable to plot.
        ax:                         matplotlib.axes object, axis to plot in.
    Parameters:
        lut:                        int, number of colors to use. Default: 15
        do_not_mask_lowest_values:  bool, disable masking of values which are equal to min(ppi_data). Default: False
    Returns:
        ax:                 matplotlib.axes-object, axis, which was used for plotting.
        pm:                 matplotlib.pcolormesh-object, the colored mesh, containing the plotted variable.
    """
    #
    var_props_dict = {
        u'AC_ZH' : {
	        'name' : "Attenuation-Corrected hor. Reflectivity (dBZ)",
            'vmin' : -0.5,
            'vmax' : 65.5,
            'cmap' : None, # default
        },
        u'A_h' : {            
            'name' : "Specific Attenuation (dB/km)",
            'vmin' : -0,
            'vmax' : 4,
            'cmap' : None, # default
        },
        u'alpha' : {
            'name' : "alpha (dB/deg)",
            'vmin' : 0.,
            'vmax' : 1.,
            'cmap' : "alpha",
        },
    }
    #
    assert var in var_props_dict
    var_props = var_props_dict[var]
    if var == 'alpha':
        register_alpha_colormap()
    #
    if do_not_mask_lowest_values:
        mask = np.isnan(ppi_data)
    else:
        mask = np.logical_or(np.isnan(ppi_data), ppi_data <= np.nanmin(ppi_data)) # mask lowest values, too.
    masked_data = np.ma.array(ppi_data, mask=mask)
    # modify colormap a bit to plot NaN in gray
    cmap = plt.cm.get_cmap(var_props['cmap'], lut=lut)
    if isinstance(cmap, matplotlib.colors.ListedColormap): # these colormaps stay and do not become discrete
        cmap = make_cmap_lutable(cmap, lut=lut)
        cmap.set_bad('gainsboro', 1.0)
    # plot PPI
    ax, pm = wradlib.vis.plot_ppi(
        ppi_data,
        r=metadata[scan_name]['r']/1e3,
        az=metadata[scan_name]['az'],
        elev=metadata[scan_name]['el'],
        ax=ax,
        vmin=var_props['vmin'],
        vmax=var_props['vmax'],
        cmap=cmap,
    )
    ax.set_xlabel("Zonal Distance (km)")
    ax.set_ylabel("Meridional Distance (km)")
    ax_ch = wradlib.vis.plot_ppi_crosshair((0,0,0), ranges=[50,100,150], ax=ax)
    # add a colorbar with label
    cbar = plt.colorbar(pm, ax=ax, shrink=1.0, extend='neither')
    cbar.set_label(var_props['name'])
    return ax, pm

def make_cmap_lutable(cmap, lut=15):
    """Converts continous colormaps to colormaps with a discrete number of colors."""
    it = np.linspace(0, cmap.N-1, lut, dtype=int)
    _a = cmap.colors[it]
    cmap = plt.cm.jet.from_list(cmap.name, _a, N=lut)
    return cmap

def register_alpha_colormap():
    """Adds a colormap, which is more suitable for plotting the attenuation coefficient alpha."""
    color_ranges = [
        # ((start, stop), (R,G,B)), # color/purpose
        ((0., 0.05), (0., 0., 0.)), # black
        ((0.05, 0.18), (0., 0., 1.0)), # dark blue
        ((0.1, 0.2), (0., 1., 0)), # green
        ((0.15, 0.4), (1.0, 0.66, 0.)), # orange
        ((0.25, 0.7), (1.0, 0.0, 0.0)), # dark red
        ((0.7, 1.0), (0.5, 0.0, 0.5)), # purple
    ]
    c_cmap = build_colormap(color_ranges, 'alpha')
    plt.register_cmap(cmap=c_cmap)
#
