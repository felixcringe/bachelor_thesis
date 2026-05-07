#! /usr/bin/env python
# -*- coding: utf8 -*-

# common
import os
import datetime

# KONRAD
import tarfile
import xml.etree.cElementTree as ET

# data
import tempfile
import h5py
import itertools # required for parallel iteration using izip

# plotting
import matplotlib # ...and for point-in-contour checks
matplotlib.use('agg') # allow plotting e.g. without DISPLAY; needs to be loaded first
import matplotlib.pyplot as plt

# radar
import wradlib
import numpy as np # at least for loading stored data
# import cPickle as pickle # for reading stored radar data
import pickle

from utilities import network_info, radar_tools # for station info and metadata handling
from utilities import hail_precursor_tools # for ZDR column detection
from utilities import att_cor_tools # for volumetric alpha calculation and attenuation correction
from utilities import utils # ACE requires

#

from utilities.plot_tools import penv
from utilities import plot_tools

# ----


# paths
konrad_base_path = "./data/"#"/automount/ags/dwd_radar/HAILSIZE/tracking_KONRAD/"
plot_save_path = "./plots/"#"/user/mschmidt/plots/hailstorm_inspect/NowcastingSuite/track_and_forecast/realtime_update/standalone/"

# ----


# KONRAD: file accessors and XML readers
class TarAccessor(object): # greatly speeding up accessing multiple files in tar
    """
    Object enabling access to content in a tar-file.
    """
    def __init__(self, tar_filepath, mode='r:gz', _disable_file_checking=False):
        self._tar_fo = tarfile.open(tar_filepath, mode)
        self._disable_file_checking = _disable_file_checking
        if not self._disable_file_checking:
            self._tar_member_names = []
            for member in self._tar_fo: # this will speed up later accessing
                self._tar_member_names.append(member.name)
    def access_file(self, fpath_in_tar):
        if not self._disable_file_checking: assert fpath_in_tar in self._tar_member_names
        return self._tar_fo.extractfile(self._tar_fo.getmember(fpath_in_tar))
    def __del__(self):
        self._tar_fo.close()
class XMLtreeObtainer(object):
    """
    Requires a TarAccessor and allows to obtain a xml_tree for the tar-content.
    """
    def __init__(self, tar_access):
        assert type(tar_access) == TarAccessor
        self.tar_access = tar_access
    def obtain_xml_tree(self, fpath_in_tar):
        fo = self.tar_access.access_file(fpath_in_tar)
        xml_tree = ET.parse(fo)
        fo.close()
        return xml_tree


def _dwd_tstr2dt(tstr):
    return datetime.datetime.strptime(tstr, '%Y-%m-%dT%H:%M:%SZ')
def obtain_tracking_infos(xml_tree, cells=dict()):
    """Obtain relevant tracking infos from KONRAD3D xml_tree."""
    for feat in xml_tree.getroot().findall('./cells/feature'):
        cell_id = feat.attrib['identifier']
        #if cell_id not in cells: cells[cell_id] = Tracked_Cell(len(cells))
        if cell_id not in cells: cells[cell_id] = Tracked_Cell(cell_id)
        ref_dt = _dwd_tstr2dt(feat.find('./metadata/reference_time').text) #ref_dt = datetime.datetime.strptime(feat.find('./metadata/reference_time').text, '%Y-%m-%dT%H:%M:%SZ')
        cells[cell_id].add_local_id(ref_dt, cell_id)
        lats = map(float, feat.find('./geometry/polygons_projected/geodetic_coordinates/polygon/latitudes').text.split())
        lons = map(float, feat.find('./geometry/polygons_projected/geodetic_coordinates/polygon/longitudes').text.split())
        cells[cell_id].add_contour(ref_dt, lons, lats)
        # find predeccessors and split/merge infos
        predecessors = feat.findall('./tracking/predecessors/predecessor')
        _merge_ev = feat.find('./tracking/merge_event')
        _split_ev = feat.find('./tracking/splits/split_event')
        # handle split events
        if _split_ev.text == 'true':
            # find out, from which predeccessors the split came
            for pred in predecessors:
                if pred.find('probability').text == '100.00':
                    _splitted_from_cell = pred.find('identifier').text
                    break
            else:
                raise ValueError("ERROR: Couldn't find a predecessor to split cell '%s' from!" % (cell_id))
            #
            cells[cell_id].add_split_event( _dwd_tstr2dt(feat.find('./tracking/splits/reference_time_last_split').text), _splitted_from_cell)
        # handle merges
        if _merge_ev.text == 'true':
            # find out which cells have been absorbed
            _absorbed_cells = []
            for pred in predecessors:
                if pred.find('probability').text != '100.00':
                    _absorbed_cells.append(pred.find('identifier').text)
            assert len(_absorbed_cells) != 0
            cells[cell_id].add_merge_event(ref_dt, _absorbed_cells)
        # add predictions
        for forc_cen in feat.findall('./forecast/centroid_forecasts/centroid_forecast'):
            forc_dt = _dwd_tstr2dt(forc_cen.attrib['forecast_time'])
            cen_pos = {
                'lat' : float(forc_cen.find('./geodetic_coordinate/latitude').text),
                'lon' : float(forc_cen.find('./geodetic_coordinate/longitude').text),
            }
            uncertainty = dict(
                major=float(forc_cen.find('./uncertainty_ellipse/major_axis').text),
                minor=float(forc_cen.find('./uncertainty_ellipse/minor_axis').text),
                angle=float(forc_cen.find('./uncertainty_ellipse/angle').text),
            )
            cells[cell_id].add_forecasts(ref_dt, forc_dt, cen_pos, uncertainty)
        # done
    return cells
    
# Tracking
class Tracked_Cell(object):
    """optimized Tracking-Object based on KONRAD output"""
    def __init__(self, global_id):
        self.global_id = global_id
        self.local_ids_at_frame = dict()
        self.contours_at_frame = dict()
        self.splits_occurred = dict()
        self.absorbed_at_frame = dict()
        self.forecasts_at_frame = dict()
        pass
    def add_local_id(self, at_frame, local_id):
        self.local_ids_at_frame[at_frame] = local_id
    def add_contour(self, at_frame, lons, lats):
        self.contours_at_frame[at_frame] = {
            'lons' : lons,
            'lats' : lats,
        }
    def add_split_event(self, at_frame, from_cell_id):
        if at_frame not in self.splits_occurred:
            self.splits_occurred[at_frame] = from_cell_id
        else:
            return False
    def add_merge_event(self, at_frame, with_cells):
        if at_frame not in self.absorbed_at_frame:
            self.absorbed_at_frame[at_frame] = []
        self.absorbed_at_frame[at_frame] += with_cells
    def add_forecasts(self, at_frame, for_frame, centroid_pos, uncertainty):
        if at_frame not in self.forecasts_at_frame:
            self.forecasts_at_frame[at_frame] = dict()
        self.forecasts_at_frame[at_frame][for_frame] = {
            'centroid' : centroid_pos,
            'uncertainty' : uncertainty,
        }

def retrieve_tracked_cells(dt_start, dt_end, konrad_tarfp, cells=dict(), td_interval=datetime.timedelta(minutes=5), verbose=False):
    """Extracts KONRAD3D tracked cells between a datetime 'dt_start' and 'dt_end' from a tarfile 'konrad_tarfp'."""
    tree_maker = XMLtreeObtainer(TarAccessor(konrad_tarfp, _disable_file_checking=True))
    cur_dt = dt_start
    while cur_dt <= dt_end:
        if verbose: print ("\r",cur_dt,)
        desired_fpath_in_tar = "KONRAD3D/%s/KONRAD3D_%s.xml" % (cur_dt.strftime('%Y%m%d'), cur_dt.strftime('%Y%m%dT%H%M%S'))
        xml_tree = tree_maker.obtain_xml_tree(desired_fpath_in_tar)
        cells = obtain_tracking_infos(xml_tree, cells=cells)
        #
        cur_dt += td_interval
    else:
        if verbose: print ("\rdone!               ")
    return cells

# ---

# ---
# Regridding polar to cartesian
def calc_cart_coords(metadata, site_loc, proj=None, scan_name=None):
    """
    Calculates cartesian coordinates from metadata.
    For 'site_loc' instead of a lon/lat/height tuple, the radar-name can be given as str.
    If 'proj' is not given, the cartesian grid will be centered on the radar site location.
    """
    if scan_name is None:
        scan_name = radar_tools.get_lowest_el_scan(metadata)
    range_diff = np.diff(metadata[scan_name]['r'])
    assert (range_diff == range_diff[-1]).all() # ensure that every rbin has the same length
    rscale = range_diff[-1]
    nbins = len(metadata[scan_name]['r'])
    nrays = len(metadata[scan_name]['az'])
    nscans = 1 # for now...
    pol_coords = np.empty((nscans, nrays, nbins, 3)) #..., r,phi,theta
    for scan_num in range(nscans):
        pol_coords[scan_num, ...] = wradlib.georef.sweep_centroids(nrays, rscale, nbins, np.deg2rad(metadata[scan_name]['el']))
    #
    if type(site_loc) == str:
        full_site_coords = list(network_info.station_info[site_loc]['geo_pos'][::-1]) + [network_info.station_info[site_loc]['height']] # lon,lat,height
    else:
        full_site_coords = site_loc # lon,lat,height
    #
    if proj is None: # projection not given, so calculate the projection parameters
        cart_coords, proj = wradlib.georef.spherical_to_xyz(r=pol_coords[...,0],phi=np.degrees(pol_coords[...,1]),theta=np.degrees(pol_coords[...,2]), sitecoords=full_site_coords)
    else:
        if type(proj) is str:
            _proj = wradlib.georef.proj4_to_osr(proj)
        else:
            _proj = proj
        cart_coords = wradlib.georef.spherical_to_proj(r=pol_coords[...,0], phi=np.degrees(pol_coords[...,1]), theta=np.degrees(pol_coords[...,2]), sitecoords=full_site_coords, proj=_proj)
    #
    return cart_coords

def build_polar2cartesian_mapper(cart_coords):
    """builds a function to transform polar coordinates to cartesian coordinates based on given cartesian coordinates 'cart_coords' (from function 'calc_cart_coords')."""
    def polar2cart(coords):
        if len(coords.shape) > 1:
            return cart_coords[...,:2][coords.T.astype(int)]
        else:
            return cart_coords[...,:2][coords.astype(int)]
    return polar2cart
# ---

# ---


class ForecastSuite(object):
    """
    Combines all relevant information sources to enable detection and prediction of hail and it's size.
    
    To initialize grid-information and parameters for the membership-functions have to be provided.
    For each time step, new data can be put in using the .feed_in(...) function. Updated detection and prediction flags can be obtained by the .get_detection() and .get_prediction() functions.
    """
    def __init__(self, radar_name, grid_info_dict, membershipfun_params):
        self.radar_name = radar_name
        self.grid_info_dict = grid_info_dict
        self.membershipfun_params = membershipfun_params
        self.recent_detection_flags = None
        self.recent_prediction_flags = None
    def feed_in(self, data_dt, active_cells, pcp_data, alpha_data, vi_zc_data):
        """
        Update information by providing current data. Detection and prediction flags will be updated based on new information. Flags can be treived by .get_detection() and .get_prediction() functions.
        
        Arguments:
            data_dt:        datetime-object, date and time of the current time frame.
            active_cells:   list of TrackedCell-objects, contains the cells, which are currently present in the frame.
            pcp_data:       dict, contains radar variable data from precipitation scan (variable : array).
            alpha_data:     dict, volumetric attenuation coefficient data (elevation : "alpha" : array).
            vi_zc_data:     dict, vertical interpolated ZDR-column data (ZC_var_key : array).
        Returns:
            None
        """
        # data is already processed, so find the fitting cells for the data
        local_cell_dict = find_dynamicsdata4cell(active_cells, pcp_data, vi_zc_data, alpha_data, data_dt, self.radar_name, self.grid_info_dict['conv_pol2cart'])
        assign_data2cells(local_cell_dict, active_cells, pcp_data, vi_zc_data, alpha_data, data_dt, self.radar_name)
        # update cell dynamics and do hail detection and prediction
        update_cell_state(active_cells, data_dt)
        self.recent_detection_flags = predict_hail_timing_method(active_cells, data_dt)
        self.recent_prediction_flags = predict_hail_fuzzy_dynamics_method(active_cells, data_dt, build_fuzzy_logic_hs_aggr(self.membershipfun_params, Q=1))
    def get_detection(self):
        """Returns the most recent detection flags."""
        return self.recent_detection_flags
    def get_prediction(self):
        """Returns the most recent prediction flags."""
        return self.recent_prediction_flags
        
class TrackingDataProvider(object):
    """
    This class provides access to tracking information. Underlying is a KONRAD_Accessor, but can be replaced with any other tracking system.
    """
    def __init__(self, konrad_accessor, ):
        self.konrad_accessor = konrad_accessor
    def get_cells_at_dt(self, desired_dt):
        """Returns a list of TrackedCell-objects, which are currently, and actively tracked in the time step 'desired_dt'."""
        _cells = obtain_tracking_infos(self.konrad_accessor.obtain_xml_tree(desired_dt))
        return sorted(_cells.values(), key=lambda c: c.global_id)

class KONRAD_Accessor(object):
    """
    This class bridges offline KONRAD-data to live-update forecasting. The underlying file access needs to be replaced by a live stream.
    """
    def __init__(self, konrad_tarfp):
        self.tree_maker = XMLtreeObtainer(TarAccessor(konrad_tarfp, _disable_file_checking=False))
    def obtain_xml_tree(self, desired_dt):
        """Returns KONRAD data as a XML tree for date and time 'desired_dt'."""
        desired_fpath_in_tar = "KONRAD3D/%s/KONRAD3D_%s.xml" % (desired_dt.strftime('%Y%m%d'), desired_dt.strftime('%Y%m%dT%H%M00')) # seconds set to zero
        xml_tree = self.tree_maker.obtain_xml_tree(desired_fpath_in_tar)
        return xml_tree

# functions
def prepare_grid_info(metadata, radar_name):
    """
    This functions sets up all relevant grid informations (and a cartesian grid) by using metadata of a radar and it's 'radar_name'.
    """
    _dwd_proj = wradlib.georef.proj4_to_osr("+proj=stere +lat_0=90 +lat_ts=60 +lon_0=10 +a=6370040 +b=6370040 +no_defs +y_0=3608769.7242655735 +x_0=543337.16692185646")
    cart_coords = calc_cart_coords(metadata, radar_name, proj=_dwd_proj)
    wgs_coords = wradlib.georef.reproject(cart_coords, projection_source=_dwd_proj, projection_target=wradlib.georef.proj4_to_osr("+proj=longlat +datum=WGS84 +no_defs "))
    pol2cart = build_polar2cartesian_mapper(wgs_coords[0])
    grid_info_dict = {
        'cart_coords' : cart_coords,
        'wgs_coords' : wgs_coords,
        'conv_pol2cart' : pol2cart,
    }
    return grid_info_dict

def calc_ellipsoid_extent_in_wgs(centroid_lat, centroid_lon, major_axis, minor_axis):
    """
    Helper function for plotting. Calculates the ellipsoid extent from kilometers to degrees in WGS coords.
    
    Arguments:
        centroid_lat:  float, latitude in WGS coordinate system in deg N.
        centroid_lon:  float, longitude in WGS coordinate system in deg E.
        major_axis:    float, length of major axis in kilometer.
        minor_axis:    float, length of minor axis in kilometer.
    Returns:
        _lon_diff:     float, "distance" in longitude in degrees.
        _lat_diff:     float, "distance" in latitude in degrees.
    """
    _lon_diff = radar_tools.calc_terminal_point(centroid_lat, centroid_lon, 90., major_axis*1e3/2.)[1] - radar_tools.calc_terminal_point(centroid_lat, centroid_lon, 270., major_axis*1e3/2.)[1]
    _lat_diff = radar_tools.calc_terminal_point(centroid_lat, centroid_lon, 0., minor_axis*1e3/2.)[0] - radar_tools.calc_terminal_point(centroid_lat, centroid_lon, 180., minor_axis*1e3/2.)[0]
    return _lon_diff, _lat_diff

# ---

def plot_cells_and_forecasts(pcp_data, pcp_metadata, chosen_radar, cur_dt, active_cells, detection_flags, prediction_flags, ax=None, fig=None, _plot_cell_number=True, xlim=None, ylim=None, title=None):
    """
    Example function to plot radar data with KONRAD3D tracks and detected and predicted hail and it's size.
    
    Arguments:
        pcp_data:           dict, contains radar variable data from precipitation scan (variable : array).
        pcp_metadata:       dict, contains radar metadata from precipitation scan.
        chosen_radar:       str, name of the radar to be plotted.
        cur_dt:             datetime-object, date and time of the frame to plot (used to select the correct cell-data from 'active_cells').
        active_cells:       list of TrackedCell-objects, contains the cells, which are currently present in the frame.
        detection_flags:    dict, contains for each cell-id a dict of boolean-flags.
        prediction_flags:   dict, contains for each cell-id a dict of boolean-flags.
    Parameters:
        ax:                 matplotlib.axes-object, axis to plot in. Default: None
        fig:                matplotlib.Figure-object, figure to plot in. Default: None
        _plot_cell_number:  bool, show the cell-id in the plot. Default: True
        xlim:               tuple/float/None, defines the limit on the x-axis. Default: None
        ylim:               tuple/float/None, defines the limit on the y-axis. Default: None
        title:              str/bool/None, sets the title on top of the plot. If not a string but a positive boolean value is given, then the 'cur_dt' will be used as title. Default: None
    Returns:
        ax:                 matplotlib.axes-object, axis, which was used for plotting.
        pm:                 matplotlib.pcolormesh-object, the colored mesh, containing the contours of the plotted reflectivity factor.
    """
    lowest_scan_name = radar_tools.get_lowest_el_scan(pcp_metadata)
    _site_loc=network_info.station_info[chosen_radar]['geo_pos'][::-1]
    # Ricardo: needed because wradlib needs the height for plotting
    lis_site_loc = list(_site_loc)
    lis_site_loc.append(0)
    _site_loc_3D = tuple(lis_site_loc)
    print("Check on _site_loc = ", _site_loc_3D)
    wgs_proj = wradlib.georef.proj4_to_osr("+proj=longlat +datum=WGS84 +no_defs ")
    #
    color_cycle = [d['color'] for i,d in enumerate(plt.rcParams['axes.prop_cycle'])]
    del color_cycle[7] # gray is not helpful here
    hsc2color = [ # TODO: optimize for color-blindness!!!
        color_cycle[0], # no hail: blue
        color_cycle[2], # small hail: green
        color_cycle[1], # large hail: orange
        color_cycle[3], # giant hail: red
        color_cycle[8], # unknown: teal
    ]
    #
    if ax is None:
        ax = plt.gca()
    if fig is None:
        fig = plt.gcf()
    #if True: fig.set_size_inches(fig.get_figwidth()*1.75,fig.get_figheight()*1.75) # better viewable
    ax, pm = wradlib.vis.plot_ppi(
        pcp_data[u'pol_DBZH'],
        r=pcp_metadata[lowest_scan_name]['r'],
        az=pcp_metadata[lowest_scan_name]['az'],
        elev=pcp_metadata[lowest_scan_name]['el'],
        ax=ax,
        vmin=0,
        vmax=60,
        cmap=matplotlib.cm.get_cmap('gray',20),
        proj=wgs_proj,
        site=_site_loc_3D,
    )
    _xlim = ax.get_xlim() ; _ylim = ax.get_ylim() # could also be calculated
    cbar = fig.colorbar(pm, ax=ax, shrink=1.0, extend='both')
    cbar.set_label("hor. reflectivity factor (dBZ)")
    ax.set_xlabel("Longitude (deg E)");
    ax.set_ylabel("Latitude (deg N)");
    # draw crosshair
    ax_ch = wradlib.vis.plot_ppi_crosshair(_site_loc_3D, ranges=[50e3,100e3,150e3], ax=ax, proj=wgs_proj)
    # plot the cell contours
    for cell in active_cells:
        # check whether this cell is within vicinity of the radar
        #_contour_center = np.mean([cell.contours_at_frame[cur_dt]['lons'],cell.contours_at_frame[cur_dt]['lats']],axis=1)
        cc_lons = np.mean(list(cell.contours_at_frame[cur_dt]['lons']))
        cc_lats = np.mean(list(cell.contours_at_frame[cur_dt]['lats']))
        _contour_center = [cc_lons,cc_lats]
                           
        _dist2radar = radar_tools.haversine(_site_loc[0], _site_loc[1], _contour_center[0], _contour_center[1])
        #if _dist2radar*1e3 > max(pcp_metadata[lowest_scan_name]['r']): continue # skip this cell
        # determine detected hailsize (for color)
        detected_hsc = detection_flags[cell.global_id]['fallout_hailsize_category']
        growing_hsc = prediction_flags[cell.global_id]['growing_hsc']
        cell_color = hsc2color[detected_hsc]
        # plot cell contour
        ax.plot(list(cell.contours_at_frame[cur_dt]['lons']),list(cell.contours_at_frame[cur_dt]['lats']),color=cell_color)
        #ax.plot(_contour_center[0], _contour_center[1], marker='x', color=cell_color[cell.global_id])
        #if len(cell.local_ids_at_frame) >= _min_lifespan4track:
        #    pass # TODO: add past tracks
        if _plot_cell_number:
            ax.text(_contour_center[0], _contour_center[1], s=cell.global_id, color='black',fontsize='x-small',fontweight='bold', clip_on=True)
            ax.text(_contour_center[0], _contour_center[1], s=cell.global_id, color=cell_color,fontsize='x-small', clip_on=True)
        # show forecasted path
        forc_track = {'lats' : [], 'lons' : [],}
        for forc_dt in sorted(cell.forecasts_at_frame[cur_dt].keys()):
            forc_track['lats'].append(cell.forecasts_at_frame[cur_dt][forc_dt]['centroid']['lat'])
            forc_track['lons'].append(cell.forecasts_at_frame[cur_dt][forc_dt]['centroid']['lon'])
            # find color for predicted hsc
            pred_color = hsc2color[-1]
            pred_lw = 0.3
            if forc_dt in prediction_flags[cell.global_id]['predicted_hsc']:
                # we have a prediction for this forecasted timestep
                _pred_cat = prediction_flags[cell.global_id]['predicted_hsc'][forc_dt]
                pred_color = hsc2color[_pred_cat]
                if _pred_cat == 1:
                    pred_lw = 1
                if _pred_cat > 1:
                    pred_lw = 2
            # add estimated location ellipsoid
            ellips_width, ellips_height = calc_ellipsoid_extent_in_wgs(forc_track['lats'][-1], forc_track['lons'][-1], cell.forecasts_at_frame[cur_dt][forc_dt]['uncertainty']['major'], cell.forecasts_at_frame[cur_dt][forc_dt]['uncertainty']['minor'])
            forc_ellipsoid = matplotlib.patches.Ellipse((forc_track['lons'][-1],forc_track['lats'][-1]), ellips_width, ellips_height, cell.forecasts_at_frame[cur_dt][forc_dt]['uncertainty']['angle']-90., edgecolor=pred_color, linestyle='dotted', facecolor='none', lw=pred_lw)
            ax.add_patch(forc_ellipsoid)
        ax.plot(forc_track['lons'], forc_track['lats'], linestyle='dashed', color=hsc2color[growing_hsc])
        _d_lats = np.diff(forc_track['lats'])[-1]; _d_lons = np.diff(forc_track['lons'])[-1]
        if _d_lats == 0:
            _d_lats = 1e-3
        if _d_lons == 0:
            _d_lons = 1e-3
        ax.arrow(forc_track['lons'][-1], forc_track['lats'][-1], _d_lons, _d_lats, shape='full', lw=0, length_includes_head=True, head_width=.05, color=hsc2color[growing_hsc])
    # add title if desired
    if title is not None:
        if title is True:
            ax.set_title(cur_dt.strftime("%Y-%m-%d %H:%M:%S"))
        else:
            ax.set_title(title)
    ax.set_xlim(_xlim if xlim is None else xlim); ax.set_ylim(_ylim if ylim is None else ylim)
    # keep the aspect equal, so that radar isn't distorted
    ax.set_aspect(1 / ax.get_data_ratio()) # Source: https://stackoverflow.com/a/50061123
    return ax, pm

# ---

# data coords in cell contour
def check_coords_in_contour(contour, coords):
    """Returns an array of bools, each is True if the specific point of 'coords' is inside the 'contour'"""
    #print(contour)
    #print("Check on coords = ", coords, coords.shape, type(coords))
    Ac = []
    for ii in contour:
        alist = list(ii)
        Ac.append(alist)
    Ac = np.array(Ac).T
    #print(Ac, type(Ac), Ac.shape)
    bbPath = matplotlib.path.Path(Ac)
    #print("Check on bbPath = ", bbPath)
    return bbPath.contains_points(coords[0])
#     bbPath = matplotlib.path.Path(contour)
#     return bbPath.contains_points(coords)

def find_dynamicsdata4cell(active_cells, pcp_data, vi_param_data_zc, param_alpha_data, data_dt, chosen_radar, pol2cart, use_vol_alpha=True, _Z_threshold=40., _raise_on_missing_frames=False):
    """
    Retrieves data for cells.
    
    Arguments:
        active_cells:               list of TrackedCell-objects, contains the cells, which are currently present in the frame.
        pcp_data:                   dict, contains radar variable data from precipitation scan (variable : array).
        vi_param_data_zc:           dict, vertical interpolated ZDR-column data (ZC_var_key : array)
        param_alpha_data:           dict, attenuation coefficient data (elevation : 'alpha' : array) or ('alpha' : array) if 'use_vol_alpha' is False.
        data_dt:                    datetime-object, date and time of the current data.
        chosen_radar:               str, name of the selected radar.
        pol2cart:                   func, transforms polar coordinates to cartesian coordinates.
    Parameters:
        use_vol_alpha:              bool, determines whether the alpha data put in is volumetric or not. Default: False
        _Z_threshold:               float, threshold in dBZ, which has to be exceeded to assume hail being present. Default: 40.
        _raise_on_missing_frames:   bool, raise exception, when frames are missing. Default: False
    Returns:
        local_cell_dict:            dict, (t : local_cell_id : param_key : coords) intended to hand over in function 'assign_data2cells'.
    """
    # modified to suite realtime-update purposes
    local_cell_dict = {} # local_cell_id : param_key : coords # coords can differ for ZC params and alpha param
    # determine shape for alpha_bool_array
    if use_vol_alpha:
        _alpha_arr_shape = vi_param_data_zc['ZC_height'].shape # is also from VOL data and already has the required shape determined
    else:
        _alpha_arr_shape = param_alpha_data['alpha'].shape
    alpha_bool_map = np.zeros(_alpha_arr_shape, dtype=bool) # need a dummy, then iterate over all elevations
    tse_flag_coords = ([],[],)
    # need to distinguish between vol and pcp
    if not use_vol_alpha: # wrap data somehow to use 'el'-iteration
        _el_data_dict = {'pcp' : param_alpha_data}
    else:
        _el_data_dict = param_alpha_data
        # need to check if we actually have elevations in here
        assert [int(k) for k in _el_data_dict] # TODO: this is not a valid assertion! It will raise an integer-error on fail...
    for el in _el_data_dict:
        if 'alpha' not in _el_data_dict[el]: continue
        _cur_alpha_mask = ~np.isnan(_el_data_dict[el]['alpha'])
        alpha_bool_map[:,:_cur_alpha_mask.shape[1]] = np.logical_or(alpha_bool_map[:,:_cur_alpha_mask.shape[1]], _cur_alpha_mask[:360,:])
        if 'tse_flags' not in _el_data_dict[el]: continue
        for az_ind in _el_data_dict[el]['tse_flags']:
            for r_ind in _el_data_dict[el]['tse_flags'][az_ind]:
                tse_flag_coords[0].append(az_ind)
                tse_flag_coords[1].append(r_ind)
    #TODO: find a better way to use inds instead of multiplying by 4
    zch_coords = np.asarray(np.where(~np.isnan(vi_param_data_zc['ZC_height'])))
    zch_coords[1] *= 4 #VOL range inds -> PCP range inds
    alpha_coords = np.asarray(np.where(alpha_bool_map))
    if use_vol_alpha: alpha_coords[1] *= 4 #VOL range inds -> PCP range inds
    tse_flag_coords = np.asarray(tse_flag_coords)
    if use_vol_alpha: tse_flag_coords[1] *= 4
    # identify propable hail bearing areas
    Z = pcp_data['pol_DBZH']
    valid_Z_coords = np.asarray(np.where(Z>=_Z_threshold))
    # coordinates are checked in polar to avoid regridding of alpha and zch data
    zch_valid_coords = np.asarray([p for p in zip(*zch_coords) if p in zip(*valid_Z_coords)])
    alpha_valid_coords = np.asarray([p for p in zip(*alpha_coords) if p in zip(*valid_Z_coords)])
    tse_flag_valid_coords = np.asarray([p for p in zip(*tse_flag_coords) if p in zip(*valid_Z_coords)])
    valid_refl_coords = np.asarray([p for p in zip(*valid_Z_coords)])
    # require cartesian coords to compare with contours (contours are on cartesian grid)
#     print("zch_valid_coords) =", zch_valid_coords)
    zch_valid_cart_coords = pol2cart(zch_valid_coords)
    alpha_valid_cart_coords = pol2cart(alpha_valid_coords)
    tse_flag_valid_cart_coords = pol2cart(tse_flag_valid_coords)
    valid_refl_cart_coords = pol2cart(valid_refl_coords)
    # gather coordinates for ZDR-columns
    for cell in active_cells:
        cell_id = cell.global_id
        cart_contour = np.asarray(
            (cell.contours_at_frame[data_dt]['lons'], cell.contours_at_frame[data_dt]['lats'])
        ).T #done: need to transform the contour to fit the expected order
        # keep the coordinate polar, since they are needed to extract from archive data without doing regridding first
        zch_coords = zch_valid_coords[check_coords_in_contour(cart_contour, zch_valid_cart_coords)] if len(zch_valid_coords) > 0 else []
        alpha_coords = alpha_valid_coords[check_coords_in_contour(cart_contour, alpha_valid_cart_coords)] if len(alpha_valid_coords) > 0 else []
        tse_flag_coords = tse_flag_valid_coords[check_coords_in_contour(cart_contour, tse_flag_valid_cart_coords)] if len(tse_flag_valid_coords) > 0 else []
        refl_coords = valid_refl_coords[check_coords_in_contour(cart_contour, valid_refl_cart_coords)]
        if len(refl_coords) == 0 and len(zch_coords) == 0 and len(alpha_coords) == 0: continue
        local_cell_dict[cell_id] = {}
        if zch_coords is not []: local_cell_dict[cell_id]['ZC_height'] = zch_coords # polar indices
        if alpha_coords is not []: local_cell_dict[cell_id]['alpha'] = alpha_coords # polar indices
        if tse_flag_coords is not []: local_cell_dict[cell_id]['tse_flag'] = tse_flag_coords # polar indices
        if refl_coords is not []: local_cell_dict[cell_id]['reflectivity'] = refl_coords # polar indices
    #
    return local_cell_dict

def assign_data2cells(local_cell_dict, active_cells, pcp_data, vi_param_data_zc, param_alpha_data, data_dt, chosen_radar, use_vol_alpha=True, _raise_on_missing_frames=False):
    """
    Adds processed radar-data to tracked cells.
    
    Arguments:
        local_cell_dict:            dict, taken from function 'find_dynamicsdata4cell'.
        active_cells:               list of TrackedCell-objects, which are present in the current frame.
        pcp_data:                   dict, contains radar variable data from precipitation scan (variable : array).
        vi_param_data_zc:           dict, vertical interpolated ZDR-column data (ZC_var_key : array)
        param_alpha_data:           dict, attenuation coefficient data (elevation : "alpha" : array) or ('alpha' : array) if 'use_vol_alpha' is False.
        data_dt:                    datetime-object, date and time of the current data.
        chosen_radar:               str, name of the selected radar.
    Parameters:
        use_vol_alpha:              bool, determines whether the alpha data put in is volumetric or not. Default: False
        _raise_on_missing_frames:   bool, raise exception, when frames are missing. Default: False
    Returns:
        None
    """
    # now iterate over the global_cells and use the local_id to obtain coords
    # use the coords to save directly fitting values
    # prepare some helper functions
    if use_vol_alpha:
        # we need a function to gather for all elevations the alpha data
        _hf_alpha_selector = lambda _el_data_dict, coord: np.nanmax([_el_data_dict[el]['alpha'][coord[0], coord[1]/4] for el in _el_data_dict if 'alpha' in _el_data_dict[el].keys() and _el_data_dict[el]['alpha'].shape[1] > coord[1]/4])
    else:
        _hf_alpha_selector = lambda _el_data_dict, coord: _el_data_dict['alpha'][coord[0], coord[1]]
    #
    for cell in active_cells:
        cid = cell.global_id
        if not hasattr(cell, 'local_data_at_frame'):
            cell.local_data_at_frame = dict()
        cell.local_data_at_frame[data_dt] = {'ZC_height' : [], 'ZC_maxval' : [], 'alpha' : [], 'reflectivity' : [], 'tse_flag' : [],}
        if cid not in local_cell_dict.keys(): continue
        #
        for coord in local_cell_dict[cid]['ZC_height']:
            _zch_vals = vi_param_data_zc['ZC_height'][coord[0],coord[1]/4]
            cell.local_data_at_frame[data_dt]['ZC_height'].append(_zch_vals)
            _zcm_vals = vi_param_data_zc['ZC_maxval'][coord[0],coord[1]/4]
            cell.local_data_at_frame[data_dt]['ZC_maxval'].append(_zcm_vals)
        for coord in local_cell_dict[cid]['alpha']:
            _alpha_vals = _hf_alpha_selector(param_alpha_data, coord)
            cell.local_data_at_frame[data_dt]['alpha'].append(_alpha_vals)
        for coord in local_cell_dict[cid]['tse_flag']:
            cell.local_data_at_frame[data_dt]['tse_flag'].append(coord)
        for coord in local_cell_dict[cid]['reflectivity']:
            _refl_val = pcp_data['pol_DBZH'][coord[0],coord[1]]
            cell.local_data_at_frame[data_dt]['reflectivity'].append(_refl_val)
    #

# Prediction Tools
class CellState(object):
    """
    This object stores past and present information about the temporal development about it's ZDR-columns and attenuation.
    """
    def __init__(self):
        self.state_valid_for_dt = datetime.datetime(1970,1,1,)
        #self.hist_had_ZC_before = False # interesting, but apparently not required
        self.hist_number_of_ZC_spikes = 0
        self.hist_number_of_alpha_spikes = 0
        self.hist_highest_possible_hail_cat = 0
        self._last_frame_had_zch_spike = False
        self._last_frame_had_alpha_spike = False
        #self.hist_
        pass
def _pad_sub_vec(sub_vec, pad_width=2):
    # pad array in front, e.g. so that range(1,3) becomes [0,0,1,2,3]
    return np.pad(sub_vec,(pad_width,0),mode='constant',constant_values=0) # adds X zeros to the beginning
def _is_latest_peak(sub_vec, _hard_threshold=500.0, _significance_fac=1.5):
    # test whether the ZCH is above a threshold
    if not sub_vec[-1] >= _hard_threshold: return False
    # test whether last increase and variance are in sum higher than the standard deviation (times X)
    if not np.diff(sub_vec)[-1] + (sub_vec - np.nanmean(sub_vec))[-1] >= np.std(sub_vec)*_significance_fac: return False
    # more significant, so return True
    return True
def _check_existance(val_vec):
    if not (val_vec[-1] > 0): return False # ZC (or other) not present in latest frame
    _existing_since = -1
    for i in range(1, len(val_vec)+1):
        if val_vec[-i] > 0:
            _existing_since += 1
        else:
            break
    return _existing_since

def update_cell_state(active_cells, cur_dt, _win_len_zch=5, _win_len_alpha=4):
    """
    This updates for each currently active cell their information on the temporal development of ZDR-columns and attenuation within the cells.
    
    Arguments:
        active_cells:       list of TrackedCell-objects, which are present in the current frame.
        cur_dt:             datetime-object, date and time of the frame to plot (used to select the correct cell-data from 'active_cells').
    Parameters:
        _win_len_zch:       int, number of past time steps used, in which spikes of ZDR-columns will be searched for. Default: 5
        _win_len_alpha:     int, number of past time steps used, in which spikes of attenuation coefficient will be searched for. Default: 4
    Returns:
        None
    """
    for cell in active_cells:
        if cur_dt not in cell.local_data_at_frame: continue # well, then this cell isn't active...
        if not hasattr(cell, 'state') or cell.state is None:
            cell.state = CellState()
            cell.state.hist_zch_vec_nn_med = []
            cell.state.hist_alpha_vec_nn_med = []
            cell.state.hist_zczdr_vec_nn_med = []
            cell.state.hist_last20min_spiking_zch = []
        # check whether each frame/timestep is 5min in length
        if len(cell.local_data_at_frame) > 1:
            ## need to filter for impossible timesteps (e.g. future time steps, occurs only at offline test)
            #for t in cell.local_data_at_frame.keys():
            #    if t > cur_dt:
            #        del cell.local_data_at_frame[t]
            #        del cell.local_ids_at_frame[t]
            #
            #assert (np.diff(sorted(cell.local_data_at_frame.keys())) == datetime.timedelta(minutes=5)).all()
            if not (np.diff(sorted(cell.local_data_at_frame.keys())) == datetime.timedelta(minutes=5)).all():
                print ("WARNING: frames of cell %s are not consecutively in 5min order! (cur_dt=%s, all_dts=(%s))" % (cell.global_id, cur_dt, ','.join(map(repr, sorted(cell.local_data_at_frame.keys())))) )
        #
        nn_fu = lambda x: 0 if np.isnan(x) else x
        nn_arr = lambda arr: np.where(np.isnan(arr), 0, arr)
        cell.state.hist_zch_vec_nn_med.append(nn_fu(np.nanmedian(cell.local_data_at_frame[cur_dt]['ZC_height'])))
        cell.state.hist_alpha_vec_nn_med.append(nn_fu(np.nanmedian(cell.local_data_at_frame[cur_dt]['alpha'])))
        cell.state.hist_zczdr_vec_nn_med.append(nn_fu(np.nanmedian(cell.local_data_at_frame[cur_dt]['ZC_maxval'])))
        cell.state.state_valid_for_dt = cur_dt # TODO: might consider moving this to the end
        # check if new ZCH spike occured
        _sub_vec = _pad_sub_vec(cell.state.hist_zch_vec_nn_med[-_win_len_zch-1:], pad_width=_win_len_zch-1) # indexing here is different (than in the explicit indexing), therefore _win_len needs the -1 here.
        cell.state._cur_frame_zch_spiking = _is_latest_peak(_sub_vec, _hard_threshold=500.0)
        if cell.state._cur_frame_zch_spiking:
            if not cell.state._last_frame_had_zch_spike: # do not increase spiking count, when last frame was spiking, too (one big spike).
                cell.state.hist_number_of_ZC_spikes += 1
                cell.state._last_frame_had_zch_spike = True
        else:
            cell.state._last_frame_had_zch_spike = False
        # store spiking info for 20min
        cell.state.hist_last20min_spiking_zch.append(cell.state._cur_frame_zch_spiking)
        if len(cell.state.hist_last20min_spiking_zch) > 4:
            cell.state.hist_last20min_spiking_zch = cell.state.hist_last20min_spiking_zch[-4:]
        # check if new alpha spike occured
        _sub_vec = _pad_sub_vec(cell.state.hist_alpha_vec_nn_med[-_win_len_alpha-1:], pad_width=_win_len_alpha-1) # indexing here is different (than in the explicit indexing), therefore _win_len needs the -1 here.
        cell.state._cur_frame_alpha_spiking = _is_latest_peak(_sub_vec, _hard_threshold=0.15)
        if cell.state._cur_frame_alpha_spiking:
            if not cell.state._last_frame_had_alpha_spike: # do not increase spiking count, when last frame was spiking, too (one big spike).
                cell.state.hist_number_of_alpha_spikes += 1
                cell.state._last_frame_had_alpha_spike = True
        else:
            cell.state._last_frame_had_alpha_spike = False
        # check for duration and existance of ZC
        cell.state._ZC_existing_since = _check_existance(cell.state.hist_zch_vec_nn_med)
        # calculate the median ZCH of current consistently existing ZC
        if cell.state._ZC_existing_since > 0:
            cell.state._med_cur_ZCH = np.median(cell.state.hist_zch_vec_nn_med[-(cell.state._ZC_existing_since+1):])
        else:
            cell.state._med_cur_ZCH = np.NaN
        # done (for this timestep)

def predict_hail_timing_method(active_cells, cur_dt):
    """
    Detect hail and size-category for each cell using a qualitative timing-based method.
    
    Arguments:
        active_cells:       list of TrackedCell-objects, which are present in the current frame.
        cur_dt:             datetime-object, date and time of the frame to plot (used to select the correct cell-data from 'active_cells').
    Returns:
        prediction_flags:   dict, contains for each cell-id a dict of boolean-flags.
    """
    prediction_flags = dict()
    for cell in active_cells:
        if cur_dt not in cell.local_data_at_frame: continue #assert cur_dt in cell.local_data_at_frame # well, if not then this cell isn't active...
        if cell.state.state_valid_for_dt != cur_dt:
            raise ValueError("ERROR: cell's state isn't updated for desired timestamp (%s, cell-state-timestamp: %s)!" % (cur_dt, cell.state.state_valid_for_dt))
        cell_id = cell.global_id
        prediction_flags[cell_id] = dict()
        prediction_flags[cell_id]['possible_hailsize_category'] = 0 # 0: no hail, 1: small, 2: large, 3: huge
        prediction_flags[cell_id]['fallout_hailsize_category'] = 0 # 0: no hail, 1: small, 2: large, 3: huge
        prediction_flags[cell_id]['ZCH_spiking'] = 0 # 0: no spike, 1: spike, 2: spike above 2500m, -1: collapse (300m below median)
        prediction_flags[cell_id]['alpha_spiking'] = False # False: no spike, True: spike
        #prediction_flags[cell_id]['total_signal_extinction'] = False # False: no, True: total signal extinction (tse) detected
        #
        frame_ind = sorted(cell.local_data_at_frame.keys()).index(cur_dt) # this should be equal to -1 if everything runs smooth, if not other state variables are wrong!
        #
        #
        #for i,(t,lid) in enumerate(sorted(tracked_cells[cell_id].local_ids_at_frame.items())):
        if cell.state._cur_frame_zch_spiking:
            prediction_flags[cell_id]['ZCH_spiking'] += 1 # spike exists
            if cell.state.hist_zch_vec_nn_med[frame_ind] > 2500.:
                prediction_flags[cell_id]['ZCH_spiking'] += 1 # huge spike
        elif cell.state.hist_zch_vec_nn_med[frame_ind] - cell.state._med_cur_ZCH <= -300.:
            # ZCH is below current median -> ZC collapse (not vanishing, but collapsing)
            prediction_flags[cell_id]['ZCH_spiking'] = -1
        #if len(_dbg_cell_dyn_hist[cell_id]['tse_flag_vec'][i]) > 0:
        #    prediction_flags[cell_id]['total_signal_extinction'][-1] = True
        # alpha analysis and hailsize possibility
        if not (cell.state.hist_alpha_vec_nn_med[frame_ind] > 0):
            continue
        else:
            # identify the possible hailsize category
            poss_hail_cat = 1 # small hail
            if cell.state._ZC_existing_since >= 3:
                # cell has ZC for at least 15min # or median ZCH of more than 2500m ?
                poss_hail_cat += 1 # small -> large
                if cell.state._ZC_existing_since >= 6 and cell.state._med_cur_ZCH >= 1200:
                    # cell has ZC for at least 30min and a median ZCH of 1200m or more
                    poss_hail_cat += 1 # large -> huge
            if poss_hail_cat < 3 and cell.state.hist_highest_possible_hail_cat > 2: # huge hail was possible before
                # check if median ZCH is still over threshold but ZC collapsed
                #_nn_last_30min_zch = np.where(np.isnan(_dbg_cell_dyn_hist[cell_id]['zch_vec'][max(0,i+1-6):i+1]), 0, _dbg_cell_dyn_hist[cell_id]['zch_vec'][max(0,i+1-6):i+1]) # TODO: this has to be moved to the update function
                # TODO: find index for last 30min
                _nn_last_30min_zch = np.where(np.isnan(cell.state.hist_zch_vec_nn_med[-6:]), 0, cell.state.hist_zch_vec_nn_med[-6:])
                if np.median(_nn_last_30min_zch) >= 1000:
                    poss_hail_cat = 3 # huge
                elif (np.asarray(cell.state.hist_last20min_spiking_zch) != 0).any(): # if ZCH was spiking in the last 20minutes, huge is still possible
                    poss_hail_cat = 3 # huge
            cell.state.hist_highest_possible_hail_cat = max(cell.state.hist_highest_possible_hail_cat, poss_hail_cat)
            # identify if we have detected hail falling out
            if cell.state._cur_frame_alpha_spiking:# or prediction_flags[cell_id]['total_signal_extinction'][i]:
                fallout_hail_cat = poss_hail_cat
                prediction_flags[cell_id]['alpha_spiking'] = True
            elif cell.state.hist_alpha_vec_nn_med[-1] >= 0.25 and poss_hail_cat > 1:
                # alpha is not spiking but high enough for medium or large hail
                fallout_hail_cat = 2#poss_hail_cat - 1
            elif cell.state.hist_alpha_vec_nn_med[-1] >= 0.1:
                fallout_hail_cat = 1
            else:
                fallout_hail_cat = 0
                # also use ZCH collapse to identify a possible fall-out
                if prediction_flags[cell_id]['ZCH_spiking'] == -1: # and alpha is present
                    fallout_hail_cat = max(1, poss_hail_cat - 1)
            prediction_flags[cell_id]['possible_hailsize_category'] = poss_hail_cat
            prediction_flags[cell_id]['fallout_hailsize_category'] = fallout_hail_cat
    return prediction_flags

def predict_hail_fuzzy_dynamics_method(active_cells, cur_dt, fl_aggregator):
    """
    Predict hail and size-category for each cell using a fuzzy-logic based method.
    
    Arguments:
        active_cells:           list of TrackedCell-objects, which are present in the current frame.
        cur_dt:                 datetime-object, date and time of the frame to plot (used to select the correct cell-data from 'active_cells').
        fl_aggregator:          func, takes a dict with ZDR-column parameters (height from melting layer to top of ZDR-column height and max. ZDR inside column) and returns for each hailsize-category an aggregated value (obtain this function from build_fuzzy_logic_hs_aggr(...) function).
    Returns:
        prediction_flags:       dict, contains for each cell-id a dict of boolean-flags.
    """
    prediction_flags = dict()
    for cell in active_cells:
        if cur_dt not in cell.local_data_at_frame: continue #assert cur_dt in cell.local_data_at_frame # well, if not then this cell isn't active...
        if cell.state.state_valid_for_dt != cur_dt:
            raise ValueError("ERROR: cell's state isn't updated for desired timestamp (%s, cell-state-timestamp: %s)!" % (cur_dt, cell.state.state_valid_for_dt))
        cell_id = cell.global_id
        prediction_flags[cell_id] = dict()
        #prediction_flags[cell_id]['ZCH_spiking'] = 0 # 0: no spike, 1: spike, 2: spike above 2500m, -1: collapse (300m below median)
        prediction_flags[cell_id]['growing_hsc'] = 0 # 0: none, 1: small, 2: large, 3: giant
        prediction_flags[cell_id]['predicted_hsc'] = dict() # datetime of prediction frames,  0: none, 1: small, 2: large, 3: giant
        #prediction_flags[cell_id]['alpha_spiking'] = False # False: no spike, True: spike
        #
        frame_ind = sorted(cell.local_data_at_frame.keys()).index(cur_dt) # this should be equal to -1 if everything runs smooth, if not other state variables are wrong!
        # identify the growing hailsize category
        grow_hail_cat = 0
        if True:#if prediction_flags[cell_id]['ZCH_spiking'] > 0 or not predict_only_at_spike:
            grow_hail_cat = 1 + discriminate_category(fl_aggregator(mask_val_bounds(dict(ZCH=cell.state.hist_zch_vec_nn_med[frame_ind]/1000., ZCmZDR=cell.state.hist_zczdr_vec_nn_med[frame_ind]))))
            prediction_flags[cell_id]['growing_hsc'] = grow_hail_cat
            # prediction is 10 or 15min for small hail, 15 or 10min for large hail and 20 or 15min for giant hail
            for i in [(2,3),(2,3),(3,4)][grow_hail_cat-1]:
                for_frame = cur_dt + datetime.timedelta(minutes=5)*i
                prediction_flags[cell_id]['predicted_hsc'][for_frame] = grow_hail_cat
    return prediction_flags
    
# Hailsize Discriminator Functions
def trapezoidal_membership_func(x,a,b,c,d):
    """Uses row-wise min/max to determine value within trapezodial membership function."""
    # Based on: https://www.mathworks.com/help/fuzzy/trapmf.html
    return np.stack([np.stack([(x-a)/(b-a),np.ones(np.asarray(x).shape),(d-x)/(d-c)]).min(axis=0),np.zeros(np.asarray(x).shape)]).max(axis=0)
def discriminate_category(aggr_arr):
    """
    Returns the row with the highest aggregated value. This is used to idenfity the hailsize-category from an aggregated, fuzzified input.
    """
    #return np.argmax(aggr_arr,axis=0)
    return np.where(np.max(np.isnan(aggr_arr),axis=0),-1,np.argmax(aggr_arr,axis=0)) # return -1, when NaN occured in any of the aggregated values; otherwise the class if highest aggregation number
def mask_val_bounds(val_dict, bound_dict=dict(ZCH=(0.4,15), ZCmZDR=(2.0,10))):
    """
    Takes a dict of fuzzified input and adjusts the fuzzy values to boundaries (values outside of bounds will be set to np.NaN).
    """
    masked_val_dict = dict()
    for k,v in val_dict.items():
        if k not in bound_dict:
            raise KeyError("ERROR: key '%s' in value-dict, but not in bound-dict!" % (k))
        #assert (bound_dict[k][0] <= np.asarray(v) < bound_dict[k][1]).all()
        masked_val_dict[k] = np.where(np.logical_and(bound_dict[k][0] <= np.asarray(v), np.asarray(v) < bound_dict[k][1]), np.asarray(v), np.NaN)
    return masked_val_dict
def build_fuzzy_logic_hs_aggr(_mf_param, Q=1.0):
    """
    Constructs a fuzzy-logic function for usage of predict_hail_fuzzy_dynamics_method(...) function using membership function parameters '_mf_param'.
    """
    assert np.sum(Q) == 1.0 # the weighting vector must be normed to 1
    def calc_fl_aggr_hsc(val_dict):
        # check if weighting vector has same dimension
        assert np.rank(Q) == 0 or np.rank(Q) == np.rank(val_dict[val_dict.keys()[0]]) +1
        aggr_arr = []
        for i, s_cat in enumerate(['small', 'large', 'giant']):
            # aggregate propabilities values across variables
            aggr_arr.append(np.mean(
                np.stack([
                    # fuzzificate values of input variables
                    trapezoidal_membership_func(val,*_mf_param[v_cat][s_cat]) for v_cat, val in val_dict.items()
                ]) * Q, axis=0, # weight them and then sum
            ))
        return np.asarray(aggr_arr)
    return calc_fl_aggr_hsc
# ---

class RawRadarDataFileHost(object):
    """
    This object gathers info about offline radar data.
    """
    data_base_folder='./data/'
    def __init__(self, data_base_folder=None, _dbg_dt_lim=None):
        if data_base_folder is not None:
            self.data_base_folder = data_base_folder
        self._dbg_dt_lim = _dbg_dt_lim
        self._file_info_dict = self.obtain_file_infos_raw_data()
    
    def obtain_file_infos_raw_data(self):
        """
        Gathers info about present offline files and stores this info internal. Shouldn't be necessary to execute this function on your own.
        """
        # find all possible valid files
        #file_infos = []
        file_info_dict = dict()
        for dirpath, dirnames, filenames in os.walk(self.data_base_folder):
            if not dirpath.split('/')[-1].startswith('20'): # TODO: improve the detection of the relevant folders. This isn't a good solution.
                continue
            for filename in filenames:
                if not filename.endswith('.tar.gz'):
                    continue
                fn_split = filename.replace('.tar.gz', '').split('_')
                ff_dict = {
                    'stat_num' : fn_split[0],
                    'stat_letter' : fn_split[1], # TODO: consider validating this
                    'scan_type' : fn_split[2].split('-')[0],
                    'timestamp' : fn_split[3], # no reason to put into dt-obj first
                }
                dt_obj = datetime.datetime.strptime(ff_dict['timestamp'], '%Y%m%d%H%M%S')
                # debug: skip files which are not in desired time-range
                if self._dbg_dt_lim is not None:
                    if not (self._dbg_dt_lim[0] <= dt_obj <= self._dbg_dt_lim[1]):
                        #print "DEBUG: skipping", dt_obj, ff_dict['stat_letter'], ff_dict['scan_type']
                        continue
                    else:
                        #print "DEBUG: adding", dt_obj, ff_dict['stat_letter'], ff_dict['scan_type']
                        pass
                # ---
                event_dt = dt_obj.date()
                if event_dt not in file_info_dict: file_info_dict[event_dt] = dict()
                if ff_dict['stat_letter'] not in file_info_dict[event_dt]: file_info_dict[event_dt][ff_dict['stat_letter']] = dict()
                if ff_dict['scan_type'] not in file_info_dict[event_dt][ff_dict['stat_letter']]: file_info_dict[event_dt][ff_dict['stat_letter']][ff_dict['scan_type']] = []
                file_info_dict[event_dt][ff_dict['stat_letter']][ff_dict['scan_type']].append((dirpath, ff_dict))
        return file_info_dict

    def gen_radar_raw_data_filepath(self, event_dt, chosen_radar, scan_type='pcp'):
        """
        Provides paths to raw radar data files as a iterative generator.
        """
        sub_folder_form = '%Y%m%d'
        filename_form = '%(stat_num)s_%(stat_letter)s_%(scan_type)s-op-sidpol-01_%(timestamp)s.tar.gz'
        #filename_form = '%(timestamp)s_%(stat_letter)s.tar.gz'
        for _, ff_dict in self._file_info_dict[event_dt.date()][chosen_radar][scan_type]:
            full_filepath = os.path.join(self.data_base_folder, event_dt.strftime(sub_folder_form), filename_form % ff_dict)
            #print "DEBUG: commencing loading with ffpath", full_filepath
            yield full_filepath

class RawRadarDataProvider(object):
    """
    This object bridges offline file-info to constant data stream.
    """
    def __init__(self, _dbg_dt_lim=None):
        self.filehost = RawRadarDataFileHost(_dbg_dt_lim=_dbg_dt_lim)
    def gen_raw_data_dicts(self, event_dt, chosen_radar, scan_type='pcp'):
        """
        Opens offline data for 'event_dt', 'chosen_radar' and 'scan_type' and iterates their processed radar data content as a generator.
        """
        assert scan_type == 'pcp' or scan_type == 'vol'
        if scan_type == 'pcp':
            for pcp_fp in self.filehost.gen_radar_raw_data_filepath(event_dt, chosen_radar, scan_type):
                for data, metadata in gen_multi_scan_data(pcp_fp, highest_scan_num=1, verbose=True, _rewrite_lines=False):
                    yield data, metadata
        elif scan_type == 'vol':
            for vol_fp in self.filehost.gen_radar_raw_data_filepath(event_dt, chosen_radar, scan_type):
                for data, metadata in gen_multi_scan_data(vol_fp, highest_scan_num=10, verbose=True, _rewrite_lines=False):
                    yield data, metadata
        else:
            raise KeyError("ERROR: unknown scan_type '%s'!" % (scan_type))

class RadarDataProvider(RawRadarDataProvider):
    """
    This object turns offline radar data into a stream of radar data for testing the online hail forecast.
    """
    # done: raw radar data needs to come from a stream
    # TODO: processed data (alpha, ZCH, ...) needs to be processed from received raw radar data
    #_default_scan_name = u'DATASET1'
    def __init__(self, **kwargs):
        super(RadarDataProvider, self).__init__(**kwargs)
        self.cur_pcp_data = None
        self.cur_metadata = None
        self.cur_zc_data = None
        self.cur_attcor_data = None
        if self.filehost._dbg_dt_lim is not None:
            print ("DEBUG: file time limit/range has been set:", self.filehost._dbg_dt_lim )
        # additional, debugging infos
        self._dt_after_rawdata_read = None
    def gen_data_updates(self, event_dt, radar_name):
        """
        iterate over data and yield current time stemp.
        current pcp, alpha and zch data can be retrieved via the .get_XXX_data() functions.
        """
        pcp_data_gen = self.gen_raw_data_dicts(event_dt, radar_name, scan_type='pcp')
        vol_data_gen = self.gen_raw_data_dicts(event_dt, radar_name, scan_type='vol')
        #
        att_cor = AttCorrHelper(_disable_warnings=True)
        fhh = FreezingHeightHelper()
        #
        self.cur_pcp_data = dict()
        self.cur_metadata = dict()
        self.cur_zc_data = dict()
        self.cur_attcor_data = dict()
        # start iterating over time and update processed data
        for pcp_tuple, vol_tuple in zip(pcp_data_gen, vol_data_gen): # itertools.izip(pcp_data_gen, vol_data_gen):
            # XXX_tuple -> (data, metadata)
            _pcp_lscan = radar_tools.get_lowest_el_scan(pcp_tuple[1])
            _pcp_dt = radar_tools.meta2dt_obj(pcp_tuple[1], _pcp_lscan)
            _vol_dt = radar_tools.meta2dt_obj(vol_tuple[1], radar_tools.get_lowest_el_scan(vol_tuple[1]))
            assert abs(_pcp_dt - _vol_dt) < datetime.timedelta(minutes=5)
            self._dt_after_rawdata_read = datetime.datetime.utcnow()
            # update processed data
            self.cur_pcp_data[u'pol_DBZH'] = pcp_tuple[0][_pcp_lscan][u'DBZH']['data'] # update pcp data
            self.cur_metadata['pcp'] = pcp_tuple[1] # update metadata, pcp scan
            self.cur_metadata['vol'] = vol_tuple[1] # update metadata, vol scan
            self.cur_attcor_data['vol'] = att_cor.make_att_corr_all_scans(*vol_tuple) # update alpha & att-cor data
            self.cur_attcor_data['pcp'] = att_cor.make_att_corr_single_scan(*pcp_tuple, scan_name=_pcp_lscan)
            self.cur_zc_data = calc_vi_zch_data(*vol_tuple, freezing_height=fhh.get_closest_melting4dt(_vol_dt, radar_name)[0.0], radar_name=radar_name) # update vi_zc data
            # yield the current timestamp
            yield _pcp_dt, _vol_dt
        #
    def get_pcp_data(self,):
        self._check_data_gen_active()
        return self.cur_pcp_data
    def get_meta_data(self,):
        self._check_data_gen_active()
        return self.cur_metadata
    def get_vi_zc_data(self,):
        self._check_data_gen_active()
        return self.cur_zc_data
    def get_att_cor_data(self,):
        self._check_data_gen_active()
        return self.cur_attcor_data
    def _check_data_gen_active(self):
        if self.cur_pcp_data is None:
            raise ValueError("ERROR: data updates have not been initialized, yet. Therefore, no data can be obtained! Iterate over .gen_data_updates() and retrieve data in each loop-step!")
        else:
            return True

class AttCorrHelper(object):
    """
    Prepares attenuation correction objects and configuration, and provides easy to use functions to correct attenuation.
    """
    def __init__(self, _raise_ZPHI_exceptions=False, _attempted_method_on_signal_extinction=None, _verbose=False, _phidp_entangle_ws=0, _entangle_Ah_half_ws=2, _disable_warnings=False):
        if _disable_warnings: # this might need to go into the thread process, not the parent
            import warnings
            warnings.filterwarnings("ignore")
        self.cur_phi_sys = 0
        self.ace = ACE()
        self.ace._raise_ZPHI_exceptions = _raise_ZPHI_exceptions
        self.ace._attempted_method_on_signal_extinction = _attempted_method_on_signal_extinction
        self.ace._verbose = _verbose
        self.ace._phidp_entangle_ws = _phidp_entangle_ws # default: 0 # no PhiDP modification (leads to worse alpha estimations)
        self.ace._entangle_Ah_half_ws = _entangle_Ah_half_ws # default: 2 # currently, the best value after testing all available cases (16)
    def make_att_corr_single_scan(self, data, metadata, scan_name='DATASET1',):
        """
        Corrects attenuation for a single scan.
        
        Arguments:
            data:           dict, containing the radar-variables (scanname : variable : 'data' : array).
            metadata:       dict, containing the meta-data of the radar sweep.
        Parameters:
            scan_name:      str, the name of the scan (determines the elevation). Default: 'DATASET1'
        Returns:
            att_cor_res:    dict, containing the attenuation correction results (intrinsic hor. reflectivity ("Z_in"), specific attenuation ("A_h"), difference in reflectivity factors before and after correction ("Z_diff"), and the attenuation coefficients ("alpha_map")).
        """
        att_cor_res = dict()
        new_phi_sys = radar_tools.est_phi_sys(data[scan_name]['UPHIDP']['data'], data[scan_name]['URHOHV']['data'])#, bin_threshold=5, wx_radials_threshold=10)
        if new_phi_sys is not None:
            self.cur_phi_sys = new_phi_sys
        res = self.ace.correct_scan(data,metadata, scan_name=scan_name, cur_phi_sys=self.cur_phi_sys)
        att_cor_res['Z_in'] = res[1]
        att_cor_res['A_h'] = res[0]
        att_cor_res['Z_diff'] = res[2]
        att_cor_res['alpha_map'] = res[3]
        return att_cor_res
    def make_att_corr_all_scans(self, data, metadata):
        """
        Corrects attenuation for all scans in a data-set.
        
        Arguments:
            data:               dict, containing the radar-variables (scanname : variable : 'data' : array).
            metadata:           dict, containing the meta-data of the radar sweep.
        Returns:
            att_cor_vol_data:   dict, attenuation correction results for each elevation (intrinsic hor. reflectivity ("Z_in"), specific attenuation ("A_h"), difference in reflectivity factors before and after correction ("Z_diff"), and the attenuation coefficients ("alpha_map")).
        """
        att_cor_vol_data = dict()
        for scan_name in metadata: # for scan_name in metadata.iterkeys():
            if not scan_name.startswith(u'SCAN'): continue
            att_cor_vol_data[metadata[scan_name]['el']] = self.make_att_corr_single_scan(data, metadata, scan_name=scan_name)
        return att_cor_vol_data

def dt_floor2hour(dt_obj):
    """Lowers the datetime of 'dt_obj' to the current hour. I.e. minutes, seconds and microseconds are set to zero."""
    return datetime.datetime(dt_obj.year, dt_obj.month, dt_obj.day, dt_obj.hour)

# ---

def make_att_cor_PPI_plot(att_cor_data, metadata, scan_name=None, scan_type='pcp', p_var='AC_ZH', ax=None):
    """
    Example function to plot attenuation corrected PPI data.
    
    Arguments:
        att_cor_data:       dict, containing attenuation correction results, with the 'scan_type' as key.
        metadata:           dict, containing the meta-data of the radar sweep.
    Parameters:
        scan_name:          str/None, name of the scan (meaning: the elevation) to use. If None, then the scan with the lowest elevation will be used. Defualt: None
        scan_type:          str, determine, whether precipitation scan data ("pcp") or volumetric data ("vol") is used. Currently only "pcp" is implemented. Default: 'pcp'
        p_var:              str, name of the variable to plot. Default: 'AC_ZH'
        ax:                 matplotlib.axes-object, axis, which is used to plot in.
    Returns:
        ax:                 matplotlib.axes-object, axis, which was used for plotting.
        pm:                 matplotlib.pcolormesh-object, the colored mesh, containing the plotted variable.
    """
    assert scan_type in ('pcp',)
    assert p_var in ('AC_ZH', 'A_h', 'alpha',)
    if ax is None:
        ax = plt.gca()
    if scan_type == 'pcp':
        if scan_name is None:
            scan_name = radar_tools.get_lowest_el_scan(metadata)
        _alt_k = {
            u'AC_ZH' : 'Z_in',
            u'A_h' : 'A_h',
            u'alpha' : 'alpha_map',
        }
        ppi_data = att_cor_data['pcp'][_alt_k[p_var]]
    elif scan_type == 'vol':
        pass # TODO: take scan_name or elevation are prepare to plot
    return plot_tools._make_ppi_plot(ppi_data, metadata, scan_name=scan_name, var=p_var, ax=ax)

# ---

def read_ODIM_DWD_hdf5(filename, wanted_elevations=None, wanted_moments=None):
    """Data reader for hdf5 files for single radar data provided by the german weather services (DWD) based on OPERA/ODIM.
    
    This data reader is based on read_GAMIC_hdf5 of the wradlib module.
    
    Arguments
    ---------
    filename : string
        path to the hdf5 file
    
    Parameters
    ----------
    wanted_elevations : None or list containing strings
        sequence of strings of elevation_angle(s) of scan (only needed for PPI). Defaults to None, which means 'all' elevations.
    wanted_moments : None or list containing strings
        sequence of strings of moment name(s).  Defaults to None, which means 'all' moments.
        
    Returns
    -------
    data : dict
        dictionary of scan and moment data (numpy arrays)
    attrs : dict
        dictionary of attributes
    """
    # check elevations
    if wanted_elevations is None:
        wanted_elevations = 'all'
    # check wanted_moments
    if wanted_moments is None:
        wanted_moments = 'all'
    # read the data from file
    with h5py.File(filename, 'r') as f:
        # placeholder for attributes and data
        attrs = {}
        vattrs = {}
        data = {}
        # check if OPERA file and ...
        try:
            swver = f['how'].attrs.get('software')
        except KeyError:
            print("WRADLIB: File is no OPERA hdf5!")
            raise
        # ... get scan_type (PVOL or RHI)
        scan_type = f['what'].attrs.get('object').decode()
        # single or volume scan
        if scan_type == 'PVOL':
            # loop over 'main' hdf5 groups (how, scanX, what, where)
            for top_group in list(f):
                if 'data' in top_group:
                    groups = f[top_group]
                    sg1 = groups['how'] # <- TODO: what's that for?
                    # get scan elevation
                    el = groups['where'].attrs.get('elangle')
                    el = str(round(el, 2))
                    # try to read scan data and attrs if wanted_elevations are found
                    if (wanted_elevations == 'all') or (el in wanted_elevations):
                        sdata, sattrs = read_odim_dwd_scan(scan=groups, scan_type=scan_type,
                                                           wanted_moments=wanted_moments)
                        if sdata:
                            data[top_group.upper()] = sdata
                        if sattrs:
                            attrs[top_group.upper()] = sattrs
        # single rhi scan
        elif scan_type == 'RHI':
            # loop over 'main' hdf5 groups (how, scanX, what, where)
            for top_group in list(f):
                if 'scan' in top_group:
                    groups = f[top_group]
                    # try to read scan data and attrs
                    sdata, sattrs = read_odim_dwd_scan(scan=groups, scan_type=scan_type,
                                                       wanted_moments=wanted_moments)
                    if sdata:
                        data[top_group.upper()] = sdata
                    if sattrs:
                        attrs[top_group.upper()] = sattrs
        else:
            raise KeyError("ERROR: Unknown scan_type '%s'! Should be either 'PVOL' or 'RHI'!" % (scan_type))
        # collect volume attributes if wanted data is available
        if data:
            vattrs['Latitude'] = f['where'].attrs.get('lat')
            vattrs['Longitude'] = f['where'].attrs.get('lon')
            vattrs['Height'] = f['where'].attrs.get('height')
            # check whether its useful to implement that feature
            # vattrs['sitecoords'] = (vattrs['Longitude'], vattrs['Latitude'], vattrs['Height'])
            attrs['VOL'] = vattrs
        else:
            print ("why didn't it stop?")
            raise ValueError("ERROR: no data.") # TODO: this exception is not helpful at all
    return data, attrs

def read_odim_dwd_scan(scan, scan_type, wanted_moments):
    """Read data from one particular scan of radar data provided by the german weather service (DWD) in OPERA/ODIM hdf5 data format.
    
    This data reader is based on read_gamic_scan of the wradlib module.
    
    Arguments:
    ----------
    scan : object
        scan object from hdf5 file
    scan_type : string
        "PVOL" (plan position indicator) or "RHI" (range height indicator)
    wanted_moments : string or list of strings
        sequence of strings containing upper case names of moment(s) to be returned or one string 'all'

    Returns
    -------
    data : dict
        dictionary of moment data (numpy arrays)
    sattrs : dict
        dictionary of scan attributes
    """
    # placeholder for data and attrs
    data = {}
    sattrs = {}
    # try to read wanted moments
    for mom in list(scan):
        if 'data' in mom:
            data1 = {}
            moment_group = scan[mom]
            #actual_moment = moment_group.attrs.get('moment').decode().upper()
            actual_moment = moment_group['what'].attrs['quantity'].decode().upper()
            if (wanted_moments == 'all') or (actual_moment in wanted_moments):
                # read attributes only once
                if not sattrs:
                    sattrs = read_odim_dwd_scan_attributes(scan, scan_type)
                mdata = moment_group['data'][...]
                #dyn_range_max = moment_group.attrs.get('dyn_range_max')
                dyn_range_min = moment_group['what'].attrs.get('offset')
                gain = moment_group['what'].attrs.get('gain')
                _mdata = dyn_range_min + mdata * gain
                _mdata[mdata == moment_group['what'].attrs.get('nodata')] = np.NaN
                mdata = _mdata
                if scan_type == 'PVOL':
                    # rotate accordingly
                    mdata = np.roll(mdata, -1 * sattrs['zero_index'], axis=0)
                if scan_type == 'RHI':
                    # remove first zero angles
                    sdiff = mdata.shape[0] - sattrs['el'].shape[0]
                    mdata = mdata[sdiff:, :]
                data1['data'] = mdata
                #data1['dyn_range_max'] = dyn_range_max
                data1['dyn_range_min'] = dyn_range_min
                data[actual_moment] = data1
    return data, sattrs

def read_odim_dwd_scan_attributes(scan, scan_type, attr_exclude_list=['startazA', 'startazT', 'startelA', 'stopazA', 'stopazT', 'stopelA']):
    """Read attributes from one particular scan of radar data provided by the german weather service (DWD) in OPERA/ODIM hdf5 data format.
    
    This data reader is based on read_gamic_scan_attributes of the wradlib module.
    
    Arguments
    ---------
    scan : object
        scan object from hdf5 file
    scan_type : string
        "PVOL" (plan position indicator) or "RHI" (range height indicator)
    
    Parameters
    ----------
    attr_exclude_list : list containing strings
        Name of attributes which shall be excluded and not copied to the attributes dictionary.
    
    Returns
    -------
    sattrs : dict
        dictionary of scan attributes
    """
    # placeholder for attributes
    sattrs = {}
    # get scan attributes
    for attrname in list(scan['where'].attrs):
        sattrs[attrname] = scan['where'].attrs.get(attrname)
    for attrname in list(scan['how'].attrs):
        if attrname not in attr_exclude_list:
            sattrs[attrname] = scan['how'].attrs.get(attrname)
    sattrs['bin_range'] = sattrs['rscale'] * sattrs['nbins']
    # get scan header
    #ray_header = scan['ray_header'] # doesn't exist
    ray_header = {
        'azimuth_start' : scan['how'].attrs.get('startazA'),
        'azimuth_stop' : scan['how'].attrs.get('stopazA'),
    }
    # az, el, zero_index for PPI scans
    if scan_type == 'PVOL':
        azi_start = ray_header['azimuth_start']
        azi_stop = ray_header['azimuth_stop']
        # Azimuth corresponding to 1st ray
        if (azi_stop < azi_start).any():
            zero_index = np.where(azi_stop < azi_start)
            azi_stop[zero_index[0]] += 360
            zero_index = zero_index[0] + 1
        else:
            zero_index = np.array([0])
        az = (azi_start + azi_stop) / 2
        #az = np.roll(az, -zero_index, axis=0)
        az = np.round(az, 1)
        el = scan['where'].attrs.get('elangle')
    # az, el, zero_index for RHI scans
    elif scan_type == 'RHI':
        pass # TODO: RHI not supported, yet
        raise Exception("ERROR: RHI is not supported, yet!")
        ele_start = np.round(ray_header['elevation_start'], 1)
        ele_stop = np.round(ray_header['elevation_stop'], 1)
        angle_step = np.round(sattrs['angle_step'], 1)
        angle_step = np.round(sattrs['ele_stop'], 1) / angle_step
        # Elevation corresponding to 1st ray
        if ele_start[0] < 0:
            ele_start = ele_start[1:]
            ele_stop = ele_stop[1:]
        zero_index = np.where(ele_stop > ele_start)
        zero_index = zero_index[0]  # - 1
        el = (ele_start + ele_stop) / 2
        el = np.round(el, 1)
        el = el[-angle_step:]
        az = sg1.attrs.get('azimuth')
    else:
        raise KeyError("ERROR: Unknown scan_type '%s'! Should be either 'PVOL' or 'RHI'!" % (scan_type))
    # save zero_index (first ray) to scan attributes
    sattrs['zero_index'] = zero_index[0]
    # create range array
    r = np.arange(sattrs['rstart'], sattrs['nbins'] * sattrs['rscale'] + sattrs['rstart'],sattrs['rscale'])
    # save variables to scan attributes
    sattrs['az'] = az
    sattrs['el'] = el
    sattrs['r'] = r
    #sattrs['Time'] = scan['what'].attrs.get('time') # <-- NONE!!
#     dt_obj = datetime.datetime.strptime(scan['what'].attrs.get('startdate') + '_' + scan['what'].attrs.get('starttime'), '%Y%m%d_%H%M%S')
    dt_obj = datetime.datetime.strptime(scan['what'].attrs.get('startdate').decode() + '_' + scan['what'].attrs.get('starttime').decode(), '%Y%m%d_%H%M%S')
    sattrs['Time'] = dt_obj.strftime('%Y-%m-%dT%H:%M:%S.' + '{:03.0f}Z'.format(dt_obj.microsecond / 1000.0))
    sattrs['max_range'] = r[-1]
    return sattrs


def gen_multi_scan_data(full_path, highest_scan_num=10, verbose=False, _rewrite_lines=True):
    """
    Combines multiple single-scan files into a single dataset. For each 5min time frame in the file, a dataset and a metadata dict is yielded as generator.
    """
    _ignore_sub_files = [
        './mvol_20130806172100_10950_mem_vol-op-sidpol-01_0000', # truncated
        './mvol_20140429130058_10557_neu_vol-op-sidpol-01_0000', # missing other elevations
        './mvol_20140429130558_10557_neu_vol-op-sidpol-01_0000', # radar maintance
        './mvol_20140610214534_10410_ess_pcp-op-sidpol-01_0000', # truncated
        './mvol_20160624120558_10605_nhb_vol-op-sidpol-01_0000', # missing other elevations
    ]
    _skip_until_next_scan_set = False
    def write_msg(msg):
        if verbose:
            if _rewrite_lines:
                print ("\r" + _warn_msg)
            else:
                print (_warn_msg)
    #
    with tarfile.open(full_path, mode='r') as tar:
        # find all members in the file
        mem_names = []
        for i, member in enumerate(tar.getmembers()):
            mem_names.append([member.name, i])
        last_scan_num = None
        for name, i in sorted(mem_names):
            # check if this file is subject to be ignored
            if name in _ignore_sub_files:
                _warn_msg = "Ignoring %s (ignore sub-file-list)!" % name
                write_msg(_warn_msg)
                _skip_until_next_scan_set = True
                continue
            if verbose:
                _msg = " ".join(["reading", name, str(i)])
                if _rewrite_lines:
                    _left_spaces = max(0, len(_msg) - 100)
                    sys.stdout.write('\r' + _msg + ' ' * _left_spaces)
                    sys.stdout.flush()
                else:
                    print (_msg)
            # construct a name for the scan/elevation
            scan_num = int(name.split('_')[-1]) + 1
            if _skip_until_next_scan_set and scan_num != 1:
                _warn_msg = "Ignoring %s (no new scan-set after ignored sub-file)!" % name
                write_msg(_warn_msg)
                continue
            if scan_num == 1:
                if (last_scan_num is None or last_scan_num == highest_scan_num):
                    _skip_until_next_scan_set = False
                    ms_data = {}
                    ms_metadata = {}
                else:
                    raise ValueError("ERROR: Invalid scan number order! Last scan number was %s and currently (file-member %s) it is %s! Check settings!" % (last_scan_num, i, scan_num))
            elif last_scan_num is None:
                raise ValueError("ERROR: Invalid scan number order! First scan number was not 0 (but %s)!" % (scan_num))
            elif last_scan_num + 1 != scan_num:
                raise ValueError("ERROR: Invalid scan number order! Scan numbers are not ascending! Last scan number was %s and currently (file-member %s) it is %s! Check settings!" % (last_scan_num, i, scan_num))
            # extract member into a temporary file to allow hdf5 routines to read the data
            member = tar.getmembers()[i]
            f = tar.extractfile(member)
            with tempfile.NamedTemporaryFile(delete=True) as temp_f:
                temp_f.write(f.read())
                temp_f.flush()
                temp_fn = temp_f.name
                if not os.path.exists(temp_fn):
                    print ("temp file", temp_fn, "exists?", os.path.exists(temp_fn) )
                # read radar data and put in wradlib compatible format
                data, metadata = read_ODIM_DWD_hdf5(temp_fn)
            last_scan_num = scan_num
            # combine new scan with other scan data
            if 'VOL' in ms_metadata:
                if ms_metadata['VOL'] != metadata['VOL']:
                    raise ValueError("ERROR: 'VOL' entry of metadata differs! Current multi-scan-dict: %s, new scan (#%s): %s" % (ms_metadata['VOL'], scan_num, metadata['vol']))
            elif 'VOL' in metadata:
                ms_metadata['VOL'] = metadata['VOL']
            scan_name = 'SCAN%s' % (scan_num)
            ms_metadata[scan_name] = metadata['DATASET1']
            ms_data[scan_name] = data['DATASET1']
            if scan_num == highest_scan_num: # all elevations done? Then yield whole time-frame.
                yield (ms_data, ms_metadata)
        else:
            # clear last line
            if verbose and _rewrite_lines:
                sys.stdout.write('\r' + ' ')
                sys.stdout.flush()

# ---

# Att. - Cor.
class ACE(object):
    """
    Advanced attenuation correction for correcting anomalous high attenuation in hail.
    """
    alpha_0=0.06
    ds=0.25
    _valid_area_min_length=2000.
    _min_radials_past_hc=4 # tweaking of this parameter might be necessary for volumetric data
    _rho_hv_va_threshold=0.5
    _rho_hv_hs_threshold=0.6
    _repair_gapped_areas=True
    _raise_ZPHI_exceptions=True
    _attempted_method_on_signal_extinction=None
    _allow_phi_sf_4_hs=True
    _use_noise_surpression4phi=True
    var_refl = u'DBZH' # variable name in data for horizontal reflecitivty factor in dBZ
    var_phidp = u'UPHIDP' # variable name in data for uncorrected differental phase in deg
    var_zdr = u'ZDR' # variable name in data for differential reflectivity in dB
    var_rhohv = u'URHOHV' # variable name in data for uncorrected co-polar correlation coefficient
    _phi_opt = 0 # PhiDP should ideally be at 0 if no propagation effects took place
    _phidp_entangle_ws = 0
    _entangle_Ah_half_ws = 2
    _verbose=False
    _valid_area_Z_thres = 10
    _valid_area_use_sf_rho = True
    _valid_area_phi_thres = 0.
    _hotspot_use_sf_rho = True
    def __init__(self):
        self._tse_flags_dict = None
        pass
    def get_current_tse_flags(self):
        """
        Returns the total-signal extinction flags.
        """
        if self._tse_flags_dict is None:
            raise ValueError("ERROR: TSE-flags cannot be treived before a scan has been corrected! Run .correct_scan first!")
        return self._tse_flags_dict
    def correct_scan(self,data, metadata, scan_name, cur_phi_sys):
        """
        Determines attenuation for a radar scan (a single elevation sweep).
        
        Arguments:
            data:           dict, containing the radar-variables (scanname : variable : 'data' : array).
            metadata:       dict, containing the meta-data of the radar sweep.
            scan_name:      str, name of the current radar scan/elevation.
            cur_phi_sys:    float, system differential phase.
        Returns:
            it_Ah:          array, interpolated specific attenuation.
            Z_in:           array, attenuation corrected (and therefore) intrinsic hor. reflectivity factor.
            Z_diff:         array, difference in reflectivity factors before and after correction.
            alpha_map:      array, attenuation coefficient.
        """
        assert self._attempted_method_on_signal_extinction in [None, 'extrapolate', 'guess_alpha']
        # prepare empty arrays
        A_h = np.zeros_like(data[scan_name][self.var_refl]['data']) * np.NaN
        Z_in = np.zeros_like(A_h) + data[scan_name][self.var_refl]['data']
        Z_in[Z_in == np.nanmin(Z_in)] = np.NaN
        Z_diff = np.zeros_like(A_h) * np.NaN
        alpha_map = np.zeros_like(Z_in) * np.NaN
        alg_rain = att_cor_tools.ZPHI()
        alg_hs = att_cor_tools.HotspotZPHI()
        alg_hs._min_radials_past_hc = self._min_radials_past_hc
        unfold_PhiDP = np.empty_like(data[scan_name][self.var_phidp]['data'])
        self._tse_flags_dict = dict()
        # check if multiple PhiDP-azimuth filters should be used (multi-pass filter)
        if utils.isiterable(self._phidp_entangle_ws):
            _filter_it = self._phidp_entangle_ws
        else:
            if self._phidp_entangle_ws is None or self._phidp_entangle_ws == 0:
                _filter_it = ()
            else:
                _filter_it = [self._phidp_entangle_ws,]
        _corrected_va_ranges = dict()
        # unfold PhiDP for scan
        for r_ind in range(unfold_PhiDP.shape[1]):
            unfold_PhiDP[:,r_ind] = radar_tools.phidp_unfolding(data[scan_name][self.var_phidp]['data'][:,r_ind],phi_sys=cur_phi_sys,phi_opt=self._phi_opt)
            # entangle azimuths of PhiDP
            for _ws in _filter_it: # allow multi-pass filter
                unfold_PhiDP[:,r_ind] = radar_tools.spike_filter_med(unfold_PhiDP[:,r_ind],_ws)
            pass # TODO: this might cause issues at 360/0deg 
        # go through every azimuth and start correcting
        for az_ind in range(metadata[scan_name]['az'].shape[0]):
            Z_H = data[scan_name][self.var_refl]['data'][az_ind].copy()
            Z_H[Z_H == np.nanmin(Z_H)] = np.NaN
            rho_hv = data[scan_name][self.var_rhohv]['data'][az_ind]
            ZDR = data[scan_name][self.var_zdr]['data'][az_ind]
            Phi_DP = unfold_PhiDP[az_ind]
            if self._use_noise_surpression4phi:
                Phi_DP = att_cor_tools.surpress_noisy_phi(Phi_DP, _verbose=self._verbose)
            Phi_sf = np.asarray(radar_tools.spike_filter_med(Phi_DP))
            # repair huge, negative spikes in PhiP
            if np.count_nonzero(Phi_sf >= 0): # without, the interpolation sample would be zero
                Phi_DP[Phi_sf < 0] = np.interp(np.nonzero(Phi_sf < 0)[0], np.arange(len(Phi_sf))[Phi_sf >= 0],Phi_sf[Phi_sf >= 0]) # fills gaps with interpolated values (using spike-filtered values)
                # repeat this line for the spike-filtered PhiDP
                Phi_sf[Phi_sf < 0] = np.interp(np.nonzero(Phi_sf < 0)[0], np.arange(len(Phi_sf))[Phi_sf >= 0],Phi_sf[Phi_sf >= 0])
            #
            _corrected_va_ranges[az_ind] = []
            # identify precipitation areas
            valid_areas = att_cor_tools.identify_hotspot(Z_H, rho_hv, ZDR, Phi_sf, Z_threshold=self._valid_area_Z_thres, rho_threshold=self._rho_hv_va_threshold, _use_sf_rho=self._valid_area_use_sf_rho, phi_threshold=self._valid_area_phi_thres, _check_ZDR=False, ds=self.ds*1000., len_threshold=self._valid_area_min_length)
            if len(valid_areas) == 0:
                if self._verbose: print ("DEBUG: no valid_areas found for azimuth %s. Skipping" % (str(metadata[scan_name]['az'][az_ind]).zfill(3)) )
                continue
            # repair seperated areas if some exist
            if self._repair_gapped_areas:
                valid_areas, Z_H, Phi_DP = att_cor_tools.repair_gapped_areas(valid_areas, Z_H, Phi_DP, rho_hv, _threshold4repair_gap_min_rho=self._rho_hv_va_threshold/2., _threshold4repair_gap_max_range=max(2,int(1./float(self.ds))),_verbose=self._verbose, _dbg_az_ind=az_ind)
            # find hotspots
            hotspots = list(att_cor_tools.identify_hotspot(Z_H, rho_hv, ZDR, Phi_DP, ds=self.ds*1000., rho_threshold=self._rho_hv_hs_threshold, _use_sf_rho=self._hotspot_use_sf_rho, _use_sf_phi=self._allow_phi_sf_4_hs))
            #
            A_h[az_ind] = np.zeros_like(A_h[az_ind])
            inds_va_with_hs = att_cor_tools.get_va_containing_hs_inds(valid_areas, hotspots)
            for va_ind, hs_inds in enumerate(inds_va_with_hs):
                r_0 = valid_areas[va_ind][0]
                r_m = valid_areas[va_ind][-1]
                # do ZPHI correction (for rain)
                alg_rain.calc_attenuation(radar_tools.dBZ2z(Z_H), Phi_DP, r_0, r_m, self.alpha_0, ds=self.ds)
                A_h[az_ind] += alg_rain.get_attenuation()
                Z_in[az_ind,:] = alg_rain.correct_attenuation(Z_H, r_0, r_m, A_h=A_h[az_ind], _cor_beyond_rm=True, ds=self.ds)
                # Hotspot/ZPHI (for hail)
                if hs_inds is not None:
                    for hotspot in np.asarray(hotspots)[hs_inds]:
                        r_1 = hotspot[0]
                        r_2 = hotspot[-1]
                        #
                        try: # ...determining an optimal alpha
                            alpha_map[az_ind, r_1:r_2] = alg_hs.find_delta_alpha(radar_tools.dBZ2z(Z_H), Phi_sf if self._allow_phi_sf_4_hs else Phi_DP, r_0, r_1, r_2, r_m, alpha_0=self.alpha_0, ds=self.ds) + self.alpha_0
                        except (att_cor_tools.NotEnoughRadialsOutsideHailcore) as e:
                            if self._raise_ZPHI_exceptions: raise
                            if alg_hs._dbg_total_signal_extinction_flag: self._sub_helper_add_tse(self._tse_flags_dict, az_ind, r_m) # store information about likely total signal extinction
                            if self._attempted_method_on_signal_extinction == 'guess_alpha':
                                _delta_Phi = Phi_DP[r_2] - Phi_DP[r_1]
                                if r_m+1 < len(Z_H) and Z_H[r_m+1] >= 10.0: # maybe there is still a valid Z value after
                                    _assumed_Att = Z_H[r_m+1]
                                else:
                                    _assumed_Att = Z_H[r_m] - 10.0
                                _guessed_alpha = _assumed_Att / _delta_Phi
                                if self._verbose: print ("INFO: '%s' was caught (at az_ind: %s). Total signal exctinction is propable and alpha was therefore guessed to be %s" % (e.__class__.__name__, az_ind, _guessed_alpha) )
                                alg_hs._res_delta_alpha = min(1.0, _guessed_alpha)
                                alpha_map[az_ind, r_1:r_2] = alg_hs._res_delta_alpha
                                alg_hs._res_A_h = alg_hs._calculate_specific_attenuation(alg_hs._res_delta_alpha, r_0, r_1, r_2, r_m, Phi_DP, radar_tools.dBZ2z(Z_H), ds=self.ds, alpha_0=self.alpha_0)
                            else:
                                if self._verbose: print ("ERROR: ZPHI-Exception at az_ind %s:" % (az_ind), e )
                                continue
                        if alg_hs._dbg_had2use_extrapolation is True:
                            if self._verbose: print ("WARNING: Total signal exctinction (at az_ind: %s) is propable and alpha was calculated to %s using extrapolation." % (az_ind, alg_hs._res_delta_alpha) )
                            self._sub_helper_add_tse(self._tse_flags_dict, az_ind, r_m)
                        A_h[az_ind] += alg_hs.get_attenuation()
                        #Z_in[az_ind,:] = alg_hs.correct_attenuation(Z_H, r_0, r_m, A_h=A_h[az_ind], _cor_beyond_rm=True, ds=self.ds) # will be done afterwards, after A_h is interpolated
                        _corrected_va_ranges[az_ind].append((r_0,r_m)) # need to store them for later correction
                #Z_diff[az_ind,:] = np.cumsum(2.*A_h[az_ind,:]*self.ds) # will be done afterwards, after A_h is interpolated
            # after each az_ind reset algorithm results
            alg_rain.reset_results()
            alg_hs.reset_results()
        # interpolate specific Attenuation field
        if self._entangle_Ah_half_ws:
            #it_Ah = broaden_attenuation_field(A_h)
            it_Ah = smooth_attenuation_field(A_h, half_win_size=self._entangle_Ah_half_ws) # yielded best results
        else:
            it_Ah = A_h
        # correct reflectivity
        for az_ind in range(metadata[scan_name]['az'].shape[0]):
            Z_H = data[scan_name][self.var_refl]['data'][az_ind].copy()
            Z_H[Z_H == np.nanmin(Z_H)] = np.NaN
            for r_0, r_m in _corrected_va_ranges[az_ind]:
                Z_in[az_ind,:] = alg_hs.correct_attenuation(Z_H, r_0, r_m, A_h=it_Ah[az_ind], _cor_beyond_rm=True, ds=self.ds)
            Z_diff[az_ind,:] = np.cumsum(2.*it_Ah[az_ind,:]*self.ds)
        #
        #return A_h, Z_in, Z_diff, alpha_map
        return it_Ah, Z_in, Z_diff, alpha_map
    def _sub_helper_add_tse(self, _tse_dict, az_ind, r_ind):
        if az_ind not in _tse_dict.keys():
            _tse_dict[az_ind] = []
        _tse_dict[az_ind].append(r_ind)
def broaden_attenuation_field(A_h, _b_fac=0.5):
    # assume found Ah is correct.
    # assume some rays might have the same, but weren't successfully detected (e.g. because noise)
    # interpolate in rays without Ah in /\ shape. Take always maximum of existing value and /\ value.
    ac_field = A_h[...,np.newaxis]
    _b_facts_vec = [_b_fac,1.,_b_fac]
    #it_ac_field = np.nanmax(np.concatenate([np.roll(ac_field, -1, axis=0)*0.5, ac_field, np.roll(ac_field, +1, axis=0)*0.5], axis=-1),axis=-1)
    it_ac_field = np.nanmax(np.concatenate([np.roll(ac_field, s, axis=0)*_b_facts_vec[i] for i,s in enumerate([-1,0,1])], axis=-1),axis=-1)
    return it_ac_field
def smooth_attenuation_field(A_h, half_win_size=2):
    """
    Use a convolution function to smooth the specific attenuation field to reduce the effect of over-corrections and noise-induced under-corrections.
    """
    it_ah = np.empty_like(A_h)
    for r_ind in range(A_h.shape[1]):
        pp_A_vec = period_pad_vec(A_h[:,r_ind], half_win_size)
        it_ah[:,r_ind] =  np.convolve(pp_A_vec,np.ones((half_win_size*2,))/float(half_win_size*2), mode='valid')[1:]
    return it_ah
def period_pad_vec(vec, N_half):
    """
    Pad a vector by values of the opposite end to make the resulting vector look partially periodic.
    E.g. period_pad_vec([1,2,3,4,5],N_half=2) would return [4,5,1,2,3,4,5,1,2].
    """
    pp_vec = np.zeros((vec.shape[0]+N_half*2,))
    pp_vec[N_half:-N_half] += vec
    pp_vec[:N_half] = vec[-N_half:]
    pp_vec[-N_half:] = vec[:N_half]
    return pp_vec

# ZCH calc
def calc_vi_zch_data(data, metadata, freezing_height, radar_name, zdr_threshold=2.0, dBZ_masking=30.0, _disable_print=False, _disable_warnings=False, _num_ip_elevs=30):
    """
    Calculates vertical interpolated ZDR-column data.
    
    Arguments:
        data:               dict, containing the radar-variables (scanname : variable : 'data' : array).
        metadata:           dict, containing the meta-data of the radar sweep.
        freezing_height:    float, height of the 0 degree Celcius isotherm in meters.
        radar_name:         str, name of the radar station. Required for determining the height of the station.
    Parameters:
        zdr_threshold:      float, ZDR-threshold to use to count as part of the ZDR-column. Default: 2.0
        dBZ_masking:        float, reflectivity factor threshold each voxel of the ZDR-column needs to exceed to be a valid part of the ZDR-column. This is to ensure to be inside a precipitating cloud and to avoid noisy voxels. Default: 30.0
        _disable_print:     bool, disable messages. Default: False
        _disable_warning:   bool, disable warnings from other modules used inside. Default: False
        _num_ip_elevs:      int, number of elevations to interpolate to. Default: 30
    Returns:
        vi_param_data_zc:   dict, results of the ZDR-column detection in the vertical interpolated data (PPI of the height of the ZDR-columns ("ZC_height"), PPI of the mean ZDR-value of the ZDR-columns ("ZC_meanval"), PPI of the maximum ZDR-value of the ZDR-columns ("ZC_maxval"), and a list of all detected ZDR-columns with the elevation, azimuth, and range index of each voxel being part of the ZDR-column ("ZC_traces")).
    """
    if _disable_warnings:
        import warnings
        warnings.filterwarnings("ignore")
    if dBZ_masking is None:
        _vi_vars = [u'ZDR']
    else:
        _vi_vars = [u'ZDR', u'DBZH']
    vi_param_data_zc = dict()
    vi_voldata, vi_volmeta = vertical_interpolate_voldata(data, metadata, desired_elev_res=_num_ip_elevs, do_vars=_vi_vars)
    #
    aci = hail_precursor_tools.AdvColumnIdentifier(vi_voldata, vi_volmeta, 'ZDR', freezing_height, station_height=network_info.station_info[radar_name]['height'], dB_threshold=zdr_threshold, _dBZ_masking_threshold=dBZ_masking)
    aci.find_columns()
    zc_traces = aci.get_traces()
    zc_height_map = aci.get_col_height_map()
    zc_meanval_map = aci.get_col_meanval_map()
    zc_maxval_map = aci.get_col_maxval_map()
    vi_param_data_zc = {
        'ZC_height' : zc_height_map,
        'ZC_meanval' : zc_meanval_map,
        'ZC_maxval' : zc_maxval_map,
        'ZC_traces' : zc_traces,
    }
    return vi_param_data_zc
# vertical interpolation algorithms
def vertical_interpolate_rhi(rhi_data, rhi_meta, desired_elev_res=30):
    """
    Vertically interpolate a RHI.
    """
    assert hasattr(rhi_data, 'shape') and hasattr(rhi_data, 'dtype')
    new_shape = (desired_elev_res, rhi_data.shape[1])
    new_elev = np.interp(np.linspace(0,len(rhi_meta['th'])-1,desired_elev_res), np.arange(len(rhi_meta['th'])),rhi_meta['th'])
    vi_data = np.zeros(new_shape, dtype=rhi_data.dtype) * np.NaN
    for r_i in range(rhi_data.shape[1]):
        vi_data[:,r_i] = np.interp(new_elev,rhi_meta['th'],rhi_data[:,r_i])
    # modify metadata
    vi_meta = dict()
    vi_meta.update(rhi_meta)
    vi_meta['th'] = new_elev
    return vi_data, vi_meta
def vertical_interpolate_voldata(data, metadata, desired_elev_res=30, do_vars=[u'ZDR'], _scan_name_suffix='SCAN'):
    """
    Vertically interpolate a whole volumetric dataset.
    """
    ascending_elev, scan_elev_map = radar_tools.calc_elev_order(metadata)
    lowest_scan_name = radar_tools.get_lowest_el_scan(metadata)
    # set up pseudo data dicts
    _create_pseudo_scan_dict = lambda: {_scan_name_suffix+"%s" % (np.round(sk,4)) : dict() for sk in np.linspace(0, len(scan_elev_map)-1,desired_elev_res)}
    vi_voldata = _create_pseudo_scan_dict(); vi_volmeta = _create_pseudo_scan_dict()
    for sk in vi_volmeta:
        vi_volmeta[sk]['az'] = metadata[lowest_scan_name]['az']
        vi_volmeta[sk]['r'] = metadata[lowest_scan_name]['r']
        vi_volmeta[sk]['el'] = None # also need el(evation), this will be given after interpolation
        vi_volmeta[sk][u'nbins'] = metadata[lowest_scan_name][u'nbins'] if u'nbins' in metadata[lowest_scan_name] else metadata[lowest_scan_name]['bin_count']
        for var in do_vars:
            vi_voldata[sk][var] = dict()
            vi_voldata[sk][var]['data'] = np.zeros_like(data[lowest_scan_name][var]['data']) * np.NaN
    # iter through all azimuths and interpolate vertically
    for az_ind, azi in enumerate(metadata[lowest_scan_name]['az']):
        for var in do_vars:
            rhi_data, rhi_meta = radar_tools.calc_rhi_data(data, metadata, azi, var, ascending_elev=ascending_elev, scan_elev_map=scan_elev_map)
            rhi_data[rhi_data <= -31] = np.NaN
            vi_data, vi_meta = vertical_interpolate_rhi(rhi_data, rhi_meta, desired_elev_res=desired_elev_res)
            assert len(vi_meta['th']) == len(vi_volmeta.keys())
            for el_in, sk in enumerate(sorted(vi_volmeta.keys())):
                vi_voldata[sk][var]['data'][az_ind,:] = vi_data[el_in,:]
    # now add the elevation information to the new meta (instead of writing it every time)
    for el_in, sk in enumerate(sorted(vi_volmeta.keys())):
        vi_volmeta[sk]['el'] = vi_meta['th'][el_in]
    #
    return vi_voldata, vi_volmeta

class FreezingHeightHelper(object):
    """
    Provides offline freezing heights for testing the ForecastSuite.
    """
    _ew_melting_heights_fp = './data/melting_heights.npy'
    def __init__(self, _ew_melting_heights_fp=None):
        if _ew_melting_heights_fp is not None:
            self._ew_melting_heights_fp = _ew_melting_heights_fp
        self.ew_melting_heights = np.load(self._ew_melting_heights_fp, allow_pickle=True, encoding = 'latin1').item()
    def get_closest_melting4dt(self, cur_dt, radar_name, verbose=False):
        avail_dts = list(self.ew_melting_heights.keys())
        while len(avail_dts) > 0:
            i = np.argmin(abs(np.asarray(avail_dts) - cur_dt))
            poss_dt = avail_dts[i]
            if radar_name in self.ew_melting_heights[poss_dt].keys():
                # found fitting result
                if verbose: print ("DEBUG: Found for radar '%s' data at %s" % (radar_name, poss_dt) )
                return self.ew_melting_heights[poss_dt][radar_name]
            else: # radar not available for timestep
                if verbose: print ("DEBUG: Removing dt '%s'" % (poss_dt) )
                del avail_dts[i]
        else:
            raise ValueError("ERROR: No melting-data is available for radar '%s'!" % (radar_name))

# --------------------------

def get_event_infos(selected_case):
    """
    Provides all necessary information to start an offline test.
    """
    _event_db = {
        'A' : (datetime.datetime(2016,6,24,), 'ess', datetime.time(10,0), 60),
        'B' : (datetime.datetime(2016,6,24,), 'nhb', datetime.time(9,15), 33),#69),
        'C' : (datetime.datetime(2016,6,24,), 'tur', datetime.time(15,0), 12),
    }
    assert selected_case in _event_db
    event_dt, chosen_radar, _start_time, _num_of_frames = _event_db[selected_case]
    _start_dt = datetime.datetime.combine(event_dt.date(), _start_time) # info taken from config
#     frames_dt_objs = [_start_dt + datetime.timedelta(minutes=5*i) for i in range(_num_of_frames)]
    frames_dt_objs = [_start_dt + datetime.timedelta(minutes=5*i) for i in range(_num_of_frames)]
    return event_dt, chosen_radar, frames_dt_objs

# These parameters determine the membership functions for the hail prediction tool.
_mf_param = {
    'ZCH': {
        'giant': np.array([ 1.76124669e+00,  2.29372918e+00, -2.12696912e+06,  1.21556075e+08]),
        'large': np.array([0.81639542, 1.78488863, 1.75304223, 2.52822438]),
        'small': np.array([-9.18659196e+07,  4.00916682e+06,  1.11882444e+00,  1.92552588e+00]),
    },
    'ZCmZDR': {
        'giant': np.array([ 4.07675646e+00,  5.82551889e+00, -8.64916125e+06,  3.07238639e+08]),
        'large': np.array([3.13142441, 4.56883097, 4.40407081, 6.50471682]),
        'small': np.array([-2.81582346e+08,  9.73296820e+06,  3.59125605e+00,  4.51498173e+00]),
    },
}

def run_offline_event(event_dt, chosen_radar, frames_dt_objs, _save_attcor_plots=True, _save_hafer_plots=True):
    """
    Example function for running hail forecasting offline.
    
    Arguments:
        event_dt:           datetime, date and time of event to plot.
        chosen_radar:       str, name of the radar to take data and domain area from.
        frames_dt_objs:     list of datetime-objects, date and time of each radar data frame.
    Parameters:
        _save_attcor_plots: bool, save attenuation corrected reflectivity factor PPIs. Default: True
        _save_hafer_plots:  bool, save hail forecasting PPIs. Default: True
    Returns:
        None
    """
    # initialize data and objects
    # prepare data accessors and provider
    konrad_accessor = KONRAD_Accessor(os.path.join(konrad_base_path, 'KONRAD3D_20160623_24.tar.gz'))
    tracking_provider = TrackingDataProvider(konrad_accessor)
    radardata_provider = RadarDataProvider(_dbg_dt_lim=(dt_floor2hour(frames_dt_objs[0]), frames_dt_objs[-1])) # need to use the debug option here, as more radar data is available than KONRAD-tracks stored
    # create dummies, which will be set at first iteration
    grid_info_dict = None
    hfs = None
    # --------------
    # run full event
    for pcp_dt, vol_dt in radardata_provider.gen_data_updates(event_dt, chosen_radar):
        print ("time after raw-data is loaded", radardata_provider._dt_after_rawdata_read )
        print ("pcp-time:", pcp_dt, "vol-time:", vol_dt, "server-time:", datetime.datetime.utcnow() )
        # prepare grid and nowcasting suite
        if grid_info_dict is None:
            print ("DEBUG: initializing grid")
            grid_info_dict = prepare_grid_info(radardata_provider.get_meta_data()['pcp'], chosen_radar)
        if hfs is None:
            print ("DEBUG: initializing ForecastSuite")
            hfs = ForecastSuite(chosen_radar, grid_info_dict, _mf_param)
        #
        coarse_dt = datetime.datetime(pcp_dt.year, pcp_dt.month, pcp_dt.day, pcp_dt.hour, pcp_dt.minute) # need to remove seconds to work with KONRAD
        tracked_cells = tracking_provider.get_cells_at_dt(coarse_dt)
        # check if cells still exist, else sort them out
        active_cells = []
        for cell in tracked_cells:
            if coarse_dt not in cell.local_ids_at_frame: # use dt-obj without seconds
                #print "DEBUG: %s cell %s not in current timestep!" % (pcp_dt, cell.global_id)
                continue
            else:
                active_cells.append(cell)
        if len(active_cells) == 0:
            print ("DEBUG: no active cells!!")
            raise ValueError("ERROR: no active cells!")
        pcp_data = radardata_provider.get_pcp_data()
        att_cor_data = radardata_provider.get_att_cor_data()
        alpha_data = {el : {'alpha' : att_cor_data['vol'][el]['alpha_map']} for el in att_cor_data['vol']}
        vi_zc_data = radardata_provider.get_vi_zc_data()
        hfs.feed_in(coarse_dt, active_cells, pcp_data, alpha_data, vi_zc_data)
        detection_flags = hfs.get_detection()
        prediction_flags = hfs.get_prediction()
        pcp_metadata = radardata_provider.get_meta_data()['pcp']
        print ("forecast done, start plotting (%s)" % (datetime.datetime.utcnow().strftime('%H:%M:%S.%f')) )
        # plotting konrad tracking with hail forecasting
        with penv(show=False) as (fig, ax_arr):
            ax = ax_arr[0,0]
            plot_cells_and_forecasts(pcp_data, pcp_metadata, chosen_radar, coarse_dt, active_cells, detection_flags, prediction_flags, ax=ax, fig=fig)
            if _save_hafer_plots:
                savename = 'hafer_konrad_ru_%s_%s_%s.png' % (coarse_dt.strftime('%Y%m%d'), chosen_radar, coarse_dt.strftime('%H%M'))
                fig.savefig(os.path.join(plot_save_path, savename), dpi=300, bbox_inches='tight')
        # plot attenuation-corrected reflectivity factor, too
        with penv(show=False) as (fig, ax_arr):
            ax = ax_arr[0,0]
            make_att_cor_PPI_plot(att_cor_data, radardata_provider.get_meta_data()['pcp'], ax=ax)
            if _save_attcor_plots:
                savename = 'att-cor_ru_%s_%s_%s.png' % (coarse_dt.strftime('%Y%m%d'), chosen_radar, coarse_dt.strftime('%H%M'))
                fig.savefig(os.path.join(plot_save_path, savename), dpi=300, bbox_inches='tight')
        #
    #



if __name__ == "__main__":
    # choose event, either 'A', 'B', or 'C' here ------------\
    event_dt, chosen_radar, frames_dt_objs = get_event_infos('A')
    #
    run_offline_event(event_dt, chosen_radar, frames_dt_objs, _save_attcor_plots=True, _save_hafer_plots=True)



