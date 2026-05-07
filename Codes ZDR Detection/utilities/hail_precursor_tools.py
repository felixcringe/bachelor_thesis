#! /usr/bin/env python

# import radar_tools
from utilities import radar_tools
import wradlib
import numpy as np
from utilities import Namespace # Ric commented it

# --- Hail Precursor tools based on ZDR and CDR
def identify_ZDR_columns(data, metadata, freezing_height, station_height, dB_threshold=2.0):
    ascending_elev, scan_elev_map = radar_tools.calc_elev_order(metadata)
    scan_name = radar_tools.get_lowest_el_scan(metadata)
    rhi_data, rhi_meta = radar_tools.calc_rhi_data(data, metadata, 0.0, 'ZDR', ascending_elev=ascending_elev, scan_elev_map=scan_elev_map)
    beam_alt = wradlib.georef.bin_altitude(rhi_meta['r'], np.asarray(rhi_meta['th'])[:, np.newaxis], station_height, re=6370040.)
    zc_height_map = np.zeros_like(data[scan_name]['ZDR']['data']) * np.NaN
    zc_meanval_map = np.zeros_like(zc_height_map) * np.NaN
    azis = metadata[scan_name]['az']
    for az_ind, azi in enumerate(azis):
        rhi_data, rhi_meta = radar_tools.calc_rhi_data(data, metadata, azi, 'ZDR', ascending_elev=ascending_elev, scan_elev_map=scan_elev_map)
        zc_height_map[az_ind], zc_meanval_map[az_ind] = detect_columns_along_ray(rhi_data, beam_alt, freezing_height, val_threshold=dB_threshold)
    return zc_height_map, zc_meanval_map

def detect_columns_along_ray(rhi_data, beam_alt, freezing_height, val_threshold=2.0):
    column_heights = np.zeros(beam_alt.shape[1]) * np.nan
    column_mean_vals = np.zeros(beam_alt.shape[1]) * np.nan
    #column_vecs = dict()
    for r_ind in range(rhi_data.shape[1]):
        vert_above = rhi_data[:,r_ind][beam_alt[:,r_ind] >= freezing_height]
        l_vat = list(vert_above >= val_threshold)
        if True not in l_vat:
            #print "DEBUG: r_ind: %s no column here" % (r_ind)
            continue
        elif l_vat.count(True) == 1:
            #print "DEBUG: r_ind: %s only a single height is above threshold. Ignoring!" % (r_ind)
            continue
        else:
            #print "DEBUG: r_ind: %s column here!" % (r_ind)
            pass
        lower_ind = l_vat.index(True)
        if False in l_vat[lower_ind:]:
            upper_ind = l_vat.index(False, lower_ind)
        else:
            upper_ind = len(l_vat)
        # 
        column_vec = vert_above[lower_ind:upper_ind]
        _h_above_fl = beam_alt[:,r_ind][beam_alt[:,r_ind] >= freezing_height]
        #col_height = _h_above_fl[-1 if upper_ind == len(l_vat) else upper_ind] - _h_above_fl[lower_ind]
        col_height = _h_above_fl[upper_ind -1] - _h_above_fl[lower_ind] # upper_ind is not a part of the column already
        col_mean_val = np.mean(column_vec)
        #
        column_heights[r_ind] = col_height
        column_mean_vals[r_ind] = col_mean_val
        #column_vecs[r_ind] = column_vec
    return column_heights, column_mean_vals

def identify_CDR_columns(data, metadata, freezing_height, station_height, dB_threshold=-15.0):
    ascending_elev, scan_elev_map = radar_tools.calc_elev_order(metadata)
    scan_name = radar_tools.get_lowest_el_scan(metadata)
    rhi_data, rhi_meta = radar_tools.calc_rhi_data(data, metadata, 0.0, 'CDR', ascending_elev=ascending_elev, scan_elev_map=scan_elev_map)
    beam_alt = wradlib.georef.bin_altitude(rhi_meta['r'], np.asarray(rhi_meta['th'])[:, np.newaxis], station_height, re=6370040.)
    cc_height_map = np.zeros_like(data[scan_name]['CDR']['data']) * np.NaN
    cc_meanval_map = np.zeros_like(cc_height_map) * np.NaN
    azis = metadata[scan_name]['az']
    for az_ind, azi in enumerate(azis):
        rhi_data, rhi_meta = radar_tools.calc_rhi_data(data, metadata, azi, 'CDR', ascending_elev=ascending_elev, scan_elev_map=scan_elev_map)
        cc_height_map[az_ind], cc_meanval_map[az_ind] = detect_columns_along_ray(-rhi_data, beam_alt, freezing_height, val_threshold=-dB_threshold)
    cc_meanval_map *= -1
    return cc_height_map, cc_meanval_map
# ---
# --- Advanced Hail Precursor (currently only ZDR) based on Synder et al. (2015)
# added UpdraftSkewness to consider advection
# added dBZ mask to avoid artifact columns
class AdvColumnIdentifier(object):
    def __init__(self, data, metadata, var, freezing_height, station_height, dB_threshold=1.0, _dBZ_masking_threshold=None):
        self.data = data
        self.metadata = metadata
        #self.desired_var = var
        self.freezing_height = freezing_height
        self.station_height = station_height
        self.val_threshold = dB_threshold
        self._disable_skewed_trace = False
        self._dBZ_masking_threshold = _dBZ_masking_threshold # dBZ
        self._disable_dBZ_masking = True if _dBZ_masking_threshold is None else False
        #
        self.ascending_elev, self.scan_elev_map = radar_tools.calc_elev_order(self.metadata)
        self.lowest_scan_name = radar_tools.get_lowest_el_scan(metadata)
        #
        self.change_var(var) # forces an update of calculated data
    def change_var(self, var):
        self.desired_var = var
        self._calc_all_slices()
        self._results = Namespace.Namespace(available=False, __updateable__=True)
    def _calc_all_slices(self):
        # calculate rhi_data for all azimuth angles
        _, self._rhi_meta = radar_tools.calc_rhi_data(self.data, self.metadata, 0.0, self.desired_var, ascending_elev=self.ascending_elev, scan_elev_map=self.scan_elev_map)
        self._beam_alt = wradlib.georef.bin_altitude(self._rhi_meta['r'], np.asarray(self._rhi_meta['th'])[:, np.newaxis], self.station_height, re=6370040.)
        self._all_rhi_data = dict()
        for var in [self.desired_var, u'DBZH']:
            if var == u'DBZH' and self._disable_dBZ_masking:
                continue
            self._all_rhi_data[var] = dict()
            for azi in self.metadata[self.lowest_scan_name]['az']:
                #cur_rhi_data, cur_rhi_meta = radar_tools.calc_rhi_data(self.data, self.metadata, azi, self.desired_var, ascending_elev=self.ascending_elev, scan_elev_map=self.scan_elev_map)
                cur_rhi_data, cur_rhi_meta = radar_tools.calc_rhi_data(self.data, self.metadata, azi, var, ascending_elev=self.ascending_elev, scan_elev_map=self.scan_elev_map)
                assert self._rhi_meta == cur_rhi_meta
                self._all_rhi_data[var][azi] = cur_rhi_data
        #
        self._square_dtype = cur_rhi_data.dtype
        self.mask_valid_height = (self._beam_alt >= self.freezing_height)
    def find_columns(self):
        if self.desired_var == 'ZDR':
            _identify_fu = np.nanargmax
        elif self.desired_var.endswith('KDP'):
            _identify_fu = np.nanargmax
        else:
            raise ValueError("ERROR: var %s not yet supported!" % (self.desired_var))
        #
        self._results.traces = []
        #self._results.column_height_map = np.zeros(dwd_radar_tools.get_data_shape_from_meta(self.metadata, self.lowest_scan_name)) * np.NaN
        self._results.column_height_map = np.zeros_like(self.data[self.lowest_scan_name][self.desired_var]['data']) * np.NaN
        self._results.column_mean_val_map = np.zeros_like(self._results.column_height_map) * np.NaN
        self._results.column_max_val_map = np.zeros_like(self._results.column_height_map) * np.NaN
        for az_ind, azi in enumerate(self.metadata[self.lowest_scan_name]['az']):
            for r_ind, r in enumerate(self._rhi_meta['r']):
                # 1. check if a possible column is attached to freezing_height
                h_inds = np.where(self.mask_valid_height[:, r_ind])[0]
                if len(h_inds) == 0: continue # no layer above freezing layer?
                #if not self._all_rhi_data[self.desired_var][azi][h_inds[0], r_ind] >= self.val_threshold: continue # check whether column is attached to freezing height
                if self._check_abort_criteria(az_ind, h_inds[0], r_ind): continue # check whether column is attached to freezing height
                # 2. climb up the column
                col_trace = self._trace_column(h_inds, az_ind, r_ind, _identify_fu)
                if len(col_trace) <= 1: continue
                self._results.traces.append(col_trace)#print col_trace, h_inds
                # 3. calculate results from trace
                self._results.column_height_map[az_ind, r_ind] = self._get_col_height_from_trace(col_trace)
                cur_col_vals = self._get_vals_vec_from_trace(col_trace)
                self._results.column_mean_val_map[az_ind, r_ind] = np.nanmean(cur_col_vals)
                self._results.column_max_val_map[az_ind, r_ind] = np.nanmax(cur_col_vals)
        self._results.available = True
    def _check_abort_criteria(self, az_ind, h_ind, r_ind):
        """Returns True when abort is adequate and False if not"""
        azi = self.metadata[self.lowest_scan_name]['az'][az_ind]
        return not (self._all_rhi_data[self.desired_var][azi][h_ind, r_ind] >= self.val_threshold) # this way, NaN will cause abortion as well
    def _trace_column(self, h_inds, az_ind, r_ind, id_fu):
        trace = [(h_inds[0], az_ind, r_ind)]
        cur_az_ind = az_ind; cur_r_ind = r_ind
        for i in range(1,len(h_inds)):
            if not self._disable_skewed_trace: # check for new az and r indices if skewed columns are allowed
                square, sq_inds = self._construct_square(h_inds[i], cur_az_ind, cur_r_ind)
                if np.isnan(square).all(): break
                cur_az_ind, cur_r_ind = np.asarray(sq_inds).reshape((-1,2))[id_fu(square)]
            # check if selected pixel fulfilles criterias
            if self._check_abort_criteria(cur_az_ind, h_inds[i], cur_r_ind): break
            # all fine, so add to trace
            trace.append((h_inds[i],cur_az_ind, cur_r_ind))
        return trace
    def _construct_square(self, h_ind, az_ind, r_ind):
        square = np.zeros((3,3),dtype=self._square_dtype) * np.NaN
        prev_az_ind = az_ind - 1
        next_az_ind = (az_ind + 1) % len(self.metadata[self.lowest_scan_name]['az'])
        prev_r_ind = r_ind - 1 if r_ind > 1 else 0
        next_r_ind = r_ind + 1 #if r_ind < len(self.metadata[self.lowest_scan_name]['r']) else r_ind
        square_inds = np.asarray([
            [(prev_az_ind, prev_r_ind), (prev_az_ind, r_ind), (prev_az_ind, next_r_ind),],
            [(az_ind,      prev_r_ind), (az_ind,      r_ind), (az_ind,      next_r_ind),],
            [(next_az_ind, prev_r_ind), (next_az_ind, r_ind), (next_az_ind, next_r_ind),],
        ])
        _max_r_i = len(self.metadata[self.lowest_scan_name]['r']) - 1
        for i in range(square_inds.shape[0]):
            for j, (azi_i, r_i) in enumerate(square_inds[i]):
                if r_i > _max_r_i: continue # empty entries in square are NaN already
                if (not self._disable_dBZ_masking) and (not self._all_rhi_data[u'DBZH'][self.metadata[self.lowest_scan_name]['az'][azi_i]][h_ind, r_i] >= self._dBZ_masking_threshold): continue # check for dBZ masking
                square[i,j] = self._all_rhi_data[self.desired_var][self.metadata[self.lowest_scan_name]['az'][azi_i]][h_ind, r_i]
        # mask values that are below freezing height
        if not self.mask_valid_height[h_ind, prev_r_ind]:
            square[:,0] = np.NaN
        #
        return square, square_inds
    def _get_col_height_from_trace(self, trace):
        heights = []
        for h_ind, az_ind, r_ind in trace:
            heights.append(self._beam_alt[h_ind, r_ind])
        return np.sum(np.diff(heights))
    def _get_vals_vec_from_trace(self, trace):
        vals = []
        for h_ind, az_ind, r_ind in trace:
            azi = self.metadata[self.lowest_scan_name]['az'][az_ind]
            vals.append(self._all_rhi_data[self.desired_var][azi][h_ind, r_ind])
        return vals
    def _check_results_ready(self):
        if not self._results.available:
            raise ValueError("ERROR: No results available, yet! Run .find_columns() first!")
    def get_traces(self):
        self._check_results_ready()
        return self._results.traces
    def get_col_height_map(self):
        self._check_results_ready()
        return self._results.column_height_map
    def get_col_maxval_map(self):
        self._check_results_ready()
        return self._results.column_max_val_map
    def get_col_meanval_map(self):
        self._check_results_ready()
        return self._results.column_mean_val_map
# ---
