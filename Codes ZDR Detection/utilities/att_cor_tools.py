#! /usr/bin/env python

# import radar_tools
from utilities import radar_tools
import numpy as np

class ZPHI(object):
    """
    This class provides all necessary functions to apply the ZPHI method for attenuation correction in rain.
    """
    def __init__(self):
        self._res_A_h = None
        self._apply_phi_spike_filt = True
    def reset_results(self):
        """In case this object is used again, the results should be reset before starting over."""
        self._res_A_h = None
    def get_attenuation(self):
        """Return the calculated specific attenuation. The objects function .calc_attenuation must have been executed first."""
        assert self._res_A_h is not None # must execute .calc_attenuation first
        return self._res_A_h
    def calc_attenuation(self, z_obs, _PhiDP, r_0, r_m, alpha_0, b=0.8, ds=0.25):
        """
        Calculates specific attenuation using linear reflectivity factor 'z_obs' and differential phase '_PhiDP' in the range interval ('r_0', 'r_m') using attenuation coefficient for rain 'alpha_0'.
        Note: The distance between two range bins 'ds' is expected to be in kilometers!
        """
        assert r_0 < r_m
        if self._apply_phi_spike_filt:
            PhiDP = np.zeros_like(_PhiDP) + np.asarray(_PhiDP)
            _r_inds = range(r_0,r_m+1)
            PhiDP[_r_inds] = np.asarray(radar_tools.spike_filter_med(_PhiDP))[_r_inds]
        else:
            PhiDP = _PhiDP
        #
        r_inds = np.arange(r_0, r_m) # make a vector of all range indices
        # the following calculations follow e.g. Ryzhkov (2013b)
        C = self._eq_C(alpha_0, PhiDP[r_m] - PhiDP[r_0])
        I_fixed = self._eq_int(b, r_0, r_m, z_obs, ds)
        I_var = np.asarray([self._eq_int(b, r_ind, r_m, z_obs, ds) for r_ind in r_inds])
        A_h = np.zeros_like(z_obs) #* np.NaN
        A_h[r_0:r_m] = self._eq_A_h(z_obs, r_inds, C, b, I_fixed, I_var)
        self._res_A_h = A_h
    def _eq_A_h(self, z_obs, r_ind, C, b, I_fixed, I_var):
        #return (z_obs[r_ind] ** b * (10. ** (0.1*b*C) - 1.)) / (I_fixed + (10. ** (0.1*b*C) - 1.)*I_var)
        _lower_term = I_fixed + (10. ** (0.1*b*C) - 1.)*I_var
        _upper_term = z_obs[r_ind] ** b * (10. ** (0.1*b*C) - 1.)
        return np.where(_lower_term == 0, np.NaN, _upper_term/_lower_term) # will always be a vector, since r_inds and therefore I_var is a vector
    def _eq_C(self, alpha_0, delta_Phi_total):
        return alpha_0 * delta_Phi_total
    def _eq_int(self, b, r_ind, r_m, z_obs, ds):
        return 0.46 * b * np.sum(z_obs[r_ind:r_m]**b) * ds
    def correct_attenuation(self, Z_H, r_0, r_m, A_h=None, ds=0.25, _cor_beyond_rm=True):
        """
        Returns a attenuation-corrected reflectivity factor.
        
        Arguments:
            Z_H:            array, one ray of measured hor. reflectivity factor in dBZ.
            r_0:            int, range bin to start the attenuation correction at.
            r_m:            int, range bin defining the end of range-interval to apply the ZPHI-method to.
        Parameters:
            A_h:            array, specific attenuation to correct reflectivity factor. If None, will be taken from the object. Default: None
            ds:             float, distance between two range bins in kilometers. Default: 0.25
            _cor_beyond_rm: bool, correct attenuation beyond r_m using A_h at r_m. Default: True.
        Returns:
            Z_cor:          array, attenuated-corrected hor. reflectivity factor.
        
        Note: The objects function .calc_attenuation must have been executed first if 'A_h' is not provided as parameter.
        """
        if A_h is None:
            A_h = self.get_attenuation()
        Z_cor = np.zeros_like(Z_H) * np.NaN
        #_int_A = 2. * np.cumsum(A_h[r_0:r_m]) * ds
        #Z_cor[r_0:r_m] = Z_H[r_0:r_m] + _int_A
        _int_A = 2. * np.cumsum(A_h[:r_m]) * ds
        Z_cor[:r_m] = Z_H[:r_m] + _int_A
        if _cor_beyond_rm:
            Z_cor[r_m:] = Z_H[r_m:] + _int_A[-1]
        return Z_cor
class HotspotZPHI(ZPHI):
    """
    Hotspot/ZPHI-method as described by Ryzhkov et al. (2007) and Gu et al. (2011).
    Modifications added:
        e.g. spike-filter for PhiDP for range gates outside hailcore.
    """
    def __init__(self):
        super(HotspotZPHI, self).__init__()
        self._res_delta_alpha = None
        self._dbg_had2use_extrapolation = None
        self._dbg_total_signal_extinction_flag = False
        self._apply_phi_spike_filt_os = True
        self._min_radials_past_hc = 4
        self._allow_extrapolation_on_signal_extinction = False
    def reset_results(self):
        """In case this object is used again, the results should be reset before starting over."""
        super(HotspotZPHI, self).reset_results()
        self._res_delta_alpha = None
        self._dbg_had2use_extrapolation = None
        self._dbg_total_signal_extinction_flag = False
    def _eq_C(self, alpha_0, delta_Phi_total, delta_alpha, delta_Phi_hs):
        return alpha_0 * delta_Phi_total + delta_alpha * delta_Phi_hs
    def _get_r_inds_outside_hs(self, r_0, r_1, r_2, r_m):
        """returns the indices of the range bins outside the hotspot."""
        return np.asarray(list(set(range(r_0, r_m)) - set(range(r_1, r_2))))
    def _get_phi_outside_hs(self, PhiDP, r_0, r_1, r_2, r_m):
        """Calculates the increase in phase outside the hotspot (by substracting the increase inside the hotspot from the total increase)."""
        return (PhiDP[r_m] - PhiDP[r_0]) - (PhiDP[r_2] - PhiDP[r_1])
    def _calculate_specific_attenuation(self, delta_alpha, r_0, r_1, r_2, r_m, PhiDP, z_obs, ds, b=0.8, alpha_0=0.06, r_inds=None):
        """Calculates the specific attenauation respecting the increased attenuation coefficient 'delta_alpha' in the hotspot (r_1, r_2)."""
        if r_inds is None: r_inds = np.arange(r_0, r_m)
        C = self._eq_C(alpha_0, PhiDP[r_m] - PhiDP[r_0], delta_alpha, PhiDP[r_2] - PhiDP[r_1])
        I_fixed = self._eq_int(b, r_0, r_m, z_obs, ds)
        I_var = np.asarray([self._eq_int(b, r_ind, r_m, z_obs, ds) for r_ind in r_inds])
        A_h = np.zeros_like(z_obs) #* np.NaN
        A_h[r_0:r_m] = self._eq_A_h(z_obs, r_inds, C, b, I_fixed, I_var)
        return A_h
    def _check_criteria4total_signal_extinction(self, Z_H, r_2, r_m):
        """Returns True if signal is likely to be totally extinct after r_m or False if not."""
        assert r_m >= r_2
        _bhs_offset = r_m - r_2 # range bins past hotspots
        if r_m + _bhs_offset + 1 >= len(Z_H): return False # reached edge of measurement area. Total signal extinction is not certain.
        if _bhs_offset > 1: return False # still able to measure a few range bins after hotspot. Certainly total signal extinction isn't the case here
        if Z_H[r_m+_bhs_offset+1] >= 20: return False # Assuming that a hor. reflectivity of 20dBZ or more is unlikely to occur in total signal extinction. However, rho_hv could still be messed up
        return True
    def find_delta_alpha(self, _z_obs, _PhiDP, r_0, r_1, r_2, r_m, alpha_0=0.06, b=0.8, ds=0.25, _quit_when_err_incr=False):
        """
        Determines an optimal delta_alpha by iterating through possible values (following Gu et al. (2011).
        
        Arguments:
            _z_obs:                 array, linear reflectivity factor for each range bin.
            _PhiDP:                 array, differential phase for each range bin.
            r_0:                    int, index determining the first range bin in the valid precipitation area.
            r_1:                    int, index determining the beginning of the hotspot inside the precipitation area.
            r_2:                    int, index determining the last range bin of the hotspot.
            r_m:                    int, index determining the last range bin in the valid precipitation area.
        Parameters:
            alpha_0:                float, background attenuation coefficient (alpha in rain). Default: 0.06
            b:                      float, exponential coefficient in empirical equation A=a*Z^b (see Table 1 in Ryzhkov et al., 2013b). Default: 0.8
            ds:                     float, distance between two range bins in kilometers. Default: 0.25
        Returns:
            self._res_delta_alpha:  float, optimal delta alpha.
        """
        assert r_0 <= r_1 < r_2 <= r_m
        if (r_1 - r_0 + (r_m - r_2)) < self._min_radials_past_hc:
            # first check if total signal extinction might be possible
            self._dbg_total_signal_extinction_flag = self._check_criteria4total_signal_extinction(radar_tools.z2dBZ(_z_obs), r_2, r_m)
            # ... then extrapolate if allowed
            if self._allow_extrapolation_on_signal_extinction and self._dbg_total_signal_extinction_flag:
                self._dbg_had2use_extrapolation = True
                _num_missing_radials = self._min_radials_past_hc - (r_1 - r_0 + (r_m - r_2))
                 # copy data, to prevent data modifications
                z_obs = np.zeros_like(_z_obs) + np.asarray(_z_obs)
                PhiDP = np.zeros_like(_PhiDP) + np.asarray(_PhiDP)
                # extrapolate values
                _z_gradient = np.mean(np.diff(_z_obs[r_m+1-4:r_m+1]))
                _z_gradient = min(_z_gradient, np.diff((z_obs[r_m], radar_tools.dBZ2z(radar_tools.z2dBZ(z_obs[r_m])-10*ds)))) # have atleast a gradient of -10dB/km
                z_obs[r_m+1:r_m+1+_num_missing_radials] = np.asarray([_z_obs[r_m]+_z_gradient*i for i in range(1,_num_missing_radials+1)])
                PhiDP[r_m+1:r_m+1+_num_missing_radials] = PhiDP[r_m]
                # modify r_m
                r_m = r_m + _num_missing_radials
            else:
                raise NotEnoughRadialsOutsideHailcore("ERROR: At least %s radials outside of hailcore are required! (r1=%s, r1=%s, r2=%s, rm=%s)" % (self._min_radials_past_hc, r_0, r_1, r_2, r_m), r_0, r_1, r_2, r_m, self._min_radials_past_hc)
        else:
            z_obs = _z_obs
            self._dbg_had2use_extrapolation = False
        if self._apply_phi_spike_filt_os: # outside of hailcore PhiDP spikes are more liekly and problematic
            PhiDP = np.zeros_like(_PhiDP) + np.asarray(_PhiDP) # copy data, to prevent data modifications
            _ohs_r_inds = list(range(r_0,r_1)) + list(range(r_2,r_m+1)) # +1 because PhiDP[r_m] will be used and range would stop at r_m-1
            PhiDP[_ohs_r_inds] = np.asarray(radar_tools.spike_filter_med(_PhiDP))[_ohs_r_inds]
            #print _PhiDP[r_m] - _PhiDP[r_0],PhiDP[r_m] - PhiDP[r_0]
            #self._dbg_ohs_sf_phi = PhiDP
        else:
            PhiDP = _PhiDP
        # solve right side of equation, because it won't change
        _rs = alpha_0/2. * self._get_phi_outside_hs(PhiDP, r_0, r_1, r_2, r_m)
        #
        r_inds = np.arange(r_0, r_m)
        r_inds_outside = self._get_r_inds_outside_hs(r_0, r_1, r_2, r_m)
        _last_diff = None
        _best_delta_alpha = None
        # iterate of increasing delta_alpha
        for delta_alpha in np.arange(-0.02, 1., 0.01):
            #C = self._eq_C(alpha_0, PhiDP[r_m] - PhiDP[r_0], delta_alpha, PhiDP[r_2] - PhiDP[r_1])
            #I_fixed = self._eq_int(b, r_0, r_m, z_obs, ds)
            #I_var = np.asarray([self._eq_int(b, r_ind, r_m, z_obs, ds) for r_ind in r_inds])
            #A_h = np.zeros_like(z_obs) #* np.NaN
            #A_h[r_0:r_m] = self._eq_A_h(z_obs, r_inds, C, b, I_fixed, I_var)
            A_h = self._calculate_specific_attenuation(delta_alpha, r_0, r_1, r_2, r_m, PhiDP, z_obs, ds, b=b, alpha_0=alpha_0)
            # calculate left side and check if condition satisfied
            _ls = np.sum(A_h[r_inds_outside]) * ds
            if _last_diff is None or abs(_ls - _rs) < _last_diff:
                #print _last_diff, abs(_ls - _rs), delta_alpha
                _last_diff = abs(_ls - _rs)
                self._res_A_h = A_h
                _best_delta_alpha = delta_alpha
            else:
                # we could abort here, since error increases again
                if _quit_when_err_incr:
                    break
        self._res_delta_alpha = _best_delta_alpha
        return self._res_delta_alpha
    def get_attenuation(self):
        """Return the calculated specific attenuation. The objects function .find_delta_alpha must have been executed first."""
        assert self._res_A_h is not None # must execute .find_delta_alpha first
        return self._res_A_h
#

def get_va_containing_hs_inds(valid_areas, hotspots, verbose=False):
    """
    Marks valid_areas, which contain hotspots.
    
    Arguments:
        valid_areas:    list-like object, containing list-like objects of indices of radials, which are considered to contain weather echoes.
        hotspots:       list-like object, containg list-like objects of indices of radials, which are considered to be hotspots.
    Parameters:
        verbose:        bool, if True print some warnings. Default: False
    Returns:
        i_va_with_hs:   list, length is equal to valid_areas. Each entry is either a None, if no hotspot is present in this area or a list of hotspot-indices, which are present in that area.
    """
    inds_valid_areas_with_hs = [None] * len(valid_areas)
    for i,hotspot in enumerate(hotspots):
        r_1 = hotspot[0]
        r_2 = hotspot[-1]
        for j,area in enumerate(valid_areas):
            if r_1 in area and r_2 in area:
                if inds_valid_areas_with_hs[j] is None:
                    inds_valid_areas_with_hs[j] = []
                inds_valid_areas_with_hs[j].append(i)
                break
        else:
            if verbose: print ( "WARNING: hotspot (%s, %s) not in valid areas!" % (r_1, r_2) )
            pass # hotspots which are out of valid areas will not be computed
    return inds_valid_areas_with_hs

def repair_gapped_areas(areas, Z_H, Phi_DP, rho_hv, _threshold4repair_gap_max_range=2, _threshold4repair_gap_min_rho=0.5, _threshold_use_intp_Phi_max_diff=10., _ret_missing_inds=[], _verbose=False, _dbg_az_ind=None):
    """Checks for intervals which are close to each other and tries to concatenate them, if reasonable.
        Also, interpolates broken Z_H and Phi_DP where necessary."""
    # assuming length of areas is greater than 1. If not, loop will be skipped and values returned normally
    for i, (intv1, intv2) in enumerate(zip(areas[:-1:], areas[1::])):
        if intv2[0] - intv1[-1] <= _threshold4repair_gap_max_range:
            _missing_inds = np.asarray(range(intv1[-1]+1,intv2[0]))
            # check if repair is possible / usefull
            if not (rho_hv[_missing_inds] >= _threshold4repair_gap_min_rho).all():
                if _verbose: print ("DEBUG: missing range bins (%s) have too low rho_hv to allow repairing. (az_ind: %s)" % (_missing_inds, _dbg_az_ind) )
                continue
            _ret_missing_inds.append(_missing_inds)
            # check if PhiDP is fine or needs to be linearly interpolated
            _intp_Phi = np.interp(_missing_inds, list(intv1)+list(intv2), list(Phi_DP[np.asarray(intv1)])+list(Phi_DP[np.asarray(intv2)]))
            if not (np.mean(abs(_intp_Phi - Phi_DP[_missing_inds])**2.)**0.5 <= _threshold_use_intp_Phi_max_diff): # hitting threshold or containg NaN will result in using interpolated Phi_DP
                if _verbose: print ("DEBUG: Diff in Phi reached threshold and is replaced by interpolated Phi! (Phi_DP: %s, intp(Phi): %s, az_ind: %s, _missing_inds: %s)" % (Phi_DP[_missing_inds], _intp_Phi, _dbg_az_ind, _missing_inds) )
                Phi_DP[_missing_inds] = _intp_Phi
            # interpolate missing Z
            if not (Z_H[_missing_inds] >= 10).all():
                Z_H[_missing_inds] = np.interp(_missing_inds, list(intv1)+list(intv2), list(Z_H[np.asarray(intv1)])+list(Z_H[np.asarray(intv2)]))
            # concatenate areas to new object
            new_areas = []
            for j in range(len(areas)):
                if j == i:
                    new_areas.append(list(intv1)+list(_missing_inds)+list(intv2))
                elif j != i and j != i+1:
                    new_areas.append(areas[j])
            # after repairing once, break and restart. Some areas might have been split into multiple parts
            return repair_gapped_areas(new_areas, Z_H, Phi_DP, rho_hv, _threshold4repair_gap_max_range=_threshold4repair_gap_max_range, _threshold4repair_gap_min_rho=_threshold4repair_gap_min_rho, _threshold_use_intp_Phi_max_diff=_threshold_use_intp_Phi_max_diff, _ret_missing_inds=_ret_missing_inds, _verbose=_verbose, _dbg_az_ind=_dbg_az_ind)
    else: # all done, return normally
        return areas, Z_H, Phi_DP

def surpress_noisy_phi(Phi_DP, deltaPhi_threshold=33., _phi_opt=0., _do_interp=True, _max_iter=2, _verbose=False):
    """Eliminates spikes in unfolded Phi_DP by canceling them and replace by interpolated values."""
    # Phi_DP is assumed to be unfolded
    cancel_mask = np.concatenate(([True,], ~(np.abs(np.diff(Phi_DP)) <= deltaPhi_threshold))) # first value is unknown, but we need shape matching
    # TODO: this cancelation mask might always cancel a valid gate as well (as the slope is determined between both gates)
    assert cancel_mask.shape == Phi_DP.shape
    nc_Phi = Phi_DP.copy()
    nc_Phi[cancel_mask] = np.NaN # this is necessary to make the median filter work
    # when whole ray is noisy, then simply return the optimal Phi value...
    if np.isnan(nc_Phi).all():
        nc_Phi[:] = _phi_opt
        if _verbose: print ( "WARN: (surpress_noisy_phi) whole ray is noisy and set to single value" )
        return nc_Phi
    #
    for i in range(_max_iter): # iterate until all huge differences are removed (they would lead to steap slopes during interpolation)
        detected_spikes = ~(np.abs(np.diff(nc_Phi[~np.isnan(nc_Phi)])) <= deltaPhi_threshold*2)
        if (detected_spikes == False).all():
            if _verbose: print ("DBG: (surpress_noisy_phi) break after iteration", i )
            break
        cancel_mask[~np.isnan(nc_Phi)] = np.concatenate(([True,], detected_spikes)) # double the threshold, because we might have bigger gaps (however, increasing the threshold by the number of missing gates, this wouldn't be any different than interpolating)
        nc_Phi[cancel_mask] = np.NaN
    else:
        if _verbose: print ("DBG: (surpress_noisy_phi) max_iter reached", _max_iter )
    # interpolate values where invalid #~~using median-spike-filtered values~~
    assert np.all(np.diff(np.nonzero(~cancel_mask)[0]) > 0)
    # check again if non-NaN values are left over
    if np.isnan(nc_Phi).all():
        nc_Phi[:] = _phi_opt
        if _verbose: print ("WARN: (surpress_noisy_phi) whole ray is noisy and set to single value (after iterations)" )
        return nc_Phi
    #
    #nc_Phi[cancel_mask] = np.interp(np.nonzero(cancel_mask)[0], np.nonzero(~cancel_mask)[0], np.asarray(radar_tools.spike_filter_med(nc_Phi))[~cancel_mask], left=_phi_opt, right=None)
    nc_Phi[cancel_mask] = np.interp(np.nonzero(cancel_mask)[0], np.nonzero(~cancel_mask)[0], nc_Phi[~cancel_mask], left=_phi_opt, right=None)
    # interpolate where there are still NaNs
    if _do_interp: nc_Phi[np.isnan(nc_Phi)] = np.interp(np.nonzero(np.isnan(nc_Phi))[0], np.nonzero(~np.isnan(nc_Phi))[0], nc_Phi[~np.isnan(nc_Phi)], left=_phi_opt, right=None)
    return nc_Phi

#
def get_consecutive_intervals(r_inds):
    """Finds consecutive range indices and returns them as intervals (two range indices for each interval)."""
    return np.split(r_inds, np.where(np.diff(r_inds) != 1)[0]+1)
def identify_hotspot_ryzhkov2007(Z_cor, rho, Z_threshold=45., rho_threshold=0.8, len_threshold=2000., ds=250.):
    _Z_filt = Z_cor >= Z_threshold
    _rho_filt = rho >= rho_threshold
    r_inds = np.nonzero(np.logical_and(_Z_filt, _rho_filt))[0]
    intv = get_consecutive_intervals(r_inds)
    _inds = np.nonzero(np.asarray(map(len, intv)) >= len_threshold/ds)[0]
    possible_intvals = np.asarray(intv)[_inds]
    return possible_intvals
def identify_hotspot_gu2011(Z_cor, rho, ZDR, PhiDP, Z_threshold=45., rho_threshold=0.7, len_threshold=2000., zdr_threshold=3.0, phi_threshold=10., ds=250., _check_ZDR=True):
    _Z_filt = Z_cor >= Z_threshold
    _rho_filt = rho >= rho_threshold
    r_inds = np.nonzero(np.logical_and(_Z_filt, _rho_filt))[0]
    intv = get_consecutive_intervals(r_inds)
    _inds = np.nonzero(np.asarray(map(len, intv)) >= len_threshold/ds)[0]
    possible_intvals = np.asarray(intv)[_inds]
    #print "DEBUG: num intervals %s, intervals=" % (len(possible_intvals)), possible_intvals
    if len(possible_intvals) == 0:
        return possible_intvals
    # check if interval's maximum ZDR exceeds ZDR minimum-threshold at least once
    if _check_ZDR:
        _check_zdr = lambda inds: (np.nanmax(ZDR[inds]) >= zdr_threshold).any()
        possible_intvals = possible_intvals[np.nonzero(map(_check_zdr, possible_intvals))[0]]
        #print "DEBUG: num intervals %s, intervals=" % (len(possible_intvals)), possible_intvals
    # check if interval's total change in PhiDP exceeds PhiDP minimum-treshold
    _check_phi = lambda inds: (PhiDP[inds[-1]] - PhiDP[inds[0]]) >= phi_threshold
    possible_intvals = possible_intvals[np.nonzero(map(_check_phi, possible_intvals))[0]]
    #print "DEBUG: num intervals %s, intervals=" % (len(possible_intvals)), possible_intvals
    return possible_intvals
def identify_hotspot_schmidt2019(Z_cor, rho, ZDR, PhiDP, Z_threshold=45., rho_threshold=0.7, len_threshold=2000., zdr_threshold=3.0, phi_threshold=10., ds=250., _check_ZDR=True, _use_sf_rho=False, _use_sf_phi=False):
    """
    Based on Ryzhkov et al. (2007) and Gu et al. (2011).
    Modifications:
        * Added option to allow spike-filtered rho_hv. This is usefull in heavy hail-bearing storms as rho_hv drops dramatically in areas with lots of hail. The bitwise OR will be used to allow every bin which would exceed the rho_hv threshold either without spike-filtering or with. This is to avoid discarding range bins, which are fine, but got smeared by the spike-filter.
        * Added option to allow for spike-filtered PhiDP. This is usefull to cancel out noise in PhiDP. The bitwise OR will be used to allow every bin which would exceed the Delta PhiDP threshold either without spike-filtering or with. This is to avoid discarding hotspots, which are fine, but got smeared by the spike-filter.
        * ...
    
    IMPORTANT: ds (range bin step width) is expected to be given in m, not km!!
    """
    _Z_filt = Z_cor >= Z_threshold
    _rho_filt = rho >= rho_threshold
    if _use_sf_rho:
        _rho_sf_filt = np.asarray(radar_tools.spike_filter_med(rho)) >= rho_threshold
        _rho_filt = _rho_filt | _rho_sf_filt
    r_inds = np.nonzero(np.logical_and(_Z_filt, _rho_filt))[0]
    intv = get_consecutive_intervals(r_inds)
    _inds = np.nonzero(np.asarray(list(map(len, intv))) >= len_threshold/ds)[0]
    possible_intvals = np.asarray(intv)[_inds]
    #print "DEBUG: num intervals %s, intervals=" % (len(possible_intvals)), possible_intvals
    if len(possible_intvals) == 0:
        return possible_intvals
    # check if interval's maximum ZDR exceeds ZDR minimum-threshold at least once
    if _check_ZDR:
        _check_zdr = lambda inds: (np.nanmax(ZDR[inds]) >= zdr_threshold).any()
        possible_intvals = possible_intvals[np.nonzero(map(_check_zdr, possible_intvals))[0]]
        #print "DEBUG: num intervals %s, intervals=" % (len(possible_intvals)), possible_intvals
    # check if interval's total change in PhiDP exceeds PhiDP minimum-treshold
    if _use_sf_phi:
        _PhiDP_sf = radar_tools.spike_filter_med(PhiDP)
        _check_phi = lambda inds: (PhiDP[inds[-1]] - PhiDP[inds[0]]) >= phi_threshold or (_PhiDP_sf[inds[-1]] - _PhiDP_sf[inds[0]]) >= phi_threshold
    else:
        _check_phi = lambda inds: (PhiDP[inds[-1]] - PhiDP[inds[0]]) >= phi_threshold
    possible_intvals = possible_intvals[np.nonzero(map(_check_phi, possible_intvals))[0]]
    #print "DEBUG: num intervals %s, intervals=" % (len(possible_intvals)), possible_intvals
    return possible_intvals
def identify_hotspot(Z_cor, rho, ZDR=None, PhiDP=None, Z_threshold=45., rho_threshold=0.7, len_threshold=2000., zdr_threshold=3.0, phi_threshold=10., ds=250., _check_ZDR=True, _use_sf_rho=False, _use_sf_phi=False):
    """
    Identification of hotspots and valid areas. Uses either methods based on ryzhkov et al. (2007) or Gu et al. (2011).
    The methods of the latter paper is used, when ZDR and PhiDP are both given.
    
    IMPORTANT: ds (range bin step width) is expected to be given in m, not km!!
    """
    if ZDR is None or PhiDP is None:
        return identify_hotspot_ryzhkov2007(Z_cor, rho, Z_threshold=Z_threshold, rho_threshold=rho_threshold, len_threshold=len_threshold, ds=ds)
    else:
        return identify_hotspot_schmidt2019(Z_cor, rho, ZDR, PhiDP, Z_threshold=Z_threshold, rho_threshold=rho_threshold, len_threshold=len_threshold, ds=ds, _check_ZDR=_check_ZDR, phi_threshold=phi_threshold, zdr_threshold=zdr_threshold, _use_sf_rho=_use_sf_rho, _use_sf_phi=_use_sf_phi)

# Validation Exceptions
class ValidationError(Exception):
    # based on: http://stackoverflow.com/questions/1319615/proper-way-to-declare-custom-exceptions-in-modern-python
    '''Raise when a specific validation went wrong.'''
    def __init__(self, message, *args):
        self.message = message # without this you may get DeprecationWarning    
        # allow users initialize misc. arguments as any other builtin Error
        super(ValidationError, self).__init__(message, *args)
class HailcoreOutsideAllValidAreas(ValidationError):
    """Raised to signal, that hailcore was outside of all ordinary weather echoes. This may be because a non-weather echo was identified as hailcore. Absence of hail can be assumed."""
    def __init__(self, message, r_1, r_2, *args):
        self.message = message # without this you may get DeprecationWarning    
        self.r_1 = r_1
        self.r_2 = r_2
        super(HailcoreOutsideAllValidAreas, self).__init__(message, *args)
class NotEnoughRadialsOutsideHailcore(ValidationError):
    """Raised to signal, that there werent enough radials outside of hailcore to ensure proper quality in alpha calculation."""
    def __init__(self, message, r_0, r_1, r_2, r_m, radials_threshold, *args):
        self.message = message # without this you may get DeprecationWarning    
        self.r_1 = r_1
        self.r_2 = r_2
        self.r_0 = r_0
        self.r_m = r_m
        super(NotEnoughRadialsOutsideHailcore, self).__init__(message, *args)

#

