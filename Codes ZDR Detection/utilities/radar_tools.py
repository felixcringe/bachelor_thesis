#! /usr/bin/env python

import numpy as np
import datetime

def z2dBZ(z):
    """Calculates logarithmic dBZ value of linear z value (mm^3/mm^6)"""
    return 10*np.log10(z)
def dBZ2z(dBZ):
    """Calculates linear z value (mm^3/mm^6) of logarithmic dBZ value."""
    return 10 ** (dBZ/10.)

def phidp_unfolding(phidp, phi_sys=120, phi_opt=-150):
    """
    Unfolds Phi_DP according to "Processing of differential phase" (internal paper, Oct. 30, 2008)
    
    Arguments:
        phidp:          array, contains Phi_DP for each range bin along a certain azimuth.
    
    Parameters:
        phi_sys:        float, Phi_DP Parameter of the system (or first valid Phi_DP value in signal without noise). Default: 120
        phi_opt:        float, optimal Phi_DP to be desired. Default: -150
    
    Returns:
        unfolded_phidp: array, unfolded Phi_DO for each range bin along a certain azimuth.
    """
    #if phidp > 90:
    #    phidp -= 270
    #else:
    #    phidp += 90
    #unfolded_pdp = np.empty_like(phidp)
    #unfolded_pdp = np.where(phidp > 90, phidp - 270, phidp + 90)
    unfolded_pdp = np.where(phidp > phi_sys - phi_opt - 180, phidp - (phi_sys - phi_opt), phidp - (phi_sys - phi_opt) + 360)
    return unfolded_pdp


def cum_step_sum(vec):
    """Constructs a cumulative step-wise sum vector. A given vector 'vec' is summed at each step, when it's entry is greater than zero. Otherwise the sum is started over."""
    y = []
    c = 0
    for x in vec:
        if x > 0: # <- this treats np.NaN as False, too.
            c += 1
        else:
            c = 0
        y.append(c)
    return y

def find_cont_range(rho_hv, bin_threshold=20, val_threshold=0.95):
    """
    Finds the first set of bins along one radial which exceeds the 'val_threshold' with at least 'bin_threshold' number of bins.
    
    Arguments:
        rho_hv:         array, one ray of rho_hv values.
    Parameters:
        bin_threshold:  integer, number of bins, which have to exceed the 'val_threshold'.
        val_threshold:  integer, value of rho_hv, which has to be exceeded.
    Returns:
        first_begin:    integer, the index of the range-bin, where at least 'bin_threshold' bins exceed the 'val_threshold' the first time.
        first_end:      integer, the index of the range-bin, where the contiguous range bins at or above the 'val_threshold' end.
    """
    val_exceeded = rho_hv >= val_threshold
    num_bins = cum_step_sum(val_exceeded)
    if (np.array(num_bins) >= bin_threshold).any():
        first_bin_thr_exc = list(num_bins).index(bin_threshold) #num_bins.index(bin_threshold)
        if 0 in num_bins[first_bin_thr_exc:]:
            first_end = list(num_bins).index(0, first_bin_thr_exc) - 1
        else: # last bin is still in the interval
            first_end = len(num_bins) - 1
        first_begin = first_bin_thr_exc - bin_threshold + 1
        return first_begin, first_end
    else:
        return None


# stat tools
def running_mean(x, N):
    """calculates a running mean of vector 'x' using 'N' bins"""
    #cumsum = np.cumsum(np.insert(x, 0, 0))
    cumsum = np.cumsum(np.insert(np.where(~np.isnan(x),x,0), 0, 0))  # replace NaNs with zeros with the calculation
    return (cumsum[int(N):] - cumsum[:-int(N)]) / int(N) 

def dynamic_ma(x, N):
    """
    Moving average, which decrease window size at border.
    
    Arguments:
        x:      array-like, input data to be smoothed.
        N:      int, size of half-window to be used to smooth.
    Returns:
        ma_res: array-like, smoothed x.
    """
    N = int(N)
    if N > 1:
        ma_res = np.zeros_like(x)
        for n in range(1,int(N)):
            ma_res[n-1] = _single_ma_val(x,n,at=n-1)
            ma_res[-n] = _single_ma_val(x,n,at=len(x)-n)
        ma_res[(N-1)//2:-(N-1)//2] = running_mean(x,N)
    else:
        ma_res = running_mean(x,N)
    return ma_res

def _single_ma_val(x, N, at):
    """Calculates the difference between the sum-vectors of vector 'x' at positions 'at' and 'N+at', normed by the number of bins 'N'."""
    N = int(N)
    #return (np.sum(x[:N+at]) - np.sum(x[:at])) / N
    return (np.nansum(x[:N+at]) - np.nansum(x[:at])) / N
# --

# geo tools
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees). Result is in kilometers.
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def bearing(lat1, lon1, lat2, lon2):
    """
    Calculate the horizontal bearing between two points
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    bearing = np.arctan2(np.sin(lon2-lon1) * np.cos(lat2),
                         np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2-lon1)
                        )
    bearing = np.degrees(bearing)
    bearing = (bearing + 360) % 360
    return bearing

def calc_terminal_point(lat1, lon1, bearing, distance):
    """
    Calculate the lat and lon of point given bearing, distance and origin.
    Based on: http://stackoverflow.com/questions/7222382/get-lat-long-given-current-point-distance-and-bearing
    """
    R = 6378.1 #Radius of the Earth in km
    brng = np.deg2rad(bearing) #Bearing in degrees converted to radians.
    d = distance/1000. #Distance in km
    lat1 = np.radians(lat1) #Current lat point converted to radians
    lon1 = np.radians(lon1) #Current long point converted to radians
    lat2 = np.arcsin( np.sin(lat1)*np.cos(d/R) +
         np.cos(lat1)*np.sin(d/R)*np.cos(brng))
    lon2 = lon1 + np.arctan2(np.sin(brng)*np.sin(d/R)*np.cos(lat1),
                 np.cos(d/R)-np.sin(lat1)*np.sin(lat2))
    lat2 = np.degrees(lat2)
    lon2 = np.degrees(lon2)
    return lat2, lon2
#

# dwd data tools
def calc_elev_order(metadata):
    """Calculates the order of elevations in the given metadata. A list of the elevations (their degrees) and a mapping of scanname->elevation is returned."""
    scan_elev_map = {}
    for k,v in metadata.items():
        if not (k.startswith('SCAN') or k.startswith('DATASET')):
            continue
        scan_elev_map[k] = v['el']
    ascending_elev = sorted(scan_elev_map.values())
#     ascending_elev.sort()
    return ascending_elev, scan_elev_map

def get_lowest_el_scan(metadata):
    """returns the name of the scan with the lowest elevation in the given metadata."""
    ascending_elev, scan_elev_map = calc_elev_order(metadata)
    i = list(scan_elev_map.values()).index(ascending_elev[0])
    return list(scan_elev_map.keys())[i]

def calc_rhi_data(data, metadata, azi, var, ascending_elev=None, scan_elev_map=None):
    """
    Calculates pseudo-RHIs from volumetric data.
    
    Arguments:
        data:           dict, containing the radar-variables (scanname : variable : 'data' : array).
        metadata:       dict, containing the meta-data of the radar sweep.
        azi:            float, azimuthal angle to calculate the RHI for.
        var:            str, the name of the variable to calculate the RHI for.
    Parameters:
        ascending_elev: list of floats, the order of the elevations in ascending order. Can be provided if already calculated. Default: None
        scan_elev_map:  dict, mapping of scanname->elevation. Can be provided if already calculated. Default: None
    Returns:
        rhi_data:       array, (elevation index, range index) contains the data in RHI form.
        rhi_meta:       dict, contains the range-bins and the elevation angle for this pseudo-RHI.
    """
    if ascending_elev is None or scan_elev_map is None:
        ascending_elev, scan_elev_map = calc_elev_order(metadata)
    num_elev = len(ascending_elev)
    _key_bin_count = 'bin_count' if 'bin_count' in metadata[list(scan_elev_map.keys())[0]] else u'nbins' # backward and forward compatibility for older and newer dwd radar data
    num_rbins = int(max(v[_key_bin_count] for k,v in metadata.items() if k.startswith('SCAN')))
    rhi_data = np.zeros((num_elev, num_rbins)) * np.NaN
    # find closest azi_index to desired azi
    azi_ind = (np.abs(metadata[list(scan_elev_map.keys())[0]]['az']-azi)).argmin() # it is assumed, that for all scans, azimuths are equally distant
    #
    for k,v in data.items():
        elev = scan_elev_map[k]
        elev_ind = list(ascending_elev).index(elev)
        if var == 'UPHIDP':
            corr_data = phidp_unfolding(v[var]['data'][azi_ind, :]) #+ 150
        else:
            corr_data = v[var]['data'][azi_ind, :]
        rhi_data[elev_ind, :metadata[k][_key_bin_count]] = corr_data
    # calc rhi_meta
    lowest_scanname = list(scan_elev_map.keys())[list(scan_elev_map.values()).index(ascending_elev[0])]
    rhi_meta = {
        'r' : metadata[lowest_scanname]['r'],
        'th' : ascending_elev,
    }
    #
    return rhi_data, rhi_meta

def meta2dt_obj(metadata, scan_name):
    """Extracts the date and time from DWD radar metadata and returns a datetime object."""
    return datetime.datetime.strptime(metadata[scan_name]['Time'], '%Y-%m-%dT%H:%M:%S.%fZ')
#

def est_phi_sys(phi_dp, rho_hv, bin_threshold=20, val_threshold=0.95, wx_radials_threshold=30):
    """
    Estimate phi_sys for one scan/sweep.
    
    Arguments:
        phi_dp:                  array, one sweep of phi_dp values in shape (number of azimuths, number of range bins).
        rho_hv:                  array, one sweep of rho_hv values in shape (number of azimuths, number of range bins).
    Parameters:
        bin_threshold:           integer, number of bins, which have to exceed the 'val_threshold'. Default: 20
        val_threshold:           integer, value of rho_hv, which has to be exceeded. Default: 0.95
        wx_radials_threshold:    integer, number of rays, which have to have an interval to allow an estimation of phi_sys. Default: 30
    Returns:
        phi_sys:                 float/None, value of estimated phi_sys or None if not enough radials had an interval.
    """
    fdp = []
    for azi in range(rho_hv.shape[0]):
        # find range intervals with acceptable rho_hv values
        interval = find_cont_range(dynamic_ma(rho_hv[azi], bin_threshold/2), bin_threshold=bin_threshold, val_threshold=val_threshold)
        if interval is None:
            fdp.append(np.NaN)
        else:
            # store median PhiDP of this range interval
            fdp.append(np.median(phi_dp[azi][interval[0]:interval[1]+1]))
    if np.count_nonzero(np.logical_not(np.isnan(fdp))) < wx_radials_threshold:
        return None # no valid intervals found at all
    else:
        nn_fdp = np.array(fdp)[np.logical_not(np.isnan(fdp))]
        sorted_nn_fdp = sorted(nn_fdp)[:wx_radials_threshold]
        return np.median(sorted_nn_fdp)
#

# filters
def spike_filter_med(data, ws=5):
    """Filters all spikes with broadness below half of the window-size"""
    # old version
    #filtered_data = []
    #for i in xrange(0,ws/2):
    #    filtered_data.append(np.median(data[i:i+ws]))
    #    filtered_data.append(np.median(data[i:i+ws]))
    #for i in xrange(ws/2,len(data)-ws):
    #    filtered_data.append(np.median(data[i:i+ws]))
    #for i in xrange(len(data)-ws,len(data),2):
    #    filtered_data.append(np.median(data[i:i+ws]))
    #return filtered_data
    
    # speed optimized array version
    filtered_data = np.median(
        np.concatenate([np.roll(data[:,np.newaxis],-i) for i in range(ws)],axis=1),
        axis=1,
    )
    # correct for wrapping-around-effects at the end of vector
    for i in range(len(data)-ws+1,len(data)):
        filtered_data[i]=np.median(data[i:i+ws])
    return filtered_data
#



