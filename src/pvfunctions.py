#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/02/21 10:27:53

@author: Javiera Jilberto Vallejos 
'''
import os

import numpy as np
import matplotlib.pyplot as plt

import cheartio as chio
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.optimize import minimize, Bounds
from shapely.geometry import Polygon

filepath = os.path.dirname(os.path.abspath(__file__))

def check_traces(time, trace):
    """
    Ensures that the trace is a closed loop by checking if the first and last elements are the same.
    If they are not, it appends the first element to the end of the trace and adjusts the time array accordingly.

    Parameters:
    time (numpy.ndarray): An array representing the time points corresponding to the trace values.
    trace (numpy.ndarray): An array representing the trace values.

    Returns:
    tuple: A tuple containing the adjusted time and trace arrays.
    """
    if trace[0] != trace[-1]:
        trace = np.concatenate((trace, trace[0:1]))
        time = np.linspace(0, 1, len(trace))

    return time, trace


def check_volume_traces(vol_trace, shift_to_zero=True):
    vol_trace_og = vol_trace.copy()
    ext_vol_trace_og = np.concatenate((vol_trace_og[:-1], vol_trace_og, vol_trace_og[1:])) 
    st_idx = len(vol_trace) - 1
    ext_vol_trace = np.concatenate((vol_trace[:-1], vol_trace, vol_trace[1:])) 

    # Find minimum and maximum values
    min_vol = np.min(ext_vol_trace)
    max_vol = np.max(ext_vol_trace)

    min_idx = np.where(ext_vol_trace == min_vol)[0]
    max_idx = np.where(ext_vol_trace == max_vol)[0]
    
    # Grab the middle index for min and the closest two indices for max
    min_idx = min_idx[1]
    max_idxs = np.sort(max_idx[np.argsort(np.abs(max_idx - min_idx))[:2]])
    shift = max_idxs[0] - st_idx

    # Decreasing trace
    dec_trace = ext_vol_trace[max_idxs[0]:min_idx+1]
    
    # Find indices where dec_trace is not monotonically decreasing
    non_decreasing_idx = np.where(np.diff(dec_trace) > 0)[0] + 1

    while len(non_decreasing_idx) > 0:
        # Interpolate only over the valid (monotonically decreasing) points
        valid_idx = np.setdiff1d(np.arange(len(dec_trace)), non_decreasing_idx)
        interp_func = interp1d(valid_idx, dec_trace[valid_idx], kind='linear', fill_value='extrapolate')
        dec_trace[non_decreasing_idx] = interp_func(non_decreasing_idx)

        # Recalculate non_decreasing_idx
        non_decreasing_idx = np.where(np.diff(dec_trace) > 0)[0]

    # Increasing trace
    inc_trace = ext_vol_trace[min_idx:max_idxs[1]+1]

    # Find indices where inc_trace is not monotonically increasing
    non_increasing_idx = np.where(np.diff(inc_trace) < 0)[0] + 1

    while len(non_increasing_idx) > 0:
        # Interpolate only over the valid (monotonically increasing) points
        valid_idx = np.setdiff1d(np.arange(len(inc_trace)), non_increasing_idx)
        interp_func = interp1d(valid_idx, inc_trace[valid_idx], kind='linear', fill_value='extrapolate')
        inc_trace[non_increasing_idx] = interp_func(non_increasing_idx)

        # Recalculate non_increasing_idx
        non_increasing_idx = np.where(np.diff(inc_trace) < 0)[0] + 1

    # Combine the traces
    vol_trace = np.concatenate((dec_trace, inc_trace[1:]))

    # # Shift back
    if shift_to_zero:
        ext_vol_trace = np.concatenate((vol_trace[:-1], vol_trace, vol_trace[1:]))
        vol_trace = ext_vol_trace[st_idx-shift:(st_idx+len(vol_trace)-shift)]

    return vol_trace, shift


    


def load_volume_traces(lv_vol_path, rv_vol_path=None, use_rv=False):
    time_vol, lv_vol = chio.read_dfile(lv_vol_path).T
    Tcycle_vol = time_vol[-1]  # seconds 
    time_vol, lv_vol = check_traces(time_vol, lv_vol)       # This normalizes the time

    if use_rv:
        _, rv_vol = chio.read_dfile(rv_vol_path).T

        # Note we assume that the time array is the same for both the left and right ventricles
        time_vol, rv_vol = check_traces(time_vol, rv_vol)

        return time_vol, lv_vol, rv_vol, Tcycle_vol

    return time_vol, lv_vol, Tcycle_vol


def load_pressure_traces(lv_pres_path=None, rv_pres_path=None, use_ref_pres_trace=True, use_rv=False):
    if lv_pres_path is None and not use_ref_pres_trace:
        raise ValueError("Either 'lv_pres_path' must be provided or 'use_ref_pres_trace' must be set to True")
    
    if lv_pres_path is not None and use_ref_pres_trace:
        use_ref_pres_trace = False
        print("Warning: Both 'lv_pres_path' and 'use_ref_pres_trace' are set. Setting 'use_ref_pres_trace' to False.")

    if use_ref_pres_trace:
        time_pres, lv_pres = np.load(f'{filepath}/reference_data/russell_pressure_trace/normalized_human_pressure.npy').T
        Tcycle_pres = 1.0
        normalized_valve_times = np.load(f'{filepath}/reference_data/russell_pressure_trace/normalized_valve_times.npz')

        if use_rv:
            rv_pres = np.copy(lv_pres)
            return time_pres, lv_pres, rv_pres, Tcycle_pres, normalized_valve_times
        
        return time_pres, lv_pres, Tcycle_pres, normalized_valve_times
    
    else:
        time_pres, lv_pres = chio.read_dfile(lv_pres_path).T
        Tcycle_pres = time_pres[-1] # seconds
        time_pres, lv_pres = check_traces(time_pres, lv_pres)
        if use_rv:
            _, rv_pres = chio.read_dfile(rv_pres_path).T
            time_pres, rv_pres = check_traces(time_pres, rv_pres)
            return time_pres, lv_pres, rv_pres, Tcycle_pres
        
        return time_pres, lv_pres, Tcycle_pres

def load_reference_pressure_trace():
    time_pres, lv_pres = np.load(f'{filepath}/reference_data/russell_pressure_trace/normalized_human_pressure.npy').T
    normalized_valve_times = np.load(f'{filepath}/reference_data/russell_pressure_trace/normalized_valve_times.npz')
    
    return time_pres, lv_pres, normalized_valve_times


def load_atrial_pressure_trace(atrial_pres_path=None, use_ref_pres_atrial_trace=True):
    if atrial_pres_path is None and not use_ref_pres_atrial_trace:
        raise ValueError("Either 'atrial_pres_path' must be provided or 'use_ref_pres_atrial_trace' must be set to True")
    
    if atrial_pres_path is not None and use_ref_pres_atrial_trace:
        use_ref_pres_atrial_trace = False
        print("Warning: Both 'atrial_pres_path' and 'use_ref_pres_atrial_trace' are set. Setting 'use_ref_pres_atrial_trace' to False.")

    if use_ref_pres_atrial_trace:
        time_atrial_pres, atrial_pres = np.load(f'{filepath}/reference_pressure_trace/normalized_atrial_pressure.npy').T
        Tcycle_pres = 1.0
        normalized_atrial_valve_times = np.load(f'{filepath}/reference_pressure_trace/atrial_valve_times.npz')
        return time_atrial_pres, atrial_pres, Tcycle_pres, normalized_atrial_valve_times
    else:
        time_atrial_pres, atrial_pres = chio.read_dfile(atrial_pres_path).T
        Tcycle_pres = time_atrial_pres[-1]
        time_atrial_pres, atrial_pres = check_traces(time_atrial_pres, atrial_pres)
        return time_atrial_pres, atrial_pres, Tcycle_pres

            
def load_aorta_pressure_trace(aorta_pres_path=None, use_ref_pres_aorta_trace=True):
    if aorta_pres_path is None and not use_ref_pres_aorta_trace:
        raise ValueError("Either 'aorta_pres_path' must be provided or 'use_ref_pres_aorta_trace' must be set to True")
    
    if aorta_pres_path is not None and use_ref_pres_aorta_trace:
        use_ref_pres_aorta_trace = False
        print("Warning: Both 'aorta_pres_path' and 'use_ref_pres_aorta_trace' are set. Setting 'use_ref_pres_aorta_trace' to False.")

    if use_ref_pres_aorta_trace:
        time_aorta_pres, aorta_pres = np.load(f'{filepath}/reference_pressure_trace/normalized_aorta_pressure.npy').T
        Tcycle_pres = 1.0
        normalized_aorta_valve_times = np.load(f'{filepath}/reference_pressure_trace/aorta_valve_times.npz')
        return time_aorta_pres, aorta_pres, Tcycle_pres, normalized_aorta_valve_times
    else:
        time_aorta_pres, aorta_pres = chio.read_dfile(aorta_pres_path).T
        Tcycle_pres = time_aorta_pres[-1]
        time_aorta_pres, aorta_pres = check_traces(time_aorta_pres, aorta_pres)
        return time_aorta_pres, aorta_pres, Tcycle_pres


def add_iv_datapoints(time, trace, iv_dur_norm, max_or_min):
    if max_or_min == 'max':
        vol_idx = np.argmax(trace)
    else:
        vol_idx = np.argmin(trace)

    time_vol = time.copy()
    trace_vol = trace.copy()
    time_vol_point = time_vol[vol_idx]

    iv_time_1 = time_vol_point - iv_dur_norm / 2
    iv_time_2 = time_vol_point + iv_dur_norm / 2
    
    time_vol = np.insert(time_vol, vol_idx, iv_time_1)
    time_vol = np.insert(time_vol, vol_idx + 2, iv_time_2)
    trace_vol = np.insert(trace_vol, vol_idx, trace_vol[vol_idx])
    trace_vol = np.insert(trace_vol, vol_idx + 2, trace_vol[vol_idx])
    

    return time_vol, trace_vol, (iv_time_1, iv_time_2)


def add_isovolumic_relaxation(time, trace, valve_times, Tcycle, ivr_time=0.07):
    time, trace, iv_times = add_iv_datapoints(time, trace, ivr_time/Tcycle, 'min')
    if 'avc' in valve_times:    # LV
        valve_times['avc'] = iv_times[0]
        valve_times['mvo'] = iv_times[1]
    else:   # RV
        valve_times['pvc'] = iv_times[0]
        valve_times['tvo'] = iv_times[1]
    return time, trace, valve_times


def add_isovolumic_contraction(time, trace, valve_times, Tcycle, ivc_time=0.05):
    time, trace, iv_times = add_iv_datapoints(time, trace, ivc_time/Tcycle, 'max')
    if 'mvc' in valve_times:    # LV
        valve_times['mvc'] = iv_times[0]
        valve_times['avo'] = iv_times[1]
        vol_shift = -valve_times['mvc']
    else:   # RV
        valve_times['tvc'] = iv_times[0]
        valve_times['pvo'] = iv_times[1]
        vol_shift = -valve_times['tvc']

    # Shift valve times
    for key in valve_times:
        if key != 'tcycle':
            valve_times[key] += vol_shift
            if valve_times[key] < 0:
                valve_times[key] += 1.0

    return time, trace, valve_times


def rescale_pressure_magnitude(pressure, ed_pressure, es_pressure):
    """
    Rescales the normalized pressure magnitude based on end-diastolic (ED) and end-systolic (ES) pressures.

    Parameters:
    pressure (array-like): The original pressure values to be rescaled.
    ed_pressure (float): The end-diastolic pressure value.
    es_pressure (float): The end-systolic pressure value.

    Returns:
    array-like: The rescaled pressure values.
    """
    if np.max(pressure) != 1.0:
        raise ValueError('Pressure must be normalized')

    pressure = pressure.copy()
    sp = es_pressure
    dp = ed_pressure

    aux = pressure.copy()
    a = (sp - dp)/(1.0 - aux[0])
    pressure[aux > aux[0]] = aux[aux > aux[0]]*a + sp - a
    pressure[aux <= aux[0]] = aux[aux <= aux[0]]/aux[0]*dp
    return pressure


def rescale_pressure_time(normalized_time, normalized_pressure, valve_times, normalized_valve_times):
    if 'avc' in valve_times:    # LV
        events = ['mvc', 'avo', 'avc', 'mvo', 'tcycle']
    elif 'pvc' in valve_times:
        events = ['tvc', 'pvo', 'pvc', 'tvo', 'tcycle']
    else:
        raise ValueError('Failed to find valve events')
        
    chamber_pressure = []
    pressure_time = []

    normalized_func = interp1d(normalized_time, normalized_pressure)
    for i in range(1, len(events)):
        x = np.linspace(normalized_valve_times[events[i-1]],normalized_valve_times[events[i]],1001, endpoint=True)
        segment_pressure = normalized_func(x)
        segment_time = np.linspace(valve_times[events[i-1]],valve_times[events[i]],1001, endpoint=True)

        chamber_pressure.append(segment_pressure[1:])
        pressure_time.append(segment_time[1:])

    chamber_pressure = np.hstack(chamber_pressure)
    pressure_time = np.hstack(pressure_time)
    chamber_pressure = np.append(normalized_pressure[0], chamber_pressure)
    pressure_time = np.append(0, pressure_time)

    return chamber_pressure, pressure_time


def rescale_normalized_pressure_trace(norm_pres_time, norm_pres, valve_times, normalized_valve_times, edp, sysp):
    # Need to rescale
    pres_mag = rescale_pressure_magnitude(norm_pres, edp, sysp)
    pres_time, time_pres = rescale_pressure_time(norm_pres_time, pres_mag, valve_times, normalized_valve_times)

    return pres_time, time_pres



def repeat_traces(trace):
    return np.concatenate((trace, trace[1:], trace[1:]))


def repeat_time(time):
    """
    Repeats the given time array three times by concatenating it with shifted versions of itself.

    Parameters:
    time (numpy.ndarray): A 1D array of time values.

    Returns:
    numpy.ndarray: A concatenated array consisting of the original time array shifted by its final value.
    """
    Tf = time[-1]
    return np.concatenate((time-Tf, time[1:], time[1:] + Tf))


def get_pv_functions(time, trace, interp='pchip'):
    assert len(time) == len(trace), "Time and trace must have the same length"

    trace_ext = repeat_traces(trace)
    time_ext = repeat_time(time)

    # Smooth traces
    if interp == 'linear':
        trace_func = interp1d(time_ext, trace_ext, fill_value='extrapolate')
    elif interp == 'pchip':
        trace_func = PchipInterpolator(time_ext, trace_ext)
    else:
        raise ValueError(f"Invalid interpolation method: {interp}")
    
    return trace_func

        
def shift_vol_func(vol_func, shift):
    time = np.linspace(-1, 2, 1501)
    vol = vol_func(time - shift)
    vol_func = interp1d(time, vol)
    return vol_func


def shift_to_ed(pres_func, vol_func, edp, edv):
    time = np.linspace(-1, 2, 1501)
    vol = vol_func(time)
    pres = pres_func(time)
    
    vol = vol - vol_func(0) + edv
    pres = pres - pres_func(0) + edp

    vol_func = interp1d(time, vol)
    pres_func = interp1d(time, pres)

    # Sanity check
    assert vol_func(0) == edv
    assert pres_func(0) == edp

    return pres_func, vol_func


def clip_pv_loop_by_passive(pres_func, vol_func, vol_passive, pres_passive, valve_opening):
#%%
    time = np.linspace(valve_opening, 1.0, 201)

    pres_pv = pres_func(time)
    vol_pv = vol_func(time)

    # Remove points outside the passive volume range
    pres_vol_pv = interp1d(vol_pv, pres_pv, kind='linear', fill_value='extrapolate')
    pres_vol_pass = interp1d(vol_passive, pres_passive, kind='linear', fill_value='extrapolate')

    # Find indexes where the pass pressure  starts being greater than the PV pressure
    pres_pv = pres_vol_pv(vol_pv)
    pres_pass = pres_vol_pass(vol_pv)
    where = np.where(pres_pass > pres_pv)[0]
    if len(where) == 0:
        # If no points are found, return the original pressure function
        return pres_func
    
    idx = where[0]

    # For all the indices after the idx, we set the pressure to the pass pressure
    pres_pv[idx:] = pres_pass[idx:]

    # Reconstruct the pressure-volume function
    pres_func_pass = interp1d(time, pres_pv, kind='linear', fill_value='extrapolate')

    # Reconstruct the pressure function
    time_before = np.linspace(0, valve_opening, 501)
    pres_before = pres_func(time_before)
    time_after = np.linspace(valve_opening, 1.0, 501)
    pres_after = pres_func_pass(time_after)

    pres_func_clipped = interp1d(np.concatenate((time_before, time_after)),
                                 np.concatenate((pres_before, pres_after)),
                                 kind='linear', fill_value='extrapolate')
    
    return pres_func_clipped
#%%

def scale_atrial_pressure_magnitude(time_atrial_pres, atrial_pres, atrial_valve_times, ven_pres_func, ven_valve_times):
    # Left side

    if 'mvo' in ven_valve_times:
        vo = atrial_valve_times['mvo']
        vc = atrial_valve_times['mvc']
        ven_pres_vc = ven_pres_func(ven_valve_times['mvc'])
        ven_pres_vo = ven_pres_func(ven_valve_times['mvo'])
    else:
        vo = atrial_valve_times['tvo']
        vc = atrial_valve_times['tvc']
        ven_pres_vc = ven_pres_func(ven_valve_times['tvc'])
        ven_pres_vo = ven_pres_func(ven_valve_times['tvo'])

    atrial_pres_func = get_pv_functions(time_atrial_pres, atrial_pres, interp='linear')
    
    scaling_values = [ven_pres_vc/atrial_pres_func(vc), ven_pres_vo/atrial_pres_func(vo), ven_pres_vc/atrial_pres_func(vc)]
    scaling_times = [vc, vo, 1.0]

    scaling_func = interp1d(scaling_times, scaling_values, kind='linear')

    time = np.linspace(0, 1, 1001)
    scaling = scaling_func(time)
    atrial_pres = atrial_pres_func(time) * scaling

    return time, atrial_pres


def scale_atrial_pressure_time(normalized_time, normalized_pressure, normalized_valve_times, ven_valve_times):

    if 'avc' in ven_valve_times:    # LV
        events = ['mvc', 'mvo', 'tcycle']
    elif 'pvc' in ven_valve_times:
        events = ['tvc', 'tvo', 'tcycle']
    else:
        raise ValueError('Failed to find valve events')
        
    chamber_pressure = []
    pressure_time = []

    normalized_func = interp1d(normalized_time, normalized_pressure)
    for i in range(1, len(events)):
        x = np.linspace(normalized_valve_times[events[i-1]],normalized_valve_times[events[i]],1001, endpoint=True)
        segment_pressure = normalized_func(x)
        segment_time = np.linspace(ven_valve_times[events[i-1]],ven_valve_times[events[i]],1001, endpoint=True)

        chamber_pressure.append(segment_pressure[1:])
        pressure_time.append(segment_time[1:])

    chamber_pressure = np.hstack(chamber_pressure)
    pressure_time = np.hstack(pressure_time)
    chamber_pressure = np.append(normalized_pressure[0], chamber_pressure)
    pressure_time = np.append(0, pressure_time)

    return pressure_time, chamber_pressure


def fix_diastolic_atrial_pressure(time_atrial_pres, atrial_pres, ven_pres_func, ven_valve_times):   
     # During diastole the atrial pressure should be higher than the ventricular pressure
    time = np.linspace(0, 1, 1001)

    pres_func = interp1d(time_atrial_pres, atrial_pres, kind='linear')
    atrial_pres = pres_func(time)
    ven_pres = ven_pres_func(time)

    if 'avc' in ven_valve_times:    # LV
        vo = 'mvo'
    elif 'pvc' in ven_valve_times:
        vo = 'tvo'
    else:
        raise ValueError('Failed to find valve events')

    vo_idx = (ven_valve_times[vo]/time[1]).astype(int)
    while not np.all((atrial_pres[vo_idx:]/ven_pres[vo_idx:]) > 1):  # Need to fix
        idx = np.argmax(ven_pres[vo_idx:]/atrial_pres[vo_idx:])
        idx_val = (ven_pres[vo_idx:]/atrial_pres[vo_idx:])[idx]*1.01

        scaling_values = [1.0, 1.0, idx_val, 1.0]
        scaling_times = [0.0, ven_valve_times[vo], time[vo_idx:][idx], 1.0]
        scaling_func = interp1d(scaling_times, scaling_values, kind='linear')
        atrial_pres = atrial_pres * scaling_func(time)

    return time, atrial_pres
    

def scale_aorta_pressure_magnitude(time_aorta_pres, aorta_pres, aorta_valve_times, ven_pres_func, ven_valve_times):
    # Left side

    if 'avo' in ven_valve_times:
        vo = aorta_valve_times['avo']
        vc = aorta_valve_times['avc']
        ven_pres_vc = ven_pres_func(ven_valve_times['avc'])
        ven_pres_vo = ven_pres_func(ven_valve_times['avo'])
    else:
        vo = aorta_valve_times['pvo']
        vc = aorta_valve_times['pvc']
        ven_pres_vc = ven_pres_func(ven_valve_times['pvc'])
        ven_pres_vo = ven_pres_func(ven_valve_times['pvo'])

    aorta_pres_func = interp1d(time_aorta_pres-vo, aorta_pres, kind='linear', fill_value='extrapolate')
    
    scaling_values = [ven_pres_vo/aorta_pres_func(0.0), ven_pres_vc/aorta_pres_func(vc-vo), ven_pres_vo/aorta_pres_func(0.0)]
    scaling_times = [0.0, vc-vo, 1.0]

    scaling_func = interp1d(scaling_times, scaling_values, kind='linear', fill_value='extrapolate')

    time = np.linspace(0, 1, 1001)
    scaling = scaling_func(time)
    aorta_pres = aorta_pres_func(time) * scaling

    # Shift back to original time
    aorta_pres_ext = repeat_traces(aorta_pres)
    time_ext = repeat_time(time)
    aorta_pres_func = interp1d(time_ext, aorta_pres_ext)
    aorta_pres = aorta_pres_func(time-vo)

    return time, aorta_pres


def scale_aorta_pressure_time(normalized_time, normalized_pressure, normalized_valve_times, ven_valve_times):

    if 'avc' in ven_valve_times:    # LV
        events = ['avo', 'avc', 'tcycle']
    elif 'pvc' in ven_valve_times:
        events = ['pvo', 'pvc', 'tcycle']
    else:
        raise ValueError('Failed to find valve events')
        
    chamber_pressure = []
    pressure_time = []

    normalized_func = interp1d(normalized_time, normalized_pressure)
    for i in range(1, len(events)):
        x = np.linspace(normalized_valve_times[events[i-1]],normalized_valve_times[events[i]],1001, endpoint=True)
        segment_pressure = normalized_func(x)
        segment_time = np.linspace(ven_valve_times[events[i-1]],ven_valve_times[events[i]],1001, endpoint=True)

        chamber_pressure.append(segment_pressure[1:])
        pressure_time.append(segment_time[1:])

    chamber_pressure = np.hstack(chamber_pressure)
    pressure_time = np.hstack(pressure_time)
    chamber_pressure = np.append(normalized_pressure[0], chamber_pressure)
    pressure_time = np.append(0, pressure_time)

    return pressure_time, chamber_pressure


def fix_systolic_aorta_pressure(time_aorta_pres, aorta_pres, ven_pres_func, ven_valve_times):   
    # During systole the aorta pressure should be lower than the ventricular pressure
    time = np.linspace(0, 1, 1001)

    pres_func = interp1d(time_aorta_pres, aorta_pres, kind='linear')
    aorta_pres = pres_func(time)
    ven_pres = ven_pres_func(time)

    if 'avc' in ven_valve_times:    # LV
        vo = 'avo'
        vc = 'avc'
    elif 'pvc' in ven_valve_times:
        vo = 'pvo'
        vc = 'pvc'
    else:
        raise ValueError('Failed to find valve events')

    vo_idx = (ven_valve_times[vo]/time[1]).astype(int)
    vc_idx = (ven_valve_times[vc]/time[1]).astype(int)

    phi = np.ones(len(time))
    idx = np.argmax(ven_pres[vo_idx:vc_idx]/aorta_pres[vo_idx:vc_idx])
    scaling_values = [1.0, 1.0, 0.98, 1.0, 1.0]
    scaling_times = [0.0, ven_valve_times[vo], time[vo_idx:vc_idx][idx], ven_valve_times[vc], 1.0]
    scaling_func = interp1d(scaling_times, scaling_values, kind='linear')
    phi[vo_idx:vc_idx] = ven_pres[vo_idx:vc_idx]/aorta_pres[vo_idx:vc_idx]
    phi = phi * scaling_func(time)
    aorta_pres = aorta_pres * phi
    # while not np.all((aorta_pres[vo_idx:vc_idx]/ven_pres[vo_idx:vc_idx]) < 1):  # Need to fix
    #     idx = np.argmax(aorta_pres[vo_idx:vc_idx]/ven_pres[vo_idx:vc_idx])
    #     idx_val = (ven_pres[vo_idx:vc_idx]/aorta_pres[vo_idx:vc_idx])[idx]*0.99

    #     scaling_values = [1.0, 1.0, idx_val, 1.0, 1.0]
    #     scaling_times = [0.0, ven_valve_times[vo], time[vo_idx:vc_idx][idx], ven_valve_times[vc], 1.0]
    #     scaling_func = interp1d(scaling_times, scaling_values, kind='linear')
    #     aorta_pres = aorta_pres * scaling_func(time)

    return time, aorta_pres


def get_pv_loop(vol_time, vol_trace, valve_times, edp, sysp):
    norm_time, ref_pres, normalized_valve_times = load_reference_pressure_trace()
    pressure, time_pressure = rescale_normalized_pressure_trace(norm_time, ref_pres, valve_times, normalized_valve_times, edp, sysp)
    
    vol_func = interp1d(vol_time/np.max(vol_time), vol_trace, fill_value='extrapolate')
    vol_trace = vol_func(norm_time)

    pres_func = interp1d(time_pressure/np.max(time_pressure), pressure, fill_value='extrapolate')
    pres_trace = pres_func(norm_time)

    return norm_time, vol_trace, pres_trace


def get_shift_pv_loop(vol_trace, vol_time_shift, valve_times, valve_times_shift, edp, sysp):
    if 'avo' in valve_times_shift:    # LV
        avo_time = valve_times['avo'] + valve_times_shift['avo']
        avc_time = valve_times['avc'] + valve_times_shift['avc']
        mvo_time = valve_times['mvo'] + valve_times_shift['mvo']
        valve_times = {'mvc': 0., 'avo': avo_time, 'avc': avc_time, 'mvo': mvo_time, 'tcycle': valve_times['tcycle']}
    elif 'pvo' in valve_times_shift:  # RV
        pvo_time = valve_times['pvo'] + valve_times_shift['pvo']
        pvc_time = valve_times['pvc'] + valve_times_shift['pvc']
        tvo_time = valve_times['tvo'] + valve_times_shift['tvo']
        valve_times = {'tvc': 0., 'pvo': pvo_time, 'pvc': pvc_time, 'tvo': tvo_time, 'tcycle': valve_times['tcycle']}

    time = np.linspace(0, 1, len(vol_trace))

    norm_time, vol, pres = get_pv_loop(time, vol_trace, valve_times, edp, sysp)  

    ext_vol = repeat_traces(vol)
    ext_time = repeat_time(norm_time)
    
    vol_func = interp1d(ext_time, ext_vol, fill_value='extrapolate')
    vol = vol_func(time + vol_time_shift)

    return vol, pres, valve_times


def compute_pv_area(x_coords, y_coords):
    polygon = Polygon(zip(x_coords, y_coords))
    area = polygon.area
    return area

    
def optimize_pv_area(vol_trace, valve_times, dt, edp, sysp, side='lv', dt_shift=1):
    bounds = Bounds([-1.5*dt_shift*dt, -dt_shift*dt, -dt_shift*dt, -dt_shift*dt], [1.5*dt_shift*dt, dt_shift*dt, dt_shift*dt, dt_shift*dt])

    def func(x):
        vol_time_shift = x[0]
        if side == 'lv':
            valve_times_shift = {'avo': x[1], 'avc': x[2], 'mvo': x[3]}
            lv_vol, lv_pres, valve_times_shifted = get_shift_pv_loop(vol_trace, vol_time_shift, valve_times, valve_times_shift, edp, sysp)
            return -compute_pv_area(lv_vol, lv_pres)
        elif side == 'rv':
            valve_times_shift = {'pvo': x[1], 'pvc': x[2], 'tvo': x[3]}
            rv_vol, rv_pres, valve_times_shifted = get_shift_pv_loop(vol_trace, vol_time_shift, valve_times, valve_times_shift, edp, sysp)
            return -compute_pv_area(rv_vol, rv_pres)

    x0 = np.array([0., 0., 0., 0.])
    sol = minimize(func, x0, method='trust-constr', bounds=bounds)

    # Generate the final volume and pressure traces after optimization
    vol_time_shift = sol.x[0]
    if side == 'lv':
        valve_times_shift = {'avo': sol.x[1], 'avc': sol.x[2], 'mvo': sol.x[3]}
        lv_vol, lv_pres, valve_times_shifted = get_shift_pv_loop(vol_trace, vol_time_shift, valve_times, valve_times_shift, edp, sysp)
        
        valve_times['avo'] = valve_times_shifted['avo']
        valve_times['avc'] = valve_times_shifted['avc']
        valve_times['mvo'] = valve_times_shifted['mvo']

        return lv_vol, lv_pres, valve_times
    
    elif side == 'rv':
        valve_times_shift = {'pvo': sol.x[1], 'pvc': sol.x[2], 'tvo': sol.x[3]}
        rv_vol, rv_pres, valve_times_shifted = get_shift_pv_loop(vol_trace, vol_time_shift, valve_times, valve_times_shift, edp, sysp)
        
        valve_times['pvo'] = valve_times_shifted['pvo']
        valve_times['pvc'] = valve_times_shifted['pvc']
        valve_times['tvo'] = valve_times_shifted['tvo']
        
        return rv_vol, rv_pres, valve_times


def sbp_to_cbp(sbp, gender='F'):
    """
    Converts systolic blood pressure (SBP) to central blood pressure (CBP).
    These regressions are based on the data from https://doi.org/10.1161/HYPERTENSIONAHA.107.105445.
    This paper also has data: 10.1097/HJH.0000000000003743 but for teens
    """
    
    if gender == 'F':
        slope = 0.9628757408659583
        intercept = -1.444588621236619
    elif gender == 'M':
        slope = 0.9610305254497609
        intercept = -4.051420714056746
    cbp = slope * sbp + intercept
    return cbp

"""
PLOT FUNCTIONS
"""

def plot_pv_loop_traces(vol_func, pres_func, valve_times, 
                        vol_func_lin=None,
                        vol_passive=None, pres_passive=None, 
                        atrial_pres_func=None, 
                        aorta_pres_func=None, Tc=1.0):
    
    time_pv = np.linspace(0, Tc, 1000)
    pres_pv = pres_func(time_pv)
    vol_pv = vol_func(time_pv)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,8))

    # Pressure vs Volume subplot
    ax1.plot(vol_pv / 1000, pres_pv * 7.50062, 'k-', label='Pressure-Volume Loop')
    if vol_passive is not None and pres_passive is not None:
        ax1.plot(vol_passive / 1000, pres_passive * 7.50062, 'k--', label='Passive Loop')
    # Add valve events as dots
    for key, val in valve_times.items():
        if key != 'tcycle':
            ax1.plot(vol_func(val) / 1000, pres_func(val) * 7.50062, 'ro', label=key)
    ax1.set_xlabel('Volume (ml)')
    ax1.set_ylabel('Pressure (mmHg)')
    ax1.set_title('Pressure vs Volume')
    ax1.grid(True)

    # Pressure subplot
    ax2.plot(time_pv, pres_pv * 7.50062, 'k-', label='Pressure')

    if atrial_pres_func is not None:
        atrial_pres_pv = atrial_pres_func(time_pv)
        ax2.plot(time_pv, atrial_pres_pv * 7.50062, 'k-', alpha=0.5, label='Atrial Pressure')
    if aorta_pres_func is not None:
        aorta_pres_pv = aorta_pres_func(time_pv)
        ax2.plot(time_pv, aorta_pres_pv * 7.50062, 'g-', label='Aorta Pressure')

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Pressure (mmHg)')
    ax2.set_title('Pressure over Time')
    ax2.grid(True)

    # Volume subplot
    ax4.plot(time_pv, vol_pv / 1000, 'k-', label='Volume')
    if vol_func_lin is not None:
        ax4.plot(time_pv, vol_func_lin(time_pv) / 1000, 'k--', label='Volume (Linear)')

    # Add valve event dots
    for ax in (ax2, ax4):
        for key, val in valve_times.items():
            if key != 'tcycle':
                ax.plot(val, vol_func(val) / 1000 if ax == ax4 else pres_func(val) * 7.50062, 'ro')

    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Volume (ml)')
    ax4.set_title('Volume over Time')
    ax4.grid(True)

    # Remove the unused subplot and use it for the legend
    fig.delaxes(ax3)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.1, 0.1, 0.5, 0.5), frameon=False)

    plt.tight_layout()

    axs = ((ax1, ax2), (ax3, ax4))
    return axs


def plot_reference_trace():
    norm_time, ref_pres, normalized_valve_times = load_reference_pressure_trace()

    plt.figure()
    plt.plot(norm_time, ref_pres * 7.50062, label='Reference Pressure Trace')
    for key, t in normalized_valve_times.items():
        plt.plot(t, interp1d(norm_time, ref_pres)(t) * 7.50062, 'o', label=key)
    plt.legend()

    return norm_time, ref_pres, normalized_valve_times