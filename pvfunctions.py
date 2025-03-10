#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/02/21 10:27:53

@author: Javiera Jilberto Vallejos 
'''

import numpy as np
import matplotlib.pyplot as plt

import cheartio as chio
from scipy.interpolate import PchipInterpolator, interp1d


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
        time_pres, lv_pres = np.load('reference_pressure_trace/normalized_human_pressure.npy').T
        Tcycle_pres = 1.0
        normalized_valve_times = np.load(f'reference_pressure_trace/normalized_valve_times.npz')

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


def load_atrial_pressure_trace(atrial_pres_path=None, use_ref_pres_atrial_trace=True):
    if atrial_pres_path is None and not use_ref_pres_atrial_trace:
        raise ValueError("Either 'atrial_pres_path' must be provided or 'use_ref_pres_atrial_trace' must be set to True")
    
    if atrial_pres_path is not None and use_ref_pres_atrial_trace:
        use_ref_pres_atrial_trace = False
        print("Warning: Both 'atrial_pres_path' and 'use_ref_pres_atrial_trace' are set. Setting 'use_ref_pres_atrial_trace' to False.")

    if use_ref_pres_atrial_trace:
        time_atrial_pres, atrial_pres = np.load('reference_pressure_trace/normalized_atrial_pressure.npy').T
        Tcycle_pres = 1.0
        normalized_atrial_valve_times = np.load(f'reference_pressure_trace/atrial_valve_times.npz')
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
        time_aorta_pres, aorta_pres = np.load('reference_pressure_trace/normalized_aorta_pressure.npy').T
        Tcycle_pres = 1.0
        normalized_aorta_valve_times = np.load(f'reference_pressure_trace/aorta_valve_times.npz')
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
    elif' pvc' in valve_times:
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
    pres = rescale_pressure_magnitude(norm_pres, edp, sysp)
    pres, time_pres = rescale_pressure_time(norm_pres_time, pres, valve_times, normalized_valve_times)

    return pres, time_pres



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
    vol = vol_func(time + shift)
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


def clip_pv_loop_by_passive(pres_func, vol_func, vol_passive, pres_passive):
    time = np.linspace(-1, 2, 1501)

    pres_pv = pres_func(time)
    vol_pv = vol_func(time)

    passive = interp1d(vol_passive, pres_passive, fill_value='extrapolate')
    passive_pres = passive(vol_pv)

    pres_pv[pres_pv < passive_pres] = passive_pres[pres_pv < passive_pres]

    pres_func = interp1d(time, pres_pv)
    return pres_func


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
    


"""
PLOT FUNCTIONS
"""


def plot_pv_loop_traces(lv_vol_func, lv_pres_func, lv_valve_times, 
                        rv_vol_func=None, rv_pres_func=None, rv_valve_times=None, 
                        lv_vol_func_lin=None, rv_func_lin=None,
                        lv_vol_passive=None, lv_pres_passive=None, 
                        rv_vol_passive=None, rv_pres_passive=None, 
                        la_pres_func=None, ra_pres_func=None, 
                        aorta_pres_func=None):
    
    time_pv = np.linspace(0, 1, 1000)
    lv_pres_pv = lv_pres_func(time_pv)
    lv_vol_pv = lv_vol_func(time_pv)
    if rv_pres_func is not None and rv_vol_func is not None:
        rv_vol_pv = rv_vol_func(time_pv)
        rv_pres_pv = rv_pres_func(time_pv)

    print(rv_vol_func, rv_pres_func)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,8))

    # Pressure vs Volume subplot
    ax1.plot(lv_vol_pv / 1000, lv_pres_pv * 7.50062, 'r-', label='LV Pressure-Volume Loop')
    if (rv_vol_pv is not None) and (rv_pres_pv is not None):
        ax1.plot(rv_vol_pv / 1000, rv_pres_pv * 7.50062, 'b-', label='RV Pressure-Volume Loop')
    if lv_vol_passive is not None and lv_pres_passive is not None:
        ax1.plot(lv_vol_passive / 1000, lv_pres_passive * 7.50062, 'r--', label='LV Passive Loop')
    if rv_vol_passive is not None and rv_pres_passive is not None:
        ax1.plot(rv_vol_passive / 1000, rv_pres_passive * 7.50062, 'b--', label='RV Passive Loop')
    ax1.set_xlabel('Volume (ml)')
    ax1.set_ylabel('Pressure (mmHg)')
    ax1.set_title('Pressure vs Volume')
    ax1.grid(True)

    # Pressure subplot
    ax2.plot(time_pv, lv_pres_pv * 7.50062, 'r-', label='LV Pressure')
    if rv_pres_func is not None:
        ax2.plot(time_pv, rv_pres_pv * 7.50062, 'b-', label='RV Pressure')

    if la_pres_func is not None:
        la_pres_pv = la_pres_func(time_pv)
        ax2.plot(time_pv, la_pres_pv * 7.50062, 'r-', alpha=0.5, label='LA Pressure')
    if ra_pres_func is not None:
        ra_pres_pv = ra_pres_func(time_pv)
        ax2.plot(time_pv, ra_pres_pv * 7.50062, 'b-', alpha=0.5, label='RA Pressure')
    if aorta_pres_func is not None:
        aorta_pres_pv = aorta_pres_func(time_pv)
        ax2.plot(time_pv, aorta_pres_pv * 7.50062, 'g-', label='Aorta Pressure')

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Pressure (mmHg)')
    ax2.set_title('Pressure over Time')
    ax2.grid(True)

    # Volume subplot
    ax4.plot(time_pv, lv_vol_pv / 1000, 'r-', label='LV Volume')
    if rv_vol_func is not None:
        ax4.plot(time_pv, rv_vol_pv / 1000, 'b-', label='RV Volume')
    if lv_vol_func_lin is not None:
        ax4.plot(time_pv, lv_vol_func_lin(time_pv) / 1000, 'r--', label='LV Volume (Linear)')
    if rv_func_lin is not None:
        ax4.plot(time_pv, rv_func_lin(time_pv) / 1000, 'b--', label='RV Volume (Linear)')

    # Add valves
    for ax in (ax2, ax4):
        for key, val in lv_valve_times.items():
            if key != 'tcycle':
                ax.axvline(x=val, color='k', linestyle='--', alpha=0.7)
        if rv_valve_times is not None:
            for key, val in rv_valve_times.items():
                if key != 'tcycle':
                    ax.axvline(x=val, color='k', linestyle='--', alpha=0.7)

    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Volume (ml)')
    ax4.set_title('Volume over Time')
    ax4.grid(True)

    # Remove the unused subplot and use it for the legend
    fig.delaxes(ax3)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles[:2], ['LV', 'RV'], loc='center', bbox_to_anchor=(0.1, 0.1, 0.5, 0.5), frameon=False)

    plt.tight_layout()