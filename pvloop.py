#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/02/05 15:56:31

@author: Javiera Jilberto Vallejos 
'''

import os
import numpy as np
import cheartio as chio
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator, interp1d

def check_traces(time, trace):
    if trace[0] != trace[-1]:
        trace = np.concatenate((trace, trace[0:1]))
        time = np.linspace(0, 1, len(trace))
    return time, trace

def repeat_traces(trace):
    return np.concatenate((trace, trace[1:], trace[1:]))

def repeat_time(time):
    Tf = time[-1]
    return np.concatenate((time-Tf, time[1:], time[1:] + Tf))

def rescale_pressure_magnitude(pressure, ed_pressure, es_pressure):
    pressure = pressure.copy()

    # Calculate mean pressure
    mbp = 2/3*ed_pressure + 1/3*es_pressure
    sp = es_pressure
    dp = ed_pressure

    aux = pressure.copy()
    a = (sp - dp)/(1.0 - aux[0])
    pressure[aux > aux[0]] = aux[aux > aux[0]]*a + sp - a
    pressure[aux <= aux[0]] = aux[aux <= aux[0]]/aux[0]*dp
    return pressure

def rescale_pressure_time(normalized_time, normalized_pressure, valve_times, normalized_valve_times, which='lv'):
    if which == 'lv':
        events = ['mvc', 'avo', 'avc', 'mvo', 'tcycle']
    elif which == 'rv':
        events = ['tvc', 'pvo', 'pvc', 'tvo', 'tcycle']
    else:
        raise ValueError('which must be either lv or rv')
        
    chamber_pressure = []
    pressure_time = []

    normalized_func = interp1d(normalized_time, normalized_pressure)
    for i in range(1, len(events)):
        x = np.linspace(normalized_valve_times[events[i-1]],normalized_valve_times[events[i]],1001, endpoint=True)
        segment_pressure = normalized_func(x)
        segment_time = np.linspace(valve_times[events[i-1]],valve_times[events[i]],1001, endpoint=True)

        chamber_pressure.append(segment_pressure[1:])
        pressure_time.append(segment_time[1:])

    # Check when is diastasis
    dias_time = 0.2625*(segment_time[-1] - segment_time[0]) + segment_time[0]

    chamber_pressure = np.hstack(chamber_pressure)
    pressure_time = np.hstack(pressure_time)
    chamber_pressure = np.append(normalized_pressure[0], chamber_pressure)
    pressure_time = np.append(0, pressure_time)
    return chamber_pressure, pressure_time

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


#%% Inputs
out_path = 'pvloop'

lv_vol_path = 'data/lv_volume.INIT'
lv_pres_path = 'data/lv_pressure.INIT'

use_rv = True
rv_vol_path = 'data/rv_volume.INIT'
rv_pres_path = 'data/rv_pressure.INIT'

use_ref_pres_trace = True
lv_edp = 1.33322
lv_sysp = 16.0
rv_edp = 0.53329
rv_sysp = 3.33306

add_ivc_points = True
ivc_time = 0.05     # seconds  (according to wikipedia)

add_ivr_points = True
ivr_time = 0.07     # seconds

shift_to_ed = True
lv_edv = 195083.80281090742
rv_edv = 253861.50516986853

use_valve_times = 'fromIV'  # or from data
lv_valve_times = {'mvc': 0.0, 'avo': 0.05, 'avc': 0.4, 'mvo': 0.47, 'tcycle': 1.0}  # MV close, AV open, MV open, AV close
rv_valve_times = {'tvc': 0.0, 'pvo': 0.05, 'pvc': 0.4, 'tvo': 0.47, 'tcycle': 1.0}  # TV close, PV open, TV open, PV close

clip_pv_loop_by_passive = True
lv_vol_passive = chio.read_scalar_dfiles('data/out/LV_Vol', (1,100,1))
lv_pres_passive = chio.read_scalar_dfiles('data/out/LV_LM', (1,100,1))
rv_vol_passive = chio.read_scalar_dfiles('data/out/RV_Vol', (1,100,1))
rv_pres_passive = chio.read_scalar_dfiles('data/out/RV_LM', (1,100,1))

#%% create output directory
if not os.path.exists(out_path):
    os.makedirs(out_path)

#%% Load traces
# Load volume traces
time_vol, lv_vol = chio.read_dfile(lv_vol_path).T
Tcycle_vol = 1. #time_vol[-1] # seconds
time_vol, lv_vol = check_traces(time_vol, lv_vol)

if use_rv:
    _, rv_vol = chio.read_dfile(rv_vol_path).T
    time_vol, rv_vol = check_traces(time_vol, rv_vol)


# Load volume traces
if use_ref_pres_trace:
    norm_pres_time, norm_pres = np.load('reference_pressure_trace/normalized_human_pressure.npy').T
    normalized_valve_times = np.load(f'reference_pressure_trace/normalized_valve_times.npz')
else:
    time_pres, lv_pres = chio.read_dfile(lv_pres_path).T
    Tcycle_pres = 1. #time_pres[-1] # seconds
    time_pres, lv_pres = check_traces(time_pres, lv_pres)
    if use_rv:
        _, rv_pres = chio.read_dfile(rv_pres_path).T
        time_pres, rv_pres = check_traces(time_pres, rv_pres)


#%% Isovolumic phases
# Add isovolumic phases if needed
if add_ivr_points:
    time_vol, lv_vol, iv_times = add_iv_datapoints(time_vol, lv_vol, ivr_time/Tcycle_vol, 'min')
    lv_valve_times['avc'] = iv_times[0]
    lv_valve_times['mvo'] = iv_times[1]
    if use_rv:
        _, rv_vol, iv_times = add_iv_datapoints(time_vol, rv_vol, ivr_time/Tcycle_vol, 'min')
        rv_valve_times['pvc'] = iv_times[0]
        rv_valve_times['tvo'] = iv_times[1]

if add_ivc_points:
    time_vol, lv_vol, iv_times = add_iv_datapoints(time_vol, lv_vol, ivc_time/Tcycle_vol, 'max')
    lv_valve_times['mvc'] = iv_times[0]
    lv_valve_times['avo'] = iv_times[1]
    if use_rv:
        _, rv_vol, iv_times = add_iv_datapoints(time_vol, rv_vol, ivc_time/Tcycle_vol, 'max')
        rv_valve_times['pvc'] = iv_times[0]
        rv_valve_times['tvo'] = iv_times[1]
    
    # Shift valve times
    vol_shift = -lv_valve_times['mvc']
    for key in lv_valve_times:
        if key != 'tcycle':
            lv_valve_times[key] += vol_shift
            if lv_valve_times[key] < 0:
                lv_valve_times[key] += 1.0
                
    for key in rv_valve_times:
        if key != 'tcycle':
            rv_valve_times[key] += vol_shift
            if rv_valve_times[key] < 0:
                rv_valve_times[key] += 1.0

    if np.abs(vol_shift) > 0.5:
        vol_shift = 1.0 - vol_shift

#%% Volume traces
# Extend traces
lv_vol_ext = repeat_traces(lv_vol)
time_vol_ext = repeat_time(time_vol)

# Smooth traces
lv_vol_func_og = interp1d(time_vol_ext, lv_vol_ext)
lv_vol_func = PchipInterpolator(time_vol_ext, lv_vol_ext)

if use_rv:
    rv_vol_ext = repeat_traces(rv_vol)
    rv_vol_func_og = interp1d(time_vol_ext, rv_vol_ext)
    rv_vol_func = PchipInterpolator(time_vol_ext, rv_vol_ext)

#%% Pressure traces
if use_ref_pres_trace:
    # Need to rescale
    lv_pres = rescale_pressure_magnitude(norm_pres, lv_edp, lv_sysp)
    lv_pres, time_pres = rescale_pressure_time(norm_pres_time, lv_pres, lv_valve_times, normalized_valve_times, which='lv')

    if use_rv:
        rv_pres = rescale_pressure_magnitude(norm_pres, rv_edp, rv_sysp)
        rv_pres, _ = rescale_pressure_time(norm_pres_time, rv_pres, rv_valve_times, normalized_valve_times, which='rv')

lv_pres_ext = repeat_traces(lv_pres)
time_pres_ext = repeat_time(time_pres)
lv_pres_func = interp1d(time_pres_ext, lv_pres_ext)

if use_rv:
    rv_pres_ext = repeat_traces(rv_pres)
    rv_pres_func = interp1d(time_pres_ext, rv_pres_ext)
    

#%% PV loop optimization
time_pv = np.linspace(-1, 2, 1501)

vol_shift = -0.08

# Apply shift
lv_vol_pv = lv_vol_func(time_pv+vol_shift)
rv_vol_pv = rv_vol_func(time_pv+vol_shift)

# Update function
lv_vol_func = interp1d(time_pv, lv_vol_pv)
rv_vol_func = interp1d(time_pv, rv_vol_pv)


#%% Shift to ED
if shift_to_ed:
    # Grab pressure and volume arrays
    lv_pres_pv = lv_pres_func(time_pv)
    rv_pres_pv = rv_pres_func(time_pv)
    lv_vol_pv = lv_vol_func(time_pv)
    rv_vol_pv = rv_vol_func(time_pv)

    lv_vol_pv = lv_vol_pv - lv_vol_func(0.) + lv_edv
    rv_vol_pv = rv_vol_pv - rv_vol_func(0.) + rv_edv
    lv_pres_pv = lv_pres_pv - lv_pres_func(0.) + lv_edp
    rv_pres_pv = rv_pres_pv - rv_pres_func(0.) + rv_edp

    # Update function
    lv_vol_func = interp1d(time_pv, lv_vol_pv)
    rv_vol_func = interp1d(time_pv, rv_vol_pv)
    lv_pres_func = interp1d(time_pv, lv_pres_pv)
    rv_pres_func = interp1d(time_pv, rv_pres_pv)

    assert lv_vol_func(0.) == lv_edv
    assert rv_vol_func(0.) == rv_edv
    assert lv_pres_func(0.) == lv_edp
    assert rv_pres_func(0.) == rv_edp


#%% Delete part of the PV loop below the passive curve
if clip_pv_loop_by_passive:
    lv_pres_pv = lv_pres_func(time_pv)
    rv_pres_pv = rv_pres_func(time_pv)
    lv_vol_pv = lv_vol_func(time_pv)
    rv_vol_pv = rv_vol_func(time_pv)

    lv_passive = interp1d(lv_vol_passive, lv_pres_passive, fill_value='extrapolate')
    rv_passive = interp1d(rv_vol_passive, rv_pres_passive, fill_value='extrapolate')
    
    lv_passive_pres = lv_passive(lv_vol_pv)
    rv_passive_pres = rv_passive(rv_vol_pv)

    lv_pres_pv[lv_pres_pv < lv_passive_pres] = lv_passive_pres[lv_pres_pv < lv_passive_pres]
    rv_pres_pv[rv_pres_pv < rv_passive_pres] = rv_passive_pres[rv_pres_pv < rv_passive_pres]

    # Redo the interpolation
    lv_pres_func = interp1d(time_pv, lv_pres_pv)
    rv_pres_func = interp1d(time_pv, rv_pres_pv)


#%% Save files
time_pv = np.linspace(0, 1, 1000)
chio.write_dfile(f'{out_path}/lv_pressure.INIT', np.array([time_pv, lv_pres_func(time_pv)]).T)
chio.write_dfile(f'{out_path}/lv_volume.INIT', np.array([time_pv, lv_vol_func(time_pv)]).T)
chio.write_dfile(f'{out_path}/rv_pressure.INIT', np.array([time_pv, rv_pres_func(time_pv)]).T)
chio.write_dfile(f'{out_path}/rv_volume.INIT', np.array([time_pv, rv_vol_func(time_pv)]).T)


#%% Plots

lv_pres_pv = lv_pres_func(time_pv)
rv_pres_pv = rv_pres_func(time_pv)
lv_vol_pv = lv_vol_func(time_pv)
rv_vol_pv = rv_vol_func(time_pv)


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,8))

# Pressure vs Volume subplot
ax1.plot(lv_vol_pv / 1000, lv_pres_pv * 7.50062, 'r-', label='LV Pressure-Volume Loop')
if use_rv:
    ax1.plot(rv_vol_pv / 1000, rv_pres_pv * 7.50062, 'b-', label='RV Pressure-Volume Loop')
if clip_pv_loop_by_passive:
    ax1.plot(lv_vol_passive / 1000, lv_pres_passive * 7.50062, 'r--', label='LV Passive Loop')
    ax1.plot(rv_vol_passive / 1000, rv_pres_passive * 7.50062, 'b--', label='RV Passive Loop')
ax1.set_xlabel('Volume (ml)')
ax1.set_ylabel('Pressure (mmHg)')
ax1.set_title('Pressure vs Volume')

# Pressure subplot
ax2.plot(time_pv, lv_pres_pv * 7.50062, 'r-', label='LV Pressure')
if use_rv:
    ax2.plot(time_pv, rv_pres_pv * 7.50062, 'b-', label='RV Pressure')
for key, val in lv_valve_times.items():
    if key != 'tcycle':
        ax2.axvline(x=val, color='k', linestyle='--', alpha=0.7)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Pressure (mmHg)')
ax2.set_title('Pressure over Time')

# Volume subplot
ax4.plot(time_pv, lv_vol_pv / 1000, 'r-', label='LV Volume')
if use_rv:
    ax4.plot(time_pv, rv_vol_pv / 1000, 'b-', label='RV Volume')
ax4.plot(time_vol, lv_vol_func_og(time_vol + vol_shift) / 1000, 'ro-', alpha=0.5, label='LV Volume (Original)')
if use_rv:
    ax4.plot(time_vol, rv_vol_func_og(time_vol + vol_shift) / 1000, 'bo-', alpha=0.5, label='RV Volume (Original)')
for key, val in lv_valve_times.items():
    if key != 'tcycle':
        ax4.axvline(x=val, color='k', linestyle='--', alpha=0.7)
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Volume (ml)')
ax4.set_title('Volume over Time')

# Remove the unused subplot and use it for the legend
fig.delaxes(ax3)
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles[:2], ['LV', 'RV'], loc='center', bbox_to_anchor=(0.1, 0.1, 0.5, 0.5))

plt.tight_layout()
plt.savefig(f'{out_path}/pvloop.png', dpi=180, bbox_inches='tight')
plt.show()
