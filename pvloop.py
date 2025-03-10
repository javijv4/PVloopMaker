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
import pvfunctions as pvf

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

use_ref_pres_atrial_trace = True

use_ref_pres_aorta_trace = True


#%% create output directory
if not os.path.exists(out_path):
    os.makedirs(out_path)

#%% Load traces
# Load volume traces
if use_rv:
    time_vol, lv_vol, rv_vol, Tcycle_vol = pvf.load_volume_traces(lv_vol_path, rv_vol_path, use_rv=use_rv)
    Tcycle_vol = 1.0
else:
    time_vol, lv_vol, Tcycle_vol = pvf.load_volume_traces(lv_vol_path, use_rv=use_rv)
    Tcycle_vol = 1.0

lv_vol_func_lin = pvf.get_pv_functions(time_vol, lv_vol, interp='linear')
rv_vol_func_lin = pvf.get_pv_functions(time_vol, rv_vol, interp='linear')

# Load pressure traces
if use_ref_pres_trace:
    if use_rv:
        time_pres, lv_pres, rv_pres, Tcycle_pres, normalized_valve_times = pvf.load_pressure_traces(lv_pres_path=None, 
                                                                                                    rv_pres_path=None, 
                                                                                                    use_rv=use_rv, 
                                                                                                    use_ref_pres_trace=use_ref_pres_trace)
    else:
        time_pres, lv_pres, Tcycle_pres, normalized_valve_times = pvf.load_pressure_traces(lv_pres_path, use_rv=use_rv,
                                                                                           use_ref_pres_trace=use_ref_pres_trace)
else:
    if use_rv:
        time_pres, lv_pres, rv_pres, Tcycle_pres = pvf.load_pressure_traces(lv_pres_path, rv_pres_path, use_rv=use_rv,
                                                                                           use_ref_pres_trace=use_ref_pres_trace)
    else:
        time_pres, lv_pres, Tcycle_pres = pvf.load_pressure_traces(lv_pres_path, use_rv=use_rv,
                                                                                           use_ref_pres_trace=use_ref_pres_trace)


#%% Isovolumic phases
# Add isovolumic phases if needed
if add_ivr_points:
    time_vol_lv = time_vol.copy()
    time_vol_lv, lv_vol, lv_valve_times = pvf.add_isovolumic_relaxation(time_vol_lv, lv_vol, lv_valve_times, Tcycle_vol, ivr_time)
    if use_rv:
        time_vol_rv = time_vol.copy()
        time_vol_rv, rv_vol, rv_valve_times = pvf.add_isovolumic_relaxation(time_vol_rv, rv_vol, rv_valve_times, Tcycle_vol, ivr_time)

if add_ivc_points:
    time_vol_lv, lv_vol, lv_valve_times = pvf.add_isovolumic_contraction(time_vol_lv, lv_vol, lv_valve_times, Tcycle_vol, ivc_time)
    if use_rv:
        time_vol_rv, rv_vol, rv_valve_times = pvf.add_isovolumic_contraction(time_vol_rv, rv_vol, rv_valve_times, Tcycle_vol, ivc_time)

#%% Pressure traces
if use_ref_pres_trace:
    # Need to rescale
    norm_time = time_pres.copy()
    lv_pres, time_pres = pvf.rescale_normalized_pressure_trace(norm_time, lv_pres, lv_valve_times, normalized_valve_times, lv_edp, lv_sysp)

    if use_rv:
        rv_pres, _ = pvf.rescale_normalized_pressure_trace(norm_time, rv_pres, lv_valve_times, normalized_valve_times, rv_edp, rv_sysp)


#%% Get PV functions
lv_vol_func = pvf.get_pv_functions(time_vol_lv, lv_vol, interp='pchip')
lv_pres_func = pvf.get_pv_functions(time_pres, lv_pres, interp='linear')
if use_rv:
    rv_vol_func = pvf.get_pv_functions(time_vol_rv, rv_vol, interp='pchip')
    rv_pres_func = pvf.get_pv_functions(time_pres, rv_pres, interp='linear')
        

#%% PV loop shift
vol_shift = -0.12  

# Update function
lv_vol_func = pvf.shift_vol_func(lv_vol_func, vol_shift)
lv_vol_func_lin = pvf.shift_vol_func(lv_vol_func_lin, vol_shift)
if use_rv:
    rv_vol_func = pvf.shift_vol_func(rv_vol_func, vol_shift)
    rv_vol_func_lin = pvf.shift_vol_func(rv_vol_func_lin, vol_shift)


#%% Shift to ED
if shift_to_ed:
    lv_pres_func, lv_vol_func = pvf.shift_to_ed(lv_pres_func, lv_vol_func, lv_edp, lv_edv)
    if use_rv:
        rv_pres_func, rv_vol_func = pvf.shift_to_ed(rv_pres_func, rv_vol_func, rv_edp, rv_edv)


#%% Delete part of the PV loop below the passive curve
if clip_pv_loop_by_passive:
    lv_pres_func = pvf.clip_pv_loop_by_passive(lv_pres_func, lv_vol_func, lv_vol_passive, lv_pres_passive)
    if use_rv:
        rv_pres_func = pvf.clip_pv_loop_by_passive(rv_pres_func, rv_vol_func, rv_vol_passive, rv_pres_passive)

#%% Deal with the atria
if use_ref_pres_atrial_trace:
    time_atrial_pres, atrial_pres, Tcycle_pres, normalized_atrial_valve_times = pvf.load_atrial_pressure_trace(use_ref_pres_atrial_trace=True)

    time_la_pres, la_pres = pvf.scale_atrial_pressure_magnitude(time_atrial_pres, atrial_pres, normalized_atrial_valve_times, lv_pres_func, lv_valve_times)
    time_la_pres, la_pres = pvf.scale_atrial_pressure_time(time_la_pres, la_pres, normalized_atrial_valve_times, lv_valve_times)
    time_la_pres, la_pres = pvf.fix_diastolic_atrial_pressure(time_la_pres, la_pres, lv_pres_func, lv_valve_times)

    time_ra_pres, ra_pres = pvf.scale_atrial_pressure_magnitude(time_atrial_pres, atrial_pres, normalized_atrial_valve_times, rv_pres_func, rv_valve_times)
    time_ra_pres, ra_pres = pvf.scale_atrial_pressure_time(time_ra_pres, ra_pres, normalized_atrial_valve_times, rv_valve_times)
    time_ra_pres, ra_pres = pvf.fix_diastolic_atrial_pressure(time_ra_pres, ra_pres, rv_pres_func, rv_valve_times)

    la_pres_func = pvf.get_pv_functions(time_la_pres, la_pres, interp='linear')
    ra_pres_func = pvf.get_pv_functions(time_ra_pres, ra_pres, interp='linear')


#%% Deal with the aorta
if use_ref_pres_aorta_trace:
    time_aorta_pres, aorta_pres, Tcycle_pres, normalized_aorta_valve_times = pvf.load_aorta_pressure_trace(use_ref_pres_aorta_trace=True)

    time_aorta_pres, aorta_pres = pvf.scale_aorta_pressure_magnitude(time_aorta_pres, aorta_pres, normalized_aorta_valve_times, lv_pres_func, lv_valve_times)
    time_aorta_pres, aorta_pres = pvf.scale_aorta_pressure_time(time_aorta_pres, aorta_pres, normalized_aorta_valve_times, lv_valve_times)
    time_aorta_pres, aorta_pres = pvf.fix_systolic_aorta_pressure(time_aorta_pres, aorta_pres, lv_pres_func, lv_valve_times)

    aorta_pres_func = pvf.get_pv_functions(time_aorta_pres, aorta_pres, interp='linear')


#%% Save files
time_pv = np.linspace(0, 1, 1000)
chio.write_dfile(f'{out_path}/lv_pressure.INIT', np.array([time_pv, lv_pres_func(time_pv)]).T)
chio.write_dfile(f'{out_path}/lv_volume.INIT', np.array([time_pv, lv_vol_func(time_pv)]).T)
chio.write_dfile(f'{out_path}/rv_pressure.INIT', np.array([time_pv, rv_pres_func(time_pv)]).T)
chio.write_dfile(f'{out_path}/rv_volume.INIT', np.array([time_pv, rv_vol_func(time_pv)]).T)
chio.write_dfile(f'{out_path}/la_pressure.INIT', np.array([time_pv, la_pres_func(time_pv)]).T)
chio.write_dfile(f'{out_path}/ra_pressure.INIT', np.array([time_pv, ra_pres_func(time_pv)]).T)
chio.write_dfile(f'{out_path}/aorta_pressure.INIT', np.array([time_pv, aorta_pres_func(time_pv)]).T)


#%% Plots
pvf.plot_pv_loop_traces(lv_vol_func, lv_pres_func, lv_valve_times, 
                        rv_vol_func=rv_vol_func, rv_pres_func=rv_pres_func, rv_valve_times=rv_valve_times, 
                        lv_vol_func_lin=lv_vol_func_lin, rv_func_lin=rv_vol_func_lin,
                        lv_vol_passive=lv_vol_passive, lv_pres_passive=lv_pres_passive, 
                        rv_vol_passive=rv_vol_passive, rv_pres_passive=rv_pres_passive,
                        la_pres_func=la_pres_func, ra_pres_func=ra_pres_func,
                        aorta_pres_func=aorta_pres_func)
plt.savefig(f'{out_path}/pvloop.png', dpi=180, bbox_inches='tight')
plt.show()