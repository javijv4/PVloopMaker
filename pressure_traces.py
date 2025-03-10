#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/02/20 15:56:31

@author: Javiera Jilberto Vallejos 
'''

import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize
from scipy.interpolate import interp1d

def find_traces_intersection(trace1, trace2):
    if np.max(trace1) < np.max(trace2):
        p_diff = trace2 - trace1
    else:
        p_diff = trace2 - trace1
    diff_p_diff = np.diff(time[p_diff > 0])
    idx = np.where(diff_p_diff > dt*2)[0]
    inter_times = (time[p_diff > 0][idx], time[p_diff > 0][idx+1])

    func = interp1d(time, p_diff, kind='linear')
    time_intersect = np.zeros(len(inter_times))
    for i, t in enumerate(inter_times):
        time_intersect[i] = optimize.root_scalar(func, x0=t, method='newton').root[0]

    return time_intersect


# Read traces
time, p_lv = np.loadtxt('0D_pressure_traces/results__p_v_l.txt', unpack=True)
dt = time[1] - time[0]
_, p_rv = np.loadtxt('0D_pressure_traces/results__p_v_r.txt', unpack=True)
_, p_la = np.loadtxt('0D_pressure_traces/results__p_at_l.txt', unpack=True)
_, p_ra = np.loadtxt('0D_pressure_traces/results__p_at_r.txt', unpack=True)
_, p_aorta = np.loadtxt('0D_pressure_traces/results__p_ar_sys.txt', unpack=True)
_, p_pul = np.loadtxt('0D_pressure_traces/results__p_ar_pul.txt', unpack=True)

time = time[:1000]
p_lv = p_lv[-1000:]
p_rv = p_rv[-1000:]
p_la = p_la[-1000:]
p_ra = p_ra[-1000:]
p_aorta = p_aorta[-1000:]
p_pul = p_pul[-1000:]

# Find valve events LA
la_left_events = find_traces_intersection(p_lv, p_la)
# la_left_events = np.append(la_left_events, find_traces_intersection(p_lv, p_aorta))
la_left_events_idx = (la_left_events/dt).astype(int)
la_left_events = np.append(la_left_events, time[la_left_events_idx[0]:la_left_events_idx[1]][np.argmin(p_la[la_left_events_idx[0]:la_left_events_idx[1]])])
la_left_events = np.append(la_left_events, time[la_left_events_idx[1]:][np.argmin(p_la[la_left_events_idx[1]:])])

p_la_ext = np.concatenate((p_la, p_la, p_la))
time_ext = np.arange(-1, 2, dt)
func = interp1d(time_ext, p_la_ext, kind='cubic')
save_trace = func(time+la_left_events[0])
save_trace = save_trace/save_trace[0]
save = np.column_stack((time, save_trace))
events = la_left_events - la_left_events[0]
events = np.sort(events)
valve_events = {'mvc': events[0], 'tvc': events[0], 'sys': events[1], 
                'mvo': events[2], 'tvo': events[2], 'dias': events[3], 'tcycle': 1.0}
np.save('reference_pressure_trace/normalized_atrial_pressure.npy', save)
np.savez('reference_pressure_trace/atrial_valve_times.npz', **valve_events)


# Find valve events Aorta
p_aorta[-1] = p_aorta[0]
aorta_left_events = find_traces_intersection(p_lv, p_aorta)
aorta_left_events_idx = (aorta_left_events/dt).astype(int)
aorta_left_events = np.append(aorta_left_events, time[aorta_left_events_idx[0]:aorta_left_events_idx[1]][np.argmax(p_aorta[aorta_left_events_idx[0]:aorta_left_events_idx[1]])])
aorta_left_events = np.append(aorta_left_events, time[aorta_left_events_idx[1]:][np.argmax(p_aorta[aorta_left_events_idx[1]:])])

p_aorta_ext = np.concatenate((p_aorta, p_aorta, p_aorta))
time_ext = np.arange(-1, 2, dt)
func = interp1d(time_ext-la_left_events[0], p_aorta_ext, kind='cubic')
save_trace = func(time)
save_trace = save_trace/save_trace[0]
save = np.column_stack((time, save_trace))
events = aorta_left_events - la_left_events[0] -.002
events = np.sort(events)
valve_events = {'avo': events[0], 'avc': events[2], 'sys': events[1], 
                'pvo': events[2], 'pvc': events[2], 'tcycle': 1.0}
np.save('reference_pressure_trace/normalized_aorta_pressure.npy', save)
np.savez('reference_pressure_trace/aorta_valve_times.npz', **valve_events)

# Load lv pressure traces
time_pres, lv_pres = np.load('reference_pressure_trace/normalized_human_pressure_og.npy').T
normalized_valve_times = dict(np.load(f'reference_pressure_trace/normalized_valve_times_og.npz'))

value = 0.06
normalized_valve_times['avo'] -= 0.025
normalized_valve_times['pvo'] -= 0.025
normalized_valve_times['avc'] -= value
normalized_valve_times['mvo'] -= value
normalized_valve_times['pvc'] -= value
normalized_valve_times['tvo'] -= value

func_pres = interp1d(time_pres, lv_pres, kind='linear')
print(func_pres(normalized_valve_times['avo'])*120)

np.save('reference_pressure_trace/normalized_human_pressure.npy', np.column_stack((time_pres, lv_pres)))
np.savez('reference_pressure_trace/normalized_valve_times.npz', **normalized_valve_times)

# Plot the pressure trace and the normalized valve times
plt.figure(figsize=(5, 3))
plt.plot(time_pres, lv_pres, label='Normalized LV Pressure',)
for key, value in normalized_valve_times.items():
    plt.axvline(x=value, linestyle='--', label=f'{key} event')
plt.xlabel('Time (s)')
plt.ylabel('Pressure (mmHg)')
plt.title('Normalized LV Pressure Trace with Valve Times')
# plt.legend(frameon=False)
plt.grid(True)
plt.savefig('normalized_lv_pressure_trace.png', bbox_inches='tight', dpi=180)
plt.show()

# # Plot left side
# plt.figure(figsize=(5, 3))
# plt.plot(time, p_lv, label='Left Ventricle Pressure')
# plt.plot(time, p_la, label='Left Atrium Pressure')
# plt.plot(time, p_aorta, label='Aorta Pressure')
# # plt.scatter(la_left_events, np.interp(la_left_events, time, p_aorta), color='red', marker='x', label='Valve Events')
# plt.xlabel('Time (s)')
# plt.ylabel('Pressure (mmHg)')
# plt.title('Left Side Pressure Traces')
# plt.legend(frameon=False)
# plt.grid(True)
# plt.savefig('0d_traces.png', bbox_inches='tight', dpi=180)
# plt.show()

# # Plot saved trace
# plt.figure(figsize=(10, 6))
# plt.plot(time, save_trace, label='Left Atrium Pressure')
# plt.scatter(events, np.interp(events, time, save_trace), color='red', marker='x', label='Valve Events')
# plt.xlabel('Time (s)')
# plt.ylabel('Pressure (mmHg)')