#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/08/29 16:53:33

@author: Javiera Jilberto Vallejos 
'''

import numpy as np
import matplotlib.pyplot as plt
import cheartio as chio
from scipy.interpolate import interp1d
from matplotlib.gridspec import GridSpec

def concat_cycles(time, var):
    mask = (time > 0) & (time < 1)
    time = time[mask]
    var = var[mask]
    times = [time[:-1] - max(time), time, time[1:] + max(time)]
    vars = [var[:-1], var, var[1:]]
    return np.concatenate(times), np.concatenate(vars)

plt.rcParams.update({'font.size': 14, 'font.family': 'Liberation Sans'})

xlim = (-0.1, 1.1)
ivc = (0, 0.06)
ivr = (0.38, 0.43)

time_lv_pres, lv_pres = np.loadtxt('lv_pressure.txt', delimiter=',').T
time_lv_pres, lv_pres = np.sort(time_lv_pres), lv_pres[np.argsort(time_lv_pres)]
time_lv_pres, lv_pres = concat_cycles(time_lv_pres, lv_pres)

time_lv_vol, lv_vol = np.loadtxt('lv_volume.txt', delimiter=',').T
time_lv_vol, lv_vol = np.sort(time_lv_vol), lv_vol[np.argsort(time_lv_vol)]
time_lv_vol, lv_vol = concat_cycles(time_lv_vol, lv_vol)

time_la_pres, la_pres = np.loadtxt('la_pressure.txt', delimiter=',').T
time_la_pres, la_pres = np.sort(time_la_pres), la_pres[np.argsort(time_la_pres)]
time_la_pres, la_pres = concat_cycles(time_la_pres, la_pres)

time_aorta_pres, aorta_pres = np.loadtxt('aorta_pressure.txt', delimiter=',').T
time_aorta_pres, aorta_pres = np.sort(time_aorta_pres), aorta_pres[np.argsort(time_aorta_pres)]
time_aorta_pres, aorta_pres = concat_cycles(time_aorta_pres, aorta_pres)

# Create smooth interpolation functions for each variable
lv_pres_func = interp1d(time_lv_pres, lv_pres, kind='cubic', fill_value="extrapolate")
lv_vol_func = interp1d(time_lv_vol, lv_vol, kind='cubic', fill_value="extrapolate")
la_pres_func = interp1d(time_la_pres, la_pres, kind='cubic', fill_value="extrapolate")
aorta_pres_func = interp1d(time_aorta_pres, aorta_pres, kind='cubic', fill_value="extrapolate")

# Create a common normalized time array from 0 to 1
n_points = 1000
time = np.linspace(xlim[0], xlim[1], n_points)

# Map time to each variable's time range
lv_pres_eval = lv_pres_func(time)
lv_vol_eval = lv_vol_func(time)
la_pres_eval = la_pres_func(time)
aorta_pres_eval = aorta_pres_func(time)


#%%
fig = plt.figure(figsize=(6, 6))
gs = GridSpec(2, 1, height_ratios=[1, 0.6])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)

# Pressure traces (interpolated)
ax1.plot(time, lv_pres_eval, label='LV Pressure (interp)', color='black')
ax1.plot(time, la_pres_eval, label='LA Pressure (interp)', color='blue')
ax1.plot(time, aorta_pres_eval, label='Aorta Pressure (interp)', color='red')
ax1.set_ylabel('Pressure (mmHg)')
ax1.grid(True)
ax1.set_xlim(xlim)


# Volume trace (original data)
ax2.plot(time, lv_vol_eval, label='LV Volume (interp)', color='black')
ax2.set_xlabel('Normalized Time (0-1)')
ax2.set_ylabel('LV Volume')
ax2.grid(True)

ylim1 = ax1.get_ylim()
ylim2 = ax2.get_ylim()

# Highlight IVC and IVR phases on pressure plot
ax1.fill_betweenx(
    y=ax1.get_ylim(), x1=ivc[0], x2=ivc[1], color='orange', alpha=0.2, label='IVC'
)
ax1.fill_betweenx(
    y=ax1.get_ylim(), x1=ivr[0], x2=ivr[1], color='cyan', alpha=0.2, label='IVR'
)

# Highlight IVC and IVR phases on volume plot
ax2.fill_betweenx(
    y=ylim1, x1=ivc[0], x2=ivc[1], color='orange', alpha=0.2, label='IVC'
)
ax2.fill_betweenx(
    y=ylim2, x1=ivr[0], x2=ivr[1], color='cyan', alpha=0.2, label='IVR'
)

ax1.set_ylim(ylim1)
ax2.set_ylim(ylim2)
ax2.set_yticks(np.arange(20, 81, 20))

# Remove all spines except the left one
for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])

plt.tight_layout()
plt.savefig('wiggers_diagram.svg', bbox_inches='tight', dpi=300)
plt.show()
