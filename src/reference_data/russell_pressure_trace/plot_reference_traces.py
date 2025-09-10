#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/02/21 12:19:24

@author: Javiera Jilberto Vallejos 
'''

import numpy as np
import matplotlib.pyplot as plt


time_pres, lv_pres = np.load('normalized_human_pressure.npy').T
Tcycle_pres = 1.0
normalized_valve_times = np.load(f'normalized_valve_times.npz')

plt.figure(figsize=(10, 6))
plt.plot(time_pres, lv_pres, 'k')

for valve, time in normalized_valve_times.items():
    plt.axvline(x=time, linestyle='--', label=f'{valve} time')

plt.xlabel('Time (s)')
plt.ylabel('Pressure (mmHg)')
plt.title('LV Pressure Trace with Valve Times')
plt.legend()
plt.grid(True)
plt.show()