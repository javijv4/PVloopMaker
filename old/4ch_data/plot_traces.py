#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/08/29 16:53:33

@author: Javiera Jilberto Vallejos 
'''

import numpy as np
import matplotlib.pyplot as plt
import cheartio as chio

time, lv_pres = chio.read_dfile('lv_pressure.INIT').T
_, lv_vol = chio.read_dfile('lv_volume.INIT').T
_, la_pres = chio.read_dfile('la_pressure.INIT').T
_, aorta_pres = chio.read_dfile('aorta_pressure.INIT').T

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

# Pressure traces
ax1.plot(time, lv_pres, label='LV Pressure', color='red')
ax1.plot(time, la_pres, label='LA Pressure', color='green')
ax1.plot(time, aorta_pres, label='Aorta Pressure', color='purple')
ax1.set_ylabel('Pressure (mmHg)')
ax1.set_title('Wiggers Diagram - Pressures')
ax1.legend(loc='upper right')
ax1.grid(True)

# Volume trace
ax2.plot(time, lv_vol, label='LV Volume', color='blue')
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('LV Volume')
ax2.set_title('LV Volume')
ax2.legend(loc='upper right')
ax2.grid(True)

plt.tight_layout()
plt.show()
