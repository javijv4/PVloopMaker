#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/02/22 12:46:30

@author: Javiera Jilberto Vallejos 
'''
import numpy as np
import matplotlib.pyplot as plt

ted = 0.139

t = np.linspace(0, 1, 1000)
act = 0.5*(1.-np.cos(2.*np.pi*(t)/(2*ted))) * (t >= 0.) * (t <= 2*ted)

plt.plot(t, act)
plt.vlines(ted, 0, 1, 'r', '--')