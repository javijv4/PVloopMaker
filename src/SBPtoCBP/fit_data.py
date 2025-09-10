#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/07/17 20:48:17

@author: Javiera Jilberto Vallejos 
'''

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

men_data = np.loadtxt('men_data.csv', delimiter=',', skiprows=1)
women_data = np.loadtxt('women_data.csv', delimiter=',', skiprows=1)

men_aortaBP, men_brachialBP = men_data.T
men_slope, men_intercept, men_r_value, men_p_value, men_std_err = linregress(men_brachialBP, men_aortaBP)
print(f"Men, Slope: {men_slope}, Intercept: {men_intercept}, R^2: {men_r_value**2}")

women_aortaBP, women_brachialBP = women_data.T
women_slope, women_intercept, women_r_value, women_p_value, women_std_err = linregress(women_brachialBP, women_aortaBP)
print(f"Women, Slope: {women_slope}, Intercept: {women_intercept}, R^2: {women_r_value**2}")


fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True, sharex=True)

# Determine common axis limits
x_min = min(women_brachialBP.min(), men_brachialBP.min())
x_max = max(women_brachialBP.max(), men_brachialBP.max())
y_min = min(women_aortaBP.min(), men_aortaBP.min())
y_max = max(women_aortaBP.max(), men_aortaBP.max())
axis_min = min(x_min, y_min)
axis_max = max(x_max, y_max)

# Women subplot
axs[0].scatter(women_brachialBP, women_aortaBP, color='r', label='Women Data')
x_vals_women = np.linspace(axis_min, axis_max, 100)
y_vals_women = women_slope * x_vals_women + women_intercept
axs[0].plot(x_vals_women, y_vals_women, color='k', label='Women Linear Regression')
axs[0].plot(x_vals_women, x_vals_women, color='b', linestyle='--', label='1:1 Line')  # 1:1 line
axs[0].set_xlabel('Brachial BP')
axs[0].set_ylabel('Aortic BP')
axs[0].set_title('Women')
axs[0].legend()
axs[0].grid(True)
axs[0].set_xlim(axis_min, axis_max)
axs[0].set_ylim(axis_min, axis_max)
axs[0].set_aspect('equal')

# Men subplot
axs[1].scatter(men_brachialBP, men_aortaBP, color='r', label='Men Data')
x_vals_men = np.linspace(axis_min, axis_max, 100)
y_vals_men = men_slope * x_vals_men + men_intercept
axs[1].plot(x_vals_men, y_vals_men, color='k', label='Men Linear Regression')
axs[1].plot(x_vals_men, x_vals_men, color='b', linestyle='--', label='1:1 Line')  # 1:1 line
axs[1].set_xlabel('Brachial BP')
axs[1].set_ylabel('Aortic BP')
axs[1].set_title('Men')
axs[1].legend()
axs[1].grid(True)
axs[1].set_xlim(axis_min, axis_max)
axs[1].set_ylim(axis_min, axis_max)
axs[1].set_aspect('equal')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()
