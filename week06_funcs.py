# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 16:09:32 2023

@author: ywang11
"""

import numpy as np
from typing import Callable
import scipy.integrate

def alpha_fn_K_n(u: float) -> float:
    """
    alpha function for Potassium n-gate
    """
    return 0.02 * (u - 25) / (1 - np.exp(-(u-25)/9))

def alpha_fn_Na_m(u: float) -> float:
    """
    alpha function for Sodium m-gate
    """
    return 0.182 * (u + 35) / (1 - np.exp(-(u+35)/9))

def alpha_fn_Na_h(u: float) -> float:
    """
    alpha function for Sodium h-gate
    """
    return 0.25 * np.exp(-((u+90)/12))

def beta_fn_K_n(u: float) -> float:
    """
    beta function for Potassium n-gate
    """
    return -0.002 * (u - 25) / (1 - np.exp((u-25)/9))

def beta_fn_Na_m(u: float) -> float:
    """
    beta function for Sodium m-gate
    """
    return -0.124 * (u + 35) / (1 - np.exp((u+35)/9))

def beta_fn_Na_h(u: float) -> float:
    """
    beta function for Sodium h-gate
    """
    return (0.25 * np.exp((u+62)/6)) / (np.exp((u+90)/12))
 

def dx_dt(x: float, u: float, alpha_fn: Callable, beta_fn: Callable) -> float:
    """
    Generates generic dx_dt for classical HH formulation.
    """
    tau = 1. / (alpha_fn(u) + beta_fn(u))
    x0 = tau * alpha_fn(u)
    return - (x - x0) / (tau) 
    

def hh_model(
    t: float, y: np.ndarray, g_leak: float, E_leak: float, 
    g_K: float, E_K: float, g_Na: float, E_Na: float, C:float,
    I_ext: float
) -> np.ndarray:
    """
    Classical Hodgkin-Huxley Model ODE system.

    Returns u (voltage), n (K-gate state), m and h (Na-gate states).
    y[0] = u; y[1] = n; y[2] = m; y[3] = h
    """
    
    u = y[0]
    n = y[1]
    m = y[2]
    h = y[3]
    
    I_leak = g_leak * (u - E_leak)
    I_K = g_K * n ** 4 * (u - E_K)
    I_Na = g_Na * m ** 3 * h * (u - E_Na)

    du_dt = (-1. / C) * np.sum([I_K, I_Na, I_leak]) + I_ext(t)
    dn_dt = dx_dt(n, u, alpha_fn_K_n, beta_fn_K_n)
    dm_dt = dx_dt(m, u, alpha_fn_Na_m, beta_fn_Na_m)
    dh_dt = dx_dt(h, u, alpha_fn_Na_h, beta_fn_Na_h)
    
    return [du_dt, dn_dt, dm_dt, dh_dt]

def current_injector(t: float, amp: float=10) -> float:
    """
    function that takes in the current time in milliseconds as t 
    and returns 10 uA when the time is between 5.0 and 15.0 ms.
    """
    llimit = 5.
    ulimit = 15.
    if  llimit <= t and t <= ulimit:
        return amp
    else: 
        return  0.
def solve ():
        
    hh_g_leak = 0.3
    hh_g_K = 35
    hh_g_Na = 40
    hh_E_leak = -65
    hh_E_K = -77
    hh_E_Na = 55
    hh_C = 1

    hh_soln = scipy.integrate.solve_ivp(method = 'BDF',
        fun=hh_model,
        t_span=(0, 20),
        y0=(-65, 0.1, 0.01, 0.9,),
        args=(hh_g_leak,  hh_E_leak, hh_g_K, hh_E_K, hh_g_Na, hh_E_Na, hh_C, 
        current_injector)
    )
    return hh_soln
