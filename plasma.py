################### CONTENT ################################
#
# This file contains functions that can be used to compute
# various plasma properties of 3D grid data
#
# it contains
# - sound_speed: computes sound speed
# - jeans_length: computes jeans length
# - plasma_pressure: computes magnetic pressure
# - magnetic_tension: computes magnetic tension
#
############################################################

pypath = '/home/stuf317/python'

import astropy as apy
import aplpy
import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
import myfunction as myfunc
import time

############################################################
#
# sound_speed:
#   TASK: 
# INPUT: + 
# RETURNS: 
#
############################################################

def sound_speed(gam, temp, mu):
	kB = apy.constants.k_B.value
	mp = apy.constants.m_p.value
	cs = np.sqrt((gam * kB * temp) / (mu * mp) )*1e3
	return cs

############################################################
#
# jeans_length: 
#   TASK: 
# INPUT: + 
# RETURNS: 
#
############################################################

def jeans_length(temp, rho):
	kB = apy.constants.k_B.value*1e7
	mp = apy.constants.m_p.value*1e3
	grav = apy.constants.G.value
	num = 15 * kB * temp
	den = 4 * np.pi * rho * grav * mp
	jl = np.sqrt(num/den)
	return jl

############################################################
#
# plasma_pressure:
#   TASK: 
# INPUT: + 
# RETURNS: 
#
############################################################

def plasma_pressure(b):
	pB = b**2 / (8 * np.pi)
	return pB

############################################################
#
# magnetic_tension:
#   TASK: 
# INPUT: + 
# RETURNS: 
#
############################################################

def magnetic_tension(bx,by,bz, dx):
	sx, sy, sz = myfunc.my_get_array_size(bx)
	db = np.zeros([sx,sy,sz])

	dbxdx = myfunc.my_derivativ(bx, 'x', res = dx)
	dbxdy = myfunc.my_derivativ(bx, 'y', res = dx)
	dbxdz = myfunc.my_derivativ(bx, 'z', res = dx)

	dbydx = myfunc.my_derivativ(by, 'x', res = dx)
	dbydy = myfunc.my_derivativ(by, 'y', res = dx)
	dbydz = myfunc.my_derivativ(by, 'z', res = dx)

	dbzdx = myfunc.my_derivativ(bz, 'x', res = dx)
	dbzdy = myfunc.my_derivativ(bz, 'y', res = dx)
	dbzdz = myfunc.my_derivativ(bz, 'z', res = dx)

	tenx = bx * dbxdx + by * dbxdy + bz * dbxdz
	teny = bx * dbydx + by * dbydy + bz * dbydz
	tenz = bx * dbzdx + by * dbzdy + bz * dbzdz

	return tenx, teny, tenz











