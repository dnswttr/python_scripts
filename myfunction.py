################### CONTENT ################################
#
# contains functions that are needed for multiple scripts
#
# it contains:
# -my_plot_init:
# -my_readfits:
# -my_writefits:
# -my_contour:
# -my_readhdf5:
# -my_readenzocorr:
# -my_ellipsoid:
# -my_scattercolored:
# -my_write_numpy_arr:
# -my_read_numpy_arr:
# -my_write_dat_arr:
# -my_read_dat_arr:
# -my_goldenratio:
# -my_write_txt:
# -my_enzo_conv_to_time:
# -my_red_to_time:
# -my_derivativ:
# -my_get_array_size:
#
############################################################

from astropy.io import fits
#import aplpy
import matplotlib.pyplot as plt
import numpy as np
import h5py as h5

############################################################
#
# my_plot_init:
# 	TASK: initializes plot properties as I like them to look like
#	INPUT: nothing
#	RETURNS: nothing
# FILES SAVED: none
#
############################################################

def my_plot_init():
	plt.rcParams['font.family'] = 'serif'
	plt.rcParams['font.size'] = '60'
	plt.rcParams['lines.linewidth'] = '7'
	plt.rcParams['axes.linewidth'] = '7'
	plt.rcParams['axes.linewidth'] = '7'
	plt.rcParams['axes.labelpad'] = '10'

	plt.rcParams['xtick.major.pad'] = '10'
	plt.rcParams['xtick.bottom'] = 'true'
	plt.rcParams['xtick.top'] = 'true'
	plt.rcParams['xtick.major.size'] = '20'
	plt.rcParams['xtick.major.width'] = '7'
	plt.rcParams['xtick.minor.size'] = '10'
	plt.rcParams['xtick.minor.width'] = '5'
	plt.rcParams['xtick.direction'] = 'in'
	plt.rcParams['xtick.minor.visible'] = 'true'

	plt.rcParams['ytick.major.pad'] = '10'
	plt.rcParams['ytick.left'] = 'true'
	plt.rcParams['ytick.right'] = 'true'
	plt.rcParams['ytick.major.size'] = '20'
	plt.rcParams['ytick.major.width'] = '7'
	plt.rcParams['ytick.minor.size'] = '10'
	plt.rcParams['ytick.minor.width'] = '5'
	plt.rcParams['ytick.direction'] = 'in'
	plt.rcParams['ytick.minor.visible'] = 'true'

	plt.rcParams['lines.markersize'] = '21'
	plt.rcParams["mathtext.fontset"] = "dejavuserif"

############################################################
#
# my_readfits:
# 	TASK: reads a fits file
#	INPUT: 
#  + path: full path plus filename to the fits file
#          e.g.: /path/filename.fits
#	RETURNS: 
#  - header: head of the fits files
#  - im: the data of the actual image
#  - sd: size of the image along the different axis
#  - x: numpy array with x coordinates of the image
#  - y: numpy array with y coordinates of the image
# FILES SAVED: none
#
############################################################

def my_readfits(path):
	HDU = fits.open(path)
	im = HDU[0].data
	header = HDU[0].header
	naxis = header[2]
	sd = np.zeros(naxis)
	for i in range(naxis):
		sd[i] = np.size(im,i)
	if naxis == 2: x = np.arange(sd[0])
	if naxis == 2: y = np.arange(sd[1])
	if naxis == 3: x = np.arange(sd[1])
	if naxis == 3: y = np.arange(sd[2])
	return header, im, sd, x, y

############################################################
#
# my_writefits:
# 	TASK: writes a fits file
#	INPUT: 
#  + fname: full path plus filename without the ending
#           ".fits" will be added by the function
#           e.g.: /path/filename
#  + data: the data that is to be written to file
#	RETURNS: nothing
# FILES SAVED:
#  ~ filename.fits: the fits file at /path/filename.fits
#
############################################################

def my_writefits(fname, data):
	hdu = fits.PrimaryHDU(data)
	hdul = fits.HDUList([hdu])
	hdul.writeto(fname + '.fits', overwrite = True)

############################################################
#
# my_contour:
# 	TASK: produce a filled contour plot of an image
#	INPUT:
#  + x: x coordinates of the image
#  + y: y coordinates of the image
#  + im: image data
#  + nlev (optional): number of contours, default 256
#  + tit (optional): title of the image, default ''
#  + xt (optional): xlabel of the image, default ''
#  + yt (optional): ylabel of the image, default ''
#  + ct (optional): label of the colorbar, default ''
#  + cm (optional): colormap, default 'magma'
#  + ocor (optional): position of the colorbar, default 'v'
#                     - 'v' vertical
#                     - 'h' horizontal
#	RETURNS: nothing
# FILES SAVED: none
#
############################################################

def my_contour(x,y,im, nlev = 256, tit = '',  xt = '', yt = '', ct = '', cm = 'magma', ocor = 'v'):
	print(np.size(x))
	print(np.size(y))
	print(np.size(im))
	print(np.size(im,0))
	print(np.size(im,1))

	plt.contourf(y,x,im,nlev, cmap = cm)
	plt.xlabel(xt)
	plt.ylabel(yt)
	plt.title(tit, pad = 20)
	if ocor == 'v': cb = plt.colorbar(label=ct, orientation="vertical", pad = 0.01)
	if ocor == 'h': cb = plt.colorbar(label=ct, orientation="horizontal", pad = 0.01)
	cb.set_label(ct, labelpad = 30)

############################################################
#
# my_readhdf5:
# 	TASK: reads a hdf5 file
#	INPUT:
#  + pathf: full path plus filename
#           e.g.: /path/file.hdf5
#  + prop: property to be read
#  + cube (optional): set, if only part of the file should be read
#                     cube is a array with 6 entries that are the 
#                     start and end coordinates of the data chunk
#                     ordered as [xm, xp, ym, yp, zm, zp]
#                     e.g. [2,3,4,5,6,7] will read a 2^3 array that 
#                     contains x coordinates 2 to 3, y coordinates 
#                     4 to 5, and z coordinates 6 to 7. Here, "to"
#                     means including
#                     default not set
#  + conv (optional): if the data cube is to be multiplied with some
#                     conversion factor (e.g. useful for enzo data)
#                     default 1
#     
#	RETURNS: 
#  - c: the data that was read
# FILES SAVED: none
#
############################################################

def my_readhdf5(pathf, prop, cube = [-1], conv = 1):
	data = h5.File(pathf, "r")
	d = data[prop]
	if cube[0] == -1:
		c = d[:,:,:] * conv
	else:
		c = d[cube[0]:cube[1]+1, cube[2]:cube[3]+1, cube[4]:cube[5]+1] * conv
	return c

############################################################
#
# my_readenzocorr:
# 	TASK: read the enzo corrections files, e.g. *conv2 files
#	INPUT:
#  + pathf: full path plus filename
#           e.g. /path/file.conv2
#	RETURNS: red, dens, vel, mag
#  - red: redshift
#  - dens: factor to convert density units to g/cm^3
#  - vel: factor to convert velocity to cm/s
#  - mag; factor to convert magnetic field to G
# FILES SAVED: none
#
############################################################

def my_readenzocorr(pathf):
	file1 = open(pathf, "r")
	val = np.zeros(5)
	for i in range(5):
		val[i] = float(file1.readline())
	red = val[2]
	dens = val[3]
	vel = val[4]
	mag = np.sqrt(4 * np.pi * dens) * vel * (1+red)**2
	return red, dens, vel, mag

############################################################
#
# my_ellipsoid:
# 	TASK: function, that creates a fits file that contains a 3D ellipsoid
#         using 
#	INPUT: 
#  + mesh: 3D numpy array, that contains the mesh coordinates (x,y,z)
#  + scale: scale of the radius, with (x-scale, y-scale, z-scale)
#           e.g.: scale = [1, 0.5, 2] means that the ellipsoid
#                 is not stretched in x direction
#                 is stretched by 0.5 in y direction
#                 is stretched by 2 in z direction
#                 scale = [1,1,1] is a unit sphere
#  + off: offset, if the ellipsoids center is not the box center (x-offset, y-offset, z-offset)
#	RETURNS: 
#  - im: 3D boolean array, that contains the ellipsoid, where
#        cells with "True" belong to the ellipsoid
#        cells with "False" do not belong to the ellipsoid
#        im is saved in "ellipsoid*.fits"
# FILES SAVED:
#  ~ 'ellipsoid'+ '_' + str(scale[0]) + '_' + str(scale[1]) + '_' + str(scale[2]) + '_' + str(off[0]) + '_' + str(off[1]) + '_' + str(off[2]) +'.fits:
#    fits file of the ellipsoid in the folder where the function was exectured
#    the filename has the scale and offset encoded, please see the source code
#
############################################################

def my_ellipsoid(mesh, scale, off):
	sx = np.size(mesh,1)
	sy = np.size(mesh,2)
	sz = np.size(mesh,3)
	im = np.zeros([sx,sy,sz], dtype = bool)
	x = mesh[0]
	y = mesh[1]
	z = mesh[2]
	r = ((x-off[0])/scale[0])**2 + ((y-off[1])/scale[1])**2 + ((z-off[2])/scale[2])**2
	idT = np.where(r <= 1.)
	im[idT] = True

	im2 = np.zeros([sx,sy,sz])
	im2[idT] = 1.
	hdu = fits.PrimaryHDU(im2)
	hdul = fits.HDUList([hdu])
	hdul.writeto('ellipsoid'+ '_' + str(scale[0]) + '_' + str(scale[1]) + '_' + str(scale[2]) + '_' + str(off[0]) + '_' + str(off[1]) + '_' + str(off[2]) +'.fits', overwrite = True)
	return im


############################################################
#
# my_scattercolored:
# 	TASK: do a colored scatter plot, with data x vs y and color col
#	INPUT:
#  + x: x coordinates of the image
#  + y: y coordinates of the image
#  + col: color coding for the plots
#  + tit (optional): title of the image, default ''
#  + xt (optional): xlabel of the image, default ''
#  + yt (optional): ylabel of the image, default ''
#  + ctit (optional): label of the colorbar, default ''
#  + ocor (optional): position of the colorbar, default 'v'
#                     - 'v' vertical
#                     - 'h' horizontal
#	RETURNS: nothing
# FILES SAVED: none
#
############################################################

def my_scattercolored(x,y,col, tit = '', xtit = '', ytit = '', ctit = '', cori = 'v'):
	xmin = np.min(x)
	xmax = np.max(x)
	ymin = np.min(y)
	ymax = np.max(y)
	plt.plot([xmin,xmax],[0,0], 'r')
	plt.plot([0,0],[ymin,ymax], 'r')
	plt.scatter(x,y, marker = ".", linestyle = 'None', c = col, cmap = 'jet')
	plt.xlim([xmin,xmax])
	plt.ylim([ymin,ymax])
	plt.title(tit, pad = 20)
	plt.xlabel(xtit)
	plt.ylabel(ytit)
	if cori == 'v': cb = plt.colorbar(label=ctit, orientation="vertical", pad = 0.01)
	if cori == 'h': cb = plt.colorbar(label=ctit, orientation="horizontal", pad = 0.01)
	cb.set_label(ctit, labelpad = 30)

###########################################################
#
# my_write_numpy_arr:
# 	TASK: saves data as "npy" (numpy array)
#	INPUT:
#  + data: the data to be saved
#  + fname: full path plus filename without ".npy"
#					  e.g.: /path/filename
#	RETURNS: nothing
# FILES SAVED:
#  ~ filename.npy: the saved numpy array at /path/filename.npy
#
############################################################

def my_write_numpy_arr(data, fname):
	np.save(fname, data)

###########################################################
#
# my_read_numpy_arr:
# 	TASK: reads "npy" array (numpy array)
#	INPUT: 
#  + fname: full path plus filename without ".npy"
#					  e.g.: /path/filename
#	RETURNS: 
#  - data: the data that was read
# FILES SAVED: none
#
############################################################

def my_read_numpy_arr(fname):
	data = np.load(fname + '.npy')
	return data

###########################################################
#
# my_write_dat_arr:
# 	TASK: saves data as "dat" array (formatred data)
#	INPUT: 
#  + data: the data to be saved
#  + fname: full path plus filename without ".dat"
#					  e.g.: /path/filename
#	RETURNS: nothing
# FILES SAVED:
#  ~ filename.dat: the saved file at /path/filename.dat
#
############################################################

def my_write_dat_arr(data, fname):
	np.savetxt(fname + '.dat', data)

###########################################################
#
# my_read_dat_arr:
# 	TASK: reads "dat" array (formatted array)
#	INPUT: 
#  + fname: full path plus filename without ".dat"
#					  e.g.: /path/filename
#	RETURNS: 
#  - data: the data that was read
# FILES SAVED: none
#
############################################################

def my_read_dat_arr(fname):
	data = np.loadtxt(fname + '.dat')
	return data

############################################################
#
# my_get_goldensize_latex:
# 	TASK: computes the golden ratio for an length x
#         in principle, this can be used to compute "perfect"
#         dimensions for plots
#	INPUT: 
#  + x: the length x
#	RETURNS: 
#  - x*goldenratio: the golden ratio of x
# FILES SAVED: none
#
############################################################

def my_goldenratio(x):
	golden_ratio = (5 ** 0.5 - 1) / 2
	return x*golden_ratio

############################################################
#
# my_write_txt:
# 	TASK: writes text into a text file
#	INPUT: 
#  + fname: full path plus filename without ".txt"
#           e.g. /path/filename
#	RETURNS: 
#  - text: the text to be written into the file
# FILES SAVED: 
#  ~ filename.txt: the saved text at /path/filename.txt
#
############################################################

def my_write_txt(fname, text):
	with open(fname + '.txt', 'w') as f:
		f.write(text)

############################################################
#
# my_enzo_conv_to_time:
# 	TASK: computes the time and redshift of the different enzo snapshots
#	INPUT:
#  + path: path to the enzo files
#  + fname: name of the *conv2 files, but only the static part
#           e.g.: for sim_cosmDD0001, sim_cosmDD0002, sim_cosmDD0003
#                 fname = 'sim_cosmDD0'
#  + snap0: first snapshot to be read
#           e.g. 1 for the example above
#  + snap1: last snapshot to be read
#           e.g. 3 for the example above
#  + h0: Hubbel Constant, i.e. H0 in (km/s)/Mpc
#        if small h0 is given, it is converted to H0
#  + ome_m: Omega_m, density parameter 
#  + ome_l: Omega_Lambda, vacuum density today
#  + unit (optional): set to get specific time units, default 'Gyr'
#                     - 'Gyr': Gigayears
#                     - 'Myr': Megayears
#                     - 'sec': seconds
#	RETURNS: nothing
# FILES SAVED: 
#  ~ redshift.dat: the redshifts saved at /path/redshift.dat
#  ~ time.dat: the times saved at /path/time.dat
#
############################################################

def my_enzo_conv_to_time(path, fname, snap0, snap1, h0, ome_m, ome_l, unit = 'Gyr'):
	if h0 < 40:
		print('h0 is smaller than 40 \n')
		print('h0 = ', str(h0))
		print('I convert it to H0')
		h0 *= 100
		print('H0 = ', str(h0))

	nbin = snap1-snap0
	red = np.zeros(nbin)
	time = np.zeros(nbin)
	for ds in range(nbin):
		snap = snap0 + ds

		pathf = path + fname + str(snap).zfill(3) + '.conv2'
		cr, cd, cv, cb =  my_readenzocorr(pathf)

		red[ds] = cr
		ct = my_red_to_time(ome_m, ome_l, h0, cr, unit = unit)
		time[ds] = ct

	my_write_dat_arr(red, path + 'redshift')
	my_write_dat_arr(time, path + 'time')



############################################################
#
# my_red_to_time:
# 	TASK: converts a redshift into time
#	INPUT: 
#  + ome_m: Omega_m, density parameter 
#  + ome_l: Omega_Lambda, vacuum density today
#  + h0: Hubbel Constant, i.e. H0 in (km/s)/Mpc
#        if small h0 is given, it is converted to H0
#  + cr: the redshift
#  + unit (optional): set to get specific time units, default 'Gyr'
#                     - 'Gyr': Gigayears
#                     - 'Myr': Megayears
#                     - 'sec': seconds
#	RETURNS: 
#  - time: the time
# FILES SAVED: none
#
############################################################

def my_red_to_time(ome_m, ome_l, h0, cr, unit = 'Gyr'):
	km_to_Mpc = 3.2408e-20
	h0=h0*km_to_Mpc

	fac1 = 2/h0
	fac2 = 3*np.sqrt(ome_l)
	asin1 = np.sqrt(ome_l / ome_m)
	asin2 = (cr + 1)**(-1.5)
	fac3 = np.arcsinh(asin1*asin2)
	time = fac1 / fac2 * fac3

	if unit == 'Gyr':
		print('Convert time to Gyr')
		time = time / (31536000 * 1e9)
	elif unit == 'Myr':
		print('Convert time to Myr')
		time = time / (31536000 * 1e6)
	elif unit == 'sec':
		print('Leave time as sec')
	else:
		print('Nothing was given so, I convert time to Gyr')
		time = time / (31536000 * 1e9)

	return time
		
############################################################
#
# my_derivative:
# 	TASK: compute derivative of an ND array
#	INPUT: 
#  + arr: array with the ND data
#  + axis: axis, along which the derivative is compute
#          it is either 'x', 'y' or 'z'
#  + res (optional): resolution that can be set, if the physical resolution 
#                    of the cell matters
#	RETURNS: 
#  - darr: array that contains the derivative
# FILES SAVED: none
#
############################################################

def my_derivativ(arr, axis, res = 1):
	if axis not in ['x', 'y', 'z']:
		print('Error in my_derivativ (my_function.py): \n axis that was given does not exist')
		exit()

	ndim = np.ndim(arr)
	if ndim == 0:
		print('Error in my_derivativ (my_function.py): \n array dimension is 0')
		exit()

	if axis == 'x':
		darr = (np.roll(arr, -1, axis = 0) - np.roll(arr, 1, axis = 0))/(2*res)
	if axis == 'y':
		if ndim < 2:
			print('Error in my_derivativ (my_function.py): \n array dimension is smaller than 2 but y axis was chosen')
			exit()
		darr = (np.roll(arr, -1, axis = 0) - np.roll(arr, 1, axis = 1))/(2*res)
	if axis == 'z':
		if ndim < 3:
			print('Error in my_derivativ (my_function.py): \n array dimension is smaller than 3 but z axis was chosen')
			exit()
		darr = (np.roll(arr, -1, axis = 0) - np.roll(arr, 1, axis = 2))/(2*res)

	return darr


############################################################
#
# my_get_array_size:
# 	TASK: computes the size of an ND array
#	INPUT: 
#  + arr: array with the ND data
#	RETURNS: 
#  - sarr: ND array, that has the size of each dimension stored
#          for a single number sarr = 0
# FILES SAVED: none
#
############################################################

def my_get_array_size(arr):
	ndim = np.ndim(arr)
	if ndim == 0:
		sarr = 0
	else:
		sarr = np.zeros(ndim)
		for i in range(ndim):
			sarr[i] = np.size(arr,i)
	return sarr
		
