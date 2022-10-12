from astropy.io import fits
import aplpy
import matplotlib.pyplot as plt
import numpy as np
import h5py as h5

############################################################
#
# my_plot_init:
# 	TASK: 
#	INPUT: + 
#	RETURNS: 
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
# 	TASK: 
#	INPUT: + 
#	RETURNS: 
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
# 	TASK: 
#	INPUT: + 
#	RETURNS: 
#
############################################################

def my_writefits(fname, data):
	hdu = fits.PrimaryHDU(data)
	hdul = fits.HDUList([hdu])
	hdul.writeto(fname + '.fits', overwrite = True)



############################################################
#
# my_contour:
# 	TASK: 
#	INPUT: + 
#	RETURNS: 
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
# 	TASK: 
#	INPUT: + 
#	RETURNS: 
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
# 	TASK: 
#	INPUT: + 
#	RETURNS: 
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
# 	TASK: 
#	INPUT: + 
#	RETURNS: 
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
	hdul.writeto('ellipsoid'+ '_' + str(scale[0]) + '_' + str(scale[1]) + '_' + str(scale[2]) + '_' + str(off[0]) + '_' + str(off[1]) + '_' + str(off[2]) +'1.fits', overwrite = True)
	return im


############################################################
#
# my_scattercolored:
# 	TASK: 
#	INPUT: + 
#	RETURNS: 
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
# 	TASK: 
#	INPUT: + 
#	RETURNS: 
#
############################################################

def my_write_numpy_arr(data, fname):
	np.save(fname, data)

###########################################################
#
# my_read_numpy_arr:
# 	TASK: 
#	INPUT: + 
#	RETURNS: 
#
############################################################

def my_read_numpy_arr(fname):
	data = np.load(fname + '.npy')
	return data

############################################################
#
# my_get_goldensize_latex:
# 	TASK: 
#	INPUT: + 
#	RETURNS: 
#
############################################################

def my_goldenratio(x):
	golden_ratio = (5 ** 0.5 - 1) / 2
	return x*golden_ratio

############################################################
#
# my_write_txt:
# 	TASK: 
#	INPUT: + 
#	RETURNS: 
#
############################################################

def my_write_txt(fname, text):
	with open(fname + '.txt', 'w') as f:
		f.write(text)

############################################################
#
# my_enzo_conv_to_time:
# 	TASK: 
#	INPUT: + 
#	RETURNS: 
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

	my_write_numpy_arr(red, path + 'redshift')
	my_write_numpy_arr(time, path + 'time')



############################################################
#
# my_red_to_time:
# 	TASK: 
#	INPUT: + 
#	RETURNS: 
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
		
	


		
		















