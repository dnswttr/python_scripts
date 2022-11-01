from scipy.ndimage import binary_fill_holes
from scipy.ndimage import gaussian_filter
from scipy.ndimage import binary_dilation
from scipy.ndimage import binary_erosion
from scipy.ndimage import generate_binary_structure
import numpy as np

st1 = generate_binary_structure(3, 2)




x = np.zeros([5,5,5], dtype = int)
x[2,1,2] = 1
x[2,3,2] = 1
print(x)
print('\n')

print(binary_dilation(x, structure=st1).astype(int))
print('\n')
im = binary_erosion(binary_dilation(x, structure=st1).astype(int)).astype(int)

print('\n')
print(im)



