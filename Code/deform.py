import torch
import numpy as np

def create_kernel(kernel_size=23, sigma=9):
	""" Create a 2D Gaussian kernel """

	"""
	Args
		kernel_size (int): size of kernel
		sigma (int): stddev of Gaussian
	"""
	# create 2D gaussian kernel
	kernel = np.linspace(-kernel_size/3, kernel_size/3, kernel_size)
	kernel = np.exp(-kernel**2/(2*sigma**2))
	kernel /= np.sum(kernel)
	kernel = kernel[:, np.newaxis] * kernel[np.newaxis, :]

	return torch.tensor(kernel).float()

def create_deform_flow(size, kernel, alpha=30):
	""" Create a flow field for torch.nn.functional.grid_sample """

	"""
	Args
		size (int): size of grid
		kernel (tensor): gaussian kernel to be applied
		alpha (float)
	"""
	def displacement_vectors(size, kernel, alpha):
		# initialise displacement vectors
		dx = torch.empty(size,size).uniform_(-1,1) * alpha
		dy = torch.empty(size,size).uniform_(-1,1) * alpha

		kernel_dim = kernel.size()[0]

		# pad with zeros along both dimensions
		dx = torch.nn.functional.pad(dx, [kernel_dim//2]*4)
		dy = torch.nn.functional.pad(dy, [kernel_dim//2]*4)

		# convolve with kernel
		dx = torch.nn.functional.conv2d(dx.unsqueeze(0).unsqueeze(0),\
			kernel.reshape(1,1,kernel_dim,kernel_dim))
		dy = torch.nn.functional.conv2d(dy.unsqueeze(0).unsqueeze(0),\
			kernel.reshape(1,1,kernel_dim,kernel_dim))

		dx = dx + torch.cat([torch.tensor(np.linspace(0,size-1,size)).\
			float().reshape(-1,1)]*size,dim=1).reshape(1,1,size,size)
		dy = dy + torch.cat([torch.tensor(np.linspace(0,size-1,size)).\
			float().reshape(1,-1)]*size).reshape(1,1,size,size)

		dx = torch.clamp(2*dx/size-1,-1,1)
		dy = torch.clamp(2*dy/size-1,-1,1)

		return dy, dx
	
	dx, dy = displacement_vectors(size, kernel, alpha)
	grid = torch.cat([dx.reshape(-1,1),dy.reshape(-1,1)],dim=1)
	grid = grid.reshape(1,size,size,2)

	return grid

def deform_grid(kernel_dim=23, sigma=6, alpha=30, img_size=320):
	kernel = create_kernel(kernel_dim, sigma)
	grid = create_deform_flow(img_size, kernel, alpha)

	return grid

#grid = deform_grid()
#grid_sample(img, grid, padding_mode='reflection')
