import torch
import random
import json
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from deform import deform_grid
import numpy as np

class ToTensor(object):
  """ Convert PIL Image to Tensor """

  def __call__(self, sample):
    img = F.to_tensor(sample['image'])
    lbl = F.to_tensor(sample['label'])

    return {'image': img, 'label': lbl}

class RandomCrop(object):
  """ Randomly crop given image """

  """
  Args:
    crop_size (int): cropping size
  """

  def __init__(self, crop_size):
    self.crop_size = crop_size

  def __call__(self, sample):
    img, lbl = sample['image'], sample['label']
    x_crop, y_crop, _, _ = T.RandomCrop.get_params(img, self.crop_size)

    return {'image': F.crop(img, x_crop, y_crop, self.crop_size[0], self.crop_size[1]),\
      'label': F.crop(lbl, x_crop, y_crop, self.crop_size[0], self.crop_size[1])}

class RandomWarp(object):
  """ Randomly apply elastic deformation to PIL Image """

  """
  Args:
    p (float, optional): probability of warping
    sigma (float, optional): stddev of gaussian
  """

  def __init__(self, p=0.5, kernel_dim=23, sigma=6, alpha=30):
    self.p = 2.0
    self.alpha = alpha
    self.sigma = sigma
    self.kernel_dim = kernel_dim

  def __call__(self, sample):
    img, lbl = sample['image'], sample['label']
    
    if random.random() < self.p:
      grid = deform_grid(self.kernel_dim, self.sigma, \
        self.alpha, img.size[0])
      img = torch.nn.functional.grid_sample(T.ToTensor()\
        (img).unsqueeze(0), grid)
      lbl = torch.nn.functional.grid_sample(T.ToTensor()\
        (lbl).unsqueeze(0), grid)

    return {'image': T.ToPILImage()(img[0]), 'label':\
      T.ToPILImage()(lbl[0])}

class CenterCrop(object):
  """ Crop given image in sample """

  """
  Args:
    size (int): size of cropped image
  """

  def __init__(self, img_size, lbl_size):
    self.img_size = img_size
    self.lbl_size = lbl_size
  
  def __call__(self, sample):
    img = F.center_crop(sample['image'], self.img_size)
    lbl = F.center_crop(sample['label'], self.lbl_size)
  
    return {'image': img, 'label': lbl}

class RandomFlip(object):
  """ Randomly flip given image in sample """

  """
  Args:
    p (tuple or int): probability of flipping horizontally and vertically
  """

  def __init__(self, p=0.5):
    if isinstance(p, list):
      self.p = p
    else:
      self.p = (p, p)
  
  def __call__(self, sample):
    img, lbl = sample['image'], sample['label']

    if random.random() < self.p[0]:
      # print("Horizontal flip")
      img = F.hflip(img)
      lbl = F.hflip(lbl)

    if random.random() < self.p[1]:
      # print("Vertical flip")
      img = F.vflip(img)
      lbl = F.vflip(lbl)

    return {'image': img, 'label': lbl}

class Pad(object):
  """ Pad given image in sample """

  """
  Args:
    padding (int): Padding value
    mode (string, optional): allowed modes same as in 
      torchvision.transforms.Pad
  """

  def __init__(self, padding, mode='constant'):
    self.padding = padding
    self.mode = mode
  
  def __call__(self, sample):
    img = F.pad(sample['image'], self.padding, padding_mode=self.mode)
    lbl = F.pad(sample['label'], self.padding, padding_mode=self.mode)
    # print("Padding - size {}, mode {}".format(self.padding, self.mode))

    return {'image': img, 'label': lbl}

class RandomAffine(object):
  """ Randomly rotate and translate the image in sample """

  """
  Args:
    degrees (tuple): Range of rotation
    translate (tuple): maximum absolute horizontal and 
      vertical translations
  """

  def __init__(self, degrees, translate):
    self.degrees = degrees
    self.translate = translate

  def __call__(self, sample):
    degrees, translate, _, _ = T.RandomAffine.get_params(self.degrees, self.translate,
      None, None, [1,1])
    # print("Affine - rotate {} degrees, translate ({}, {})".format(degrees, translate[0], translate[1]))
    
    img = F.affine(sample['image'], degrees, translate, 1, 0)
    lbl = F.affine(sample['label'], degrees, translate, 1, 0)

    return {'image': img, 'label': lbl}

class Segmentation(Dataset):
  """ Segmentation dataset """

  def __init__(self,directory='../Data/train/',json_file='training.json',transform = None):
    """
    Args:
      json_file (string): path to json file which contains image metadata
      transform (callable, optional): Optional transform to be applied
        on a sample
    """

    self.json_path = directory + json_file
    self.directory = directory
    with open(self.json_path) as F:
      self.data = json.load(F)
    self.data = [datum['image'] for datum in self.data]
    self.transform = transform
  
  def __len__(self):
    return 250 #len(self.data)
  
  def __getitem__(self, idx):
    sample = {
      'image': Image.open(self.directory + 'images/' + self.data[idx]['pathname'][8:]),
      'label': Image.open(self.directory + 'labels/' + self.data[idx]['pathname'][8:]),
    }
        
    if self.transform:
      sample = self.transform(sample)
    
    if torch.cuda.is_available():
      sample['image'] = sample['image'].cuda()
      sample['label'] = sample['label'].cuda()

    return sample
  
  def get_images(self):
    return self.images
