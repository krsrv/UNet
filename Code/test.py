import torch
from torchvision import transforms as T
from PIL import Image
from dataset import Segmentation, RandomFlip, Pad, RandomAffine, CenterCrop, ToTensor, RandomWarp, RandomCrop
from model import UNet
from deform import deform_grid

t = T.Compose([
#  RandomCrop((512,512)),
#  RandomWarp(),
  ToTensor()
])
transform = T.Compose([ \
      Pad(150, mode='symmetric'), \
      RandomAffine((0, 90), (31, 31)), \
      RandomFlip(), \
      RandomWarp(),
      CenterCrop(512, 512), \
      ToTensor()
    ])
 
dataset = Segmentation('../Data/train/training.json',t)
original = Segmentation('../Data/train/training.json',ToTensor())

pil = T.ToPILImage()

def label(sample):
  pil(sample['label'].float()).show()
  return sample['label']

def img(sample):
  pil(sample['image']).show()
  return sample['image']

# model = UNet(n_class=2)
