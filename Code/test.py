import torch
from torchvision import transforms as T
from PIL import Image
from dataset import Segmentation, RandomFlip, Pad, RandomAffine, CenterCrop, ToTensor, RandomWarp, RandomCrop
from model import UNet
from deform import deform_grid

dataset = Segmentation('../Data/train/','training.json',ToTensor())
original = Segmentation('../Data/train/','training.json',ToTensor())

model = UNet(n_class=1, in_channel=3)

def print_help():
  print("""Invoke test() with following arguments to properly initialise the dataset and models:
n_class   : number of output classes
in_channel: number of input channels
load      : whether to load saved model
img_size  : size of image, if it is to be cropped
directory : directory which stores image metadata json file""")

def test(n_class=1, in_channel=1, load=False, img_size=None, directory='../Data/train/'):
  global original, dataset, model
  original = Segmentation(directory, 'training.json', ToTensor())
  if img_size is None:
    dataset = Segmentation(directory, 'test.json', ToTensor())
  else:
    dataset = Segmentation(directory, 'test.json', T.Compose([\
      RandomCrop((img_size, img_size)), ToTensor()]))

  model = UNet(n_class=n_class, in_channel=in_channel)
  if load:
    filename = "unet.pth"
    map_location = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    try:
      checkpoint = torch.load(filename, map_location=map_location)
      model.load_state_dict(checkpoint['state_dict'])
      print("Loaded saved model")
    except:
      print("Unable to load saved model")
    

pil = T.ToPILImage()

def label(sample):
  pil(sample['label'].float()).show()
  return sample['label']

def img(sample):
  pil(sample['image']).show()
  return sample['image']

def run(sample):
  output = model(sample['image'].unsqueeze(0))
  pil(output.squeeze(0)).show()

