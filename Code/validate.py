import os
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torchvision
from model import UNet
from dataset import Segmentation, RandomAffine, Pad, RandomFlip, CenterCrop, ToTensor, RandomWarp, RandomCrop

from torchvision import transforms as T
from PIL import Image
import numpy as np

# Neural network
model = UNet(n_class = 1).cuda() if torch.cuda.is_available() else UNet(n_class = 1)

def get_checkpoint(model, optimizer, loss, filename):
  if filename is None:
    filename = "unet.pth"
  map_location = 'cuda:0' if torch.cuda.is_available() else 'cpu'
  try:
    checkpoint = torch.load(filename, map_location=map_location)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    loss.extend(checkpoint['loss_log'])
    print("Loaded saved model")
  except FileNotFoundError:
    print("Unable to find saved model. Continuing with new model")
  except RuntimeError:
    print("Unable to load saved model. Continuing with new model")

def get_dataset(directory, img_size, data_size=None):
  """
  Unlike the train file, this returns the intact image and label regardless
  of the img_size option. It is the work of the main function to crop the
  image and send it as input to the model and stitch back the output
  """
  return Segmentation(directory, 'training.json', \
    transform = ToTensor(), data_size=data_size)

def apply_model(model, n_class, img_batch, img_size):
  """
  Apply the model to the image in broken tiles according to img_size and stitch
  the outputs back together
  """
  if img_size is None:
    return model(img_batch)

  size = img_batch.size()
  x_iter = size[2]//img_size+1 if size[2]%img_size != 0 else size[2]//img_size
  y_iter = size[3]//img_size+1 if size[3]%img_size != 0 else size[3]//img_size
  padded_output = torch.zeros(size[0], n_class, x_iter*img_size, y_iter*img_size)
  output = torch.zeros(size[0], n_class, size[2], size[3])

  for index, img in enumerate(img_batch):
    img = T.functional.pad(T.ToPILImage()(img), (0, 0, y_iter*img_size - size[3], \
      x_iter*img_size - size[2]), padding_mode='reflect')
    for i in range(x_iter):
      for j in range(y_iter):
        input_img = T.ToTensor()(T.functional.crop(img, i*img_size, \
          j*img_size, img_size, img_size))
        output_img = model(input_img.unsqueeze(0))[0]
        print(output_img.size(), padded_output[index,0:n_class,i*img_size:(i+1)*img_size,j*img_size:(j+1)*img_size].size())
        padded_output[index,0:n_class,i*img_size:(i+1)*img_size,j*img_size:(j+1)*img_size] = output_img
  
  for index, padded_img in enumerate(padded_output):
    for channel, layer in enumerate(padded_img):
      output[index, channel, :, :] = T.ToTensor()(T.functional.crop(T.ToPILImage()(layer.unsqueeze(0)), 0, 0, size[2], size[3]))[0]

  return output

def validate(lr=0.001, n_class=1, in_channel=1, loss_fn='BCE', display=False, \
directory='../Data/train/', img_size=None, data_size=None, load_file=None):
  # Dataset
  dataset = get_dataset(directory, img_size, data_size)

  # Neural network model
  model = UNet(n_class, in_channel).cuda() if torch.cuda.is_available() else UNet(n_class, in_channel)

  optimizer = torch.optim.Adam(model.parameters(),lr = lr)
  loss_log = []

  get_checkpoint(model, optimizer, loss_log, load_file)

  criterion = torch.nn.BCELoss()
  if loss_fn == 'CE':
    weights = torch.Tensor([10,90])
    if torch.cuda.is_available():
      weights = weights.cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

  testloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
  dataiter = iter(testloader)

  while True:
    try:
      testimg = dataiter.next()
      img, lbl = testimg['image'], testimg['label']
      trained = apply_model(model, n_class, img, img_size)
      if loss_fn == 'CE':
        output = torch.argmax(trained, 1)
        if display:
          T.ToPILImage()(img[0]).show()
          T.ToPILImage()(lbl.squeeze(0)).show()
          T.ToPILImage()(trained[0][0].unsqueeze(0).float()).show()
          T.ToPILImage()(trained[0][1].unsqueeze(0).float()).show()
          T.ToPILImage()(output.float()).show()


      elif loss_fn == 'BCE':
        output = (trained > torch.tensor([0.6]))
        if display:
          T.ToPILImage()(img[0]).show()
          T.ToPILImage()(lbl.float()).show()
          T.ToPILImage()((trained[0]).float()).show()
          T.ToPILImage()((output[0]).float()).show()

      TP = ((output[0].long() == lbl.long()) & (output[0].long() == 1)).sum()
      TN = ((output[0].long() == lbl.long()) & (output[0].long() == 0)).sum()
      FP = ((output[0].long() != lbl.long()) & (output[0].long() == 1)).sum()
      FN = ((output[0].long() != lbl.long()) & (output[0].long() == 0)).sum()
      matching = (output[0].long() == lbl.long()).sum()
      accuracy = float(matching) / lbl.numel()
#      print("matching {}, total {}, accuracy {}".format(matching, lbl.numel(), accuracy))
      print("IoU: {}".format(TP/(TP+FP+FN)))
      try:
        precision = float(TP.float()/(TP.float()+FP.float()))
        recall = float(TP.float()/(TP.float()+FN.float()))
        F1 = float(2*precision*recall/(precision + recall))
        print("({},{},{},{}) accuracy {}, precision {}, recall {}, F1 score {}".format(TP,TN,FP,FN,accuracy, precision, recall, F1))
      except FloatingPointError:
        continue
      except ZeroDivisionError:
        print(TP,TN,FP,FN)
        continue
    except StopIteration:
      break
