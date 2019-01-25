import os
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision.transforms import Compose
from model import UNet
from dataset import Segmentation, RandomAffine, Pad, RandomFlip, CenterCrop, ToTensor, RandomWarp, RandomCrop

from torchvision import transforms as T
from PIL import Image
import numpy as np

from random import random

def save_checkpoint(checkpt, filename):
  if filename is None:
    torch.save(checkpt, "unet.pth")
  else:
    torch.save(checkpt,filename)

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
  if img_size == None:
    return Segmentation(directory, 'training.json', \
    transform = ToTensor(), data_size=data_size)
  else:
    return Segmentation(directory, 'training.json', \
    transform = Compose([ \
      RandomCrop((img_size, img_size)), \
      ToTensor()
    ]), data_size=data_size)

def train(epochs=10, lr=0.001, n_class=1, in_channel=1, loss_fn='BCE', display=False, save=False, \
  load=False, directory='../Data/train/', img_size=None, data_size=None, load_file=None, save_file=None):
    #if torch.cuda.is_available():
    #  torch.cuda.set_device(1)
    # Dataset
  dataset = get_dataset(directory, img_size, data_size)
  
  #optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = momentum, weight_decay = decay)
  print("Epochs:\t{}\nLearning Rate:\t{}\nOutput classes:\t{}\nInput channels:\t{}\n\
Loss function:\t{}\nImage cropping size:\t{}\nDataset size:\t{}\n".format(epochs, lr, n_class, \
in_channel, loss_fn, img_size, data_size))

  # Neural network model
  model = UNet(n_class, in_channel).cuda() if torch.cuda.is_available() else UNet(n_class, in_channel)

  # Optimizer
  optimizer = torch.optim.Adam(model.parameters(),lr = lr)
  loss_log = []

  if load:
    get_checkpoint(model, optimizer, loss_log, load_file)

  criterion = torch.nn.BCELoss()
  if loss_fn == 'CE':
    weights = torch.Tensor([10,90])
    if torch.cuda.is_available():
      weights = weights.cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

  for epoch in range(epochs):
    #print("Starting Epoch #{}".format(epoch))

    train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    epoch_loss = 0

    for i,images in enumerate(train_loader):
      # get the inputs
      image, label = images['image'], images['label']
      
      # zero the parameter gradients
      optimizer.zero_grad()
      
      ## Run the forward pass
      outputs = model.forward(image).cuda() if torch.cuda.is_available() else model.forward(image)
     
      if display:
        T.ToPILImage()(outputs[0].float()).show()

      if loss_fn == 'CE':
        label = label.squeeze(1).long()
      elif loss_fn == 'BCE':
        label = label.float()

      loss = criterion(outputs, label)
      loss.backward()
      
      epoch_loss = epoch_loss + loss.item()
      
      optimizer.step()

      #if i % 10 == 0 :
      #  print("Epoch #{} Batch #{} Loss: {}".format(epoch,i,loss.item()))
    loss_log.append(epoch_loss)
    
    #print("Epoch",epoch," finished. Loss :",loss.item())
    print(epoch,loss.item())
    epoch_loss = 0
  if save:
    save_checkpoint({'state_dict':model.state_dict(),
				          'optimizer':optimizer.state_dict(),
													'loss_log':loss_log,
													}, save_file)
  print(loss_log)
  #T.ToPILImage()(outputs[0].float()).show()

  if display:
    testloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    dataiter = iter(testloader)

    testimg = dataiter.next()
    img, lbl = testimg['image'], testimg['label']
    trained = model(img)
    thresholded = (trained > torch.tensor([0.5]))
    T.ToPILImage()(img[0]).show()
    T.ToPILImage()(lbl.float()).show()
    T.ToPILImage()((trained[0]).float()).show()
    T.ToPILImage()((thresholded[0]).float()).show()

    matching = (thresholded[0].long() == lbl.long()).sum()
    accuracy = float(matching) / lbl.numel()
    print("matching {}, total {}, accuracy {}".format(matching, lbl.numel(),\
    accuracy))
