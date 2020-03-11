# -*- coding: utf-8 -*-
"""MLPImageDenoising.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VTuzDItMhFQ4ih8vrdDeWRiRCFKrVotA
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib
# %matplotlib inline
from matplotlib import pyplot
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import os
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt

import sys
import argparse
import cv2 as cv
import h5py
from glob import glob


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#Code to generate data

import os
import sys
import numpy as np
import argparse
import cv2 as cv
import h5py
from glob import glob
from tqdm import tqdm

#import matplotlib.pyplot as plt


class Dataset(torch.utils.data.Dataset):
    def __init__(self, file_name):
        super(Dataset, self).__init__()
        self.file_name = file_name
        with h5py.File(file_name, 'r') as data:
            self.keys = list(data.keys())

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        with h5py.File(self.file_name, 'r') as data:
            example = np.array(data[self.keys[index]])
        return torch.Tensor(example)

    def shape(self):
        with h5py.File(self.file_name, 'r') as data:
            return np.array(data[self.keys[0]]).shape

def batch_psnr(clean_image, denoised_image):
    clean_image = clean_image.data.cpu().numpy().astype(np.float32)
    denoised_image = denoised_image.data.cpu().numpy().astype(np.float32)

    batch_psnr_val = 0
    for i in range(clean_image.shape[0]):
        batch_psnr_val += psnr(clean_image[i,:,:,:], denoised_image[i,:,:,:], data_range=1)

    return batch_psnr_val / clean_image.shape[0]

def setup_gpus():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device_ids = [i for i in range(torch.cuda.device_count())]
    if len(device_ids) > 3:
        device_ids = device_ids[:-1]
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, device_ids))
    return device_ids

train_set = 'train.h5'
val_set = 'val.h5'
batch_size = 100

assert os.path.exists(train_set), f'Cannot find training vectors file {train_set}'
assert os.path.exists(val_set), f'Cannot find validation vectors file {val_set}'

print('Loading datasets')

train_data = Dataset(train_set)
val_data = Dataset(val_set)

print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(val_data)}')

train_loader = DataLoader(dataset=train_data, num_workers=os.cpu_count(), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_data, num_workers=os.cpu_count(), batch_size=batch_size, shuffle=False)   

#train_loader.size()

'''for batch in tqdm(train_loader):
  print(batch.shape[0])
  #reshaped_batch = (batch.reshape(128, 13*13))
  #print("reshaped: ", reshaped_batch.shape )
  
  #noise_25 = torch.FloatTensor(batch.size()).normal_(mean=0, std=25/255)
  #noise_25_flat = torch.flatten(noise_25)
  
  #noise_25 = torch.FloatTensor(reshaped_batch.size()).normal_(mean=0, std=25/255)
  #print(noise_25.shape)
  #print(noise_25_flat.shape)'''

RESUME_TRAINING = False

DEPTH = 17
INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 64
FILTER_SIZE = 3

LEARNING_RATE = 0.1
WEIGHT_DECAY = 0.00001
MOMENTUM = 0.9

END_LR = 0.00001
START_LR = 0.01
LR_EPOCHS = 50
#GAMMA = np.log(END_LR / START_LR) / (-LR_EPOCHS)
GAMMA = 0.94
NUM_ITERATIONS = 50

# detect gpus and setup environment variables
device_ids = setup_gpus()
print(f'Cuda devices found: {[torch.cuda.get_device_name(i) for i in device_ids]}')

"""Large training set: The union of the LabelMe dataset [22]
(containing approximately 150, 000 images) and the
Berkeley segmentation dataset.

Small training set: The Berkeley segmentation dataset
[15], containing 200 images.

We train networks with different architectures and patch
sizes. We write for instance L–17–4x2047 for a network
that is trained on the large training set with patch size 17 ×
17 and 4 hidden layers of size 2047; 


similarly S–13–2x511
for a network that is trained on the small training set with
patch size 13 × 13 and 2 hidden layers of size 511.
"""

#Hyperparameters
num_epochs = 128
batch_size = 100
learning_rate = 0.1

#Implementing for the small data set, number of hidden layers = 2
patch_size = 17
num_input = patch_size*patch_size #Vector of 
num_hidden = 2047 #number of neurons on hidden layer = 511
num_output= patch_size*patch_size

def init_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight.data, a=0, mode='fan_in')
    if isinstance(layer, nn.BatchNorm2d):
        layer.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant_(layer.bias.data, 0.0)

#Denoising model with 17x17 patch input, two hidden layers
class MLP_denoising(nn.Module):
    def __init__(self, num_input, num_hidden, num_output):
      super(MLP_denoising, self).__init__()
      self.num_input  =  num_input
      self.num_hidden =  num_hidden
      self.num_output  =  num_output

      self.fc1 = nn.Linear(self.num_input, self.num_hidden)
      self.fc2 = nn.Linear(self.num_hidden, self.num_hidden)
      self.fc3 = nn.Linear(self.num_hidden, self.num_hidden)
      self.fc4 = nn.Linear(self.num_hidden, self.num_hidden)
      self.out = nn.Linear(self.num_hidden, self.num_output)
          
        
    def forward(self, x):
      s1 = F.relu(self.fc1(x))
      s2 = F.relu(self.fc2(s1))
      s3 = F.relu(self.fc3(s2))
      s4 = F.relu(self.fc3(s3))
      out = self.out(s4)

      return out

#from DnCNN import DnCNN
#from DnCNN import init_weights

model = MLP_denoising(num_input, num_hidden, num_output)
model.apply(init_weights)
model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()

print(model)
print('Number of trainable params: ',sum(p.numel() for p in model.parameters() if p.requires_grad))

loss = nn.MSELoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=START_LR)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA)

epochs_trained = 0

if RESUME_TRAINING:
    checkpoint = torch.load('/logs/model.state')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epochs_trained = checkpoint['epoch']

epoch_losses = []
epoch_val_losses = []
epoch_psnrs = []
min_val_loss = 1000

#print(model)
for epoch in range(NUM_ITERATIONS - epochs_trained):
    print(f'Training epoch {epoch+1} with lr={optimizer.param_groups[0]["lr"]}')
    epoch_loss = 0
    num_steps = 0
    model.train()

    for batch in tqdm(train_loader):

        
        model.zero_grad()
        optimizer.zero_grad()

        noise_25 = torch.FloatTensor(batch.size()).normal_(mean=0, std=25/255)
        noisy_image = batch + noise_25

        noisy_image = Variable(noisy_image.cuda())
        #noise_25 =  Variable(noise_25.cuda())
        batch =  Variable(batch.cuda())
        noisy_input = (noisy_image.reshape(batch.size()[0], num_input)) #reshape the batch, flatten the input

        predict = model(noisy_input)
        predict = (predict.reshape(batch.size())) #reshape back to og size
        #batch_loss = loss(noise_25, predict) / batch.size()[0]
        batch_loss = loss(batch, predict) / batch.size()[0]
        batch_loss.backward()
        optimizer.step()
        epoch_loss += batch_loss.detach()
        num_steps += 1

        '''if (num_steps+1)%10 == 0:
          print("loss: ", epoch_loss/num_steps)'''
    
    epoch_loss/= num_steps

        

    epoch_losses.append(epoch_loss)

    epoch_val_loss = 0
    epoch_psnr = 0
    val_num_steps = 0
    model.eval()

    with torch.no_grad(): 

      for batch in tqdm(val_loader):
          
          noise_25 = torch.FloatTensor(batch.size()).normal_(mean=0, std=25/255)
          noisy_image = batch + noise_25

          

          noisy_image = Variable(noisy_image.cuda())
          #noise_25 =  Variable(noise_25.cuda())
          batch =  Variable(batch.cuda())
          noisy_input = (noisy_image.reshape(batch.size()[0], num_input)) #reshape the batch, flatten the input

          predict = model(noisy_input)
          predict = (predict.reshape(batch.size())) #reshape back to og size
          #val_loss = loss(noise_25, predict) / batch.size()[0]
          val_loss = loss(batch, predict) / batch.size()[0]
          epoch_val_loss += val_loss.detach()

          # Calculate PSNR
          #denoised_image = torch.clamp(noisy_image - predict, 0.0, 1.0)
          denoised_image = torch.clamp(predict, 0.0, 1.0)
          epoch_psnr += batch_psnr(batch, denoised_image)          
          
          val_num_steps += 1

    epoch_val_loss/= val_num_steps
    epoch_psnr /= val_num_steps

    if epoch_val_loss < min_val_loss:
        print('Saving best model')
        min_val_loss = val_loss
        
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch_train_losses': epoch_losses,
                'epoch_val_losses': epoch_val_losses,
                'epoch_psnrs': epoch_psnrs,
                }, 'logs/t_star.state')



    scheduler.step()
    
    epoch_val_losses.append(epoch_val_loss)
    epoch_psnrs.append(epoch_psnr)

    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch_train_losses': epoch_losses,
            'epoch_val_losses': epoch_val_losses,
            'epoch_psnrs': epoch_psnrs,
            }, 'logs/model.state')

    print(f'Epoch {epoch} train loss = {epoch_loss}, val loss = {epoch_val_loss}, psnr = {epoch_psnr}')

