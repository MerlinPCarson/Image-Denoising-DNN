import torch.nn as nn
import torch, h5py
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
from DnCNN import DnCNN
from DnCNN import init_weights


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
batch_size = 128

assert os.path.exists(train_set), f'Cannot find training vectors file {train_set}'
assert os.path.exists(val_set), f'Cannot find validation vectors file {val_set}'

print('Loading datasets')

train_data = Dataset(train_set)
val_data = Dataset(val_set)

print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(val_data)}')

train_loader = DataLoader(dataset=train_data, num_workers=os.cpu_count(), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_data, num_workers=os.cpu_count(), batch_size=batch_size, shuffle=False)              

RESUME_TRAINING = False

DEPTH = 17
INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 64
FILTER_SIZE = 3

LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.00001
MOMENTUM = 0.9

END_LR = 0.00001
START_LR = 0.01
LR_EPOCHS = 50
GAMMA = np.log(END_LR / START_LR) / (-LR_EPOCHS)

NUM_ITERATIONS = 50

# detect gpus and setup environment variables
device_ids = setup_gpus()
print(f'Cuda devices found: {[torch.cuda.get_device_name(i) for i in device_ids]}')

model = DnCNN(DEPTH, INPUT_CHANNELS, OUTPUT_CHANNELS, FILTER_SIZE)
model.apply(init_weights)
model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()

loss = nn.MSELoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

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

print(model)
for epoch in range(NUM_ITERATIONS - epochs_trained):
    print(f'Training epoch {epoch+1} with lr={optimizer.param_groups[0]["lr"]}')
    epoch_loss = 0
    num_steps = 0
    model.train()

    for batch in tqdm(train_loader):
        model.zero_grad()
        optimizer.zero_grad()

        noise_25 = torch.FloatTensor(batch.size()).normal_(mean=0, std=25/255)
        batch = batch + noise_25

        batch = Variable(batch.cuda())
        noise_25 =  Variable(noise_25.cuda())

        predict = model(batch)
        batch_loss = loss(noise_25, predict) / batch.size()[0]
        batch_loss.backward()
        optimizer.step()
        epoch_loss += batch_loss.item()
        num_steps += 1

    epoch_losses.append(epoch_loss/num_steps)

    epoch_val_loss = 0
    epoch_psnr = 0
    num_steps = 0
    model.eval()

    for batch in tqdm(val_loader):
        noise_25 = torch.FloatTensor(batch.size()).normal_(mean=0, std=25/255)
        noisy_image = batch + noise_25

        noisy_image = Variable(noisy_image.cuda())
        noise_25 =  Variable(noise_25.cuda())

        predict = model(noisy_image)
        val_loss = loss(noise_25, predict) / batch.size()[0]
        epoch_val_loss += val_loss.item()
        num_steps += 1

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict, f'logs/t_star.state')

        # Calculate PSNR
        denoised_image = torch.clamp(noisy_image - predict, 0.0, 1.0)
        epoch_psnr += batch_psnr(batch, denoised_image)

    scheduler.step()
    
    epoch_val_losses.append(epoch_val_loss/num_steps)
    epoch_psnrs.append(epoch_psnr/num_steps)

    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch_train_losses': epoch_losses,
            'epoch_val_losses': epoch_val_losses,
            'epoch_psnr': epoch_psnr,
            }, 'logs/model.state')

    print(f'Epoch {epoch} train loss = {epoch_loss/num_steps}, val loss = {epoch_val_loss/num_steps}, PSNR = {epoch_psnr/num_steps}')