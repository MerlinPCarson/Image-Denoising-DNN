import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

model_name = sys.argv[1]

checkpoint = torch.load(model_name)

train_losses = checkpoint['epoch_train_losses']
val_losses = checkpoint['epoch_val_losses']
psnrs = checkpoint['epoch_psnrs']

plt.figure()
plt.plot(train_losses, label='training loss')
plt.plot(val_losses, label='validation loss')
plt.yscale('log')
plt.legend()
plt.grid()
plt.title('Training Loss Curves')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.tight_layout()

plt.figure()
plt.plot(psnrs, label='validation PSNR')
plt.legend()
plt.grid()
plt.title('Peak Signal to Noise Ratio')
plt.ylabel('PSNR')
plt.xlabel('epoch')
plt.tight_layout()

plt.show()
