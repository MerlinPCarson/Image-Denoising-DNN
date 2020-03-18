import os
import sys
import pickle
import argparse
import cv2 as cv
import numpy as np
from glob import glob
from train import setup_gpus
#import train
import torch
from torch.autograd import Variable
from skimage.metrics import peak_signal_noise_ratio as psnr
import time

from DnCNN import DnCNN
from DnCNN import init_weights

DEPTH = 18 
INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 64
FILTER_SIZE = 3

def main():

    parser = argparse. ArgumentParser(description='Image Denoising')
    parser.add_argument('--img_dir', type=str, default='data/CBSD68', help='location of files to denoise')
    parser.add_argument('--out_dir', type=str, default='output', help='location to save output images')
    parser.add_argument('--img', type=str, help='location of a file to denoise')
    parser.add_argument('--noise_levels', type=str, default='15,25,50', help='noise level for training')
    parser.add_argument('--model_name', type=str, default='logs/t_star.state', help='location of model file')
    args = parser.parse_args()

    # normalize noise level
    #noise_level = args.noise_level/255
    noise_levels = [float(nl) for nl in args.noise_levels.split(',')] 

    print(f'Using noise levels {noise_levels}')

    assert os.path.exists(args.img_dir), f'Image directory {args.img_dir} not found'
    assert os.path.exists(args.model_name), f'Model {args.model_name} not found'

    # detect gpus and setup environment variables
    device_ids = setup_gpus()
    print(f'Cuda devices found: {[torch.cuda.get_device_name(i) for i in device_ids]}')

    # load model params
    #model_history = pickle.load(open(args.model_name.replace('.pt', '.npy'), 'rb'))
    num_channels = 3 #model_history['model']['num_channels']

    # load model 
    model = DnCNN(DEPTH, INPUT_CHANNELS, OUTPUT_CHANNELS, FILTER_SIZE)
    #model.apply(init_weights)
    model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
    checkpoint = torch.load(args.model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    #model = torch.load(args.model_name)
    model.eval() 

    model_dir = os.path.dirname(args.model_name)
    out_dir = os.path.join(model_dir, args.out_dir)
    # make sure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    timings = []

    for nl in noise_levels:
      test_psnr = 0
      psnr_improvement = 0
      num_test_files = 0
      for f in sorted(glob(os.path.join(args.img_dir, '*.png'))):
          print(f'Denoising {f} with noise level {nl}')

          img = cv.imread(f).astype(np.float32)[:,:,:num_channels]
          clean_img = np.einsum('ijk->kij', img.astype(np.float32)/255) 
          clean_img = np.expand_dims(clean_img, axis=0)
          clean_img = torch.FloatTensor(clean_img)

          # prepare noisy image
          noise = torch.FloatTensor(clean_img.size()).normal_(mean=0, std=nl/255)
          noisy_img = Variable((clean_img + noise).cuda())

          with torch.no_grad():
              start = time.time()
              denoised_img = torch.clamp(noisy_img - model(noisy_img), 0.0, 1.0)
              timings.append(time.time()-start)
              
          # save images
          file_name = os.path.basename(f)
          denoised_img = denoised_img.cpu().data.numpy().astype(np.float32)[0,:,:,:]
          denoised_img *= 255     # undo normalization
          denoised_img = np.einsum('ijk->jki', denoised_img)
          cv.imwrite(img=denoised_img.clip(0.0, 255.0).astype('uint8'), filename=os.path.join(out_dir, file_name.replace('.png', f'-{nl}-denoised.png')))

          noisy_img = noisy_img.cpu().data.numpy().astype(np.float32)[0,:,:,:]
          noisy_img *= 255        # undo normalization
          noisy_img = np.einsum('ijk->jki', noisy_img) 
          cv.imwrite(img=noisy_img.clip(0.0, 255.0).astype('uint8'), filename=os.path.join(out_dir, file_name.replace('.png', f'-{nl}-noisy.png')))

          psnr_pre = psnr(img, noisy_img, data_range=255)
          psnr_post = psnr(img, denoised_img, data_range=255)
          psnr_diff = psnr_post-psnr_pre
          print(f'PNSR of {f}: {psnr_post}, increase of {psnr_diff}')

          psnr_improvement += psnr_diff
          test_psnr += psnr_post
          num_test_files += 1

      print(f'Average PSNR of testset at noise level {nl} is {test_psnr/num_test_files}')
      print(f'Average increase in PSNR of testset at noise level {nl} is {psnr_improvement/num_test_files}')

      print(f'Average processing time is {sum(timings)/len(timings)}')
    return 0

if __name__ == '__main__':
    sys.exit(main())
