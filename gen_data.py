import os
import sys
import numpy as np
import argparse
import cv2 as cv
import h5py
from glob import glob



def generate_data(train_path, valid_path, patch_size, stride, scaling_factors):
    print(f'[Data Generation] Creating training data from {train_path}')
    num_train = 0
    h5f = h5py.File('train.h5', 'w')
    num_train = 0
    for f in sorted(glob(os.path.join(train_path, '*.png'))):
        print(f'Preprocessing {f}')
        img = cv.imread(f)
        height, width, ch = img.shape

        for scale in scaling_factors:
            img_scaled = cv.resize(img, (int(height*scale), int(width*scale)), interpolation=cv.INTER_CUBIC)
            img_scaled = np.array(img_scaled[:,:,0].reshape((img_scaled.shape[0],img_scaled.shape[1],1))/255)
            patches = get_image_patches(img_scaled, patch_size, stride)
            print(f'  scaling: {scale}, num patches: {patches.shape[0]}')
            for patch_num in range(patches.shape[0]):
                # chanels first
                patch = np.einsum('ijk->kij', patches[patch_num,:,:,:].astype(np.float32))
                h5f.create_dataset(str(num_train), data=patch)
                num_train += 1

    h5f.close()

    print(f'[Data Generation] Creating validation data from {valid_path}')
    num_valid = 0
    h5f = h5py.File('val.h5', 'w')
    for f in sorted(glob(os.path.join(valid_path, '*.png'))):
        print(f'Preprocessing {f}')
        img = cv.imread(f)
        # channels first
        img = np.array(img[:,:,0].reshape((1,img.shape[0],img.shape[1]))/255, dtype=np.float32)
        h5f.create_dataset(str(num_valid), data=img)
        num_valid += 1
    h5f.close()
        
    print(f'Number of training examples {num_train}')    
    print(f'Number of validation examples {num_valid}')    


    pass

def get_image_patches(img, patch_size, stride):
    win_row_end = img.shape[0] - patch_size
    win_col_end = img.shape[1] - patch_size
    num_patches_rows = int((img.shape[0]-patch_size)/stride + 1)
    num_patches_cols = int((img.shape[1]-patch_size)/stride + 1)
    num_chs = int(img.shape[2])
    total_patches = int(num_patches_rows * num_patches_cols)

    patches = np.zeros((total_patches, patch_size, patch_size, num_chs), dtype=float)

    rows = np.arange(0,win_row_end+1, stride)
    cols = np.arange(0,win_col_end+1, stride)
    patch_num = 0
    for row in rows:
        for col in cols: 
            patch = img[row:row+patch_size, col:col+patch_size,:]
            patches[patch_num,:,:,:] = patch
            patch_num += 1

    return patches

def main():

    script_dir = os.path.dirname(os.path.realpath(__file__))
    
    parser = argparse.ArgumentParser(description="DnCNN-data generation")
    parser.add_argument("--train_path", type=str, default='data/train', help='root directory for training data')
    parser.add_argument("--valid_path", type=str, default='data/Set12', help='root directory for validation data')
    parser.add_argument("--patch_size", type=int, default=40, help="image patch size to train on")
    parser.add_argument("--stride", type=int, default=10, help="image patch stride")
    parser.add_argument("--scaling_factors", type=str, default='1,.9,.8,.7', help="image scaling")
    args = parser.parse_args()

    train_path = os.path.join(script_dir, args.train_path)
    valid_path = os.path.join(script_dir, args.valid_path)
    patch_size = args.patch_size
    stride = args.stride
    scaling_factors = [float(scale) for scale in args.scaling_factors.split(',')] 

    print(f'[args] training data: {train_path}')
    print(f'[args] validation data: {valid_path}')
    print(f'[args] patch size: {patch_size}, stride: {stride}')
    print(f'[args] scaling factors: {scaling_factors}')

    generate_data(train_path, valid_path, patch_size, stride, scaling_factors)

    return 0

if __name__ == '__main__':
    sys.exit(main())
