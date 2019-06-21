import h5py
import os
import glob
import numpy as np
from io import BytesIO
from skimage.io import imread
import matplotlib.pylab as plt
from tqdm.auto import tqdm



images = 'data/images'
dataset_name = 'my_dataset.h5'



def list_images(data_dir, name):
    images = []
    path = os.path.join(data_dir, name)
    for img_type in ['png', 'jpg', 'jpeg']:
        images += glob.glob(f'{path}/**/*.{img_type}')
    return images


def file2uint(fpath):
    with open(fpath, 'rb') as file:
        data = file.read()
        return np.frombuffer(data, dtype='uint8')


def create_dataset(data_dir, dataset_name):

	train_imgs = list_images(data_dir, 'train')
	val_imgs = list_images(data_dir, 'val')
	test_imgs = list_images(data_dir, 'test')

	# Define special type with dynamic length
	dt = h5py.special_dtype(vlen=np.dtype('uint8'))

	# Create datasets
	with h5py.File(dataset, 'w') as out:
		out.create_dataset("x_train", (len(train_imgs),), dtype=dt)
		out.create_dataset("x_val", (len(val_imgs),), dtype=dt)
		out.create_dataset("x_test", (len(test_imgs),), dtype=dt)
   
    # Write data
	with h5py.File(dataset, 'a') as dset:

		for i, img in enumerate(tqdm(train_imgs, 'Processing train set..')):
			dset['x_train'][i] = file2uint(img)

		for i, img in enumerate(tqdm(val_imgs, 'Processing val set..')):
			dset['x_val'][i] = file2uint(img)
        
		for i, img in enumerate(tqdm(val_imgs, 'Processing test set..')):
			dset['x_test'][i] = file2uint(img)



if __name__ == "__main__":
	create_dataset(images, dataset_name)

	# Test
	hf = h5py.File(dataset, 'r')
	img_data = hf['x_train'][0]
	decoded_img = imread(BytesIO(img_data))

	plt.axis('off')
	plt.imshow(decoded_img)
	plt.show()