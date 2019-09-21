from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import transform
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader


class CatDogDataset(Dataset):
	"""CatDog Landmarks dataset."""
	
	def __init__(self, train_files, root_dir, transform=None):
		"""
		Args:
			json_file (string): Path to the csv file with .
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied on a sample.
		"""
		self._frame = pd.read_csv(train_files)
		self.root_dir = root_dir
		self.transform = transform
	
	def __len__(self):
		return self._frame.shape[0]
	
	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		
		img_name = os.path.join(self.root_dir, self._frame.iloc[idx, 0])
		image = Image.open(img_name)
		label = self._frame.iloc[idx, 1].astype('float')
		
		if self.transform:
			image = self.transform(image)
		
		return image, label


if __name__ == '__main__':
	dataset = CatDogDataset(train_files='train_files.csv', root_dir='../datasets/Cat-Dog-data/cat-dog-train')
	dataset.__getitem__(0)
