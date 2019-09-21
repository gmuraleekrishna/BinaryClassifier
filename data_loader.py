from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from cat_dog_dataset import CatDogDataset

data_transform = transforms.Compose([
	transforms.Resize(200),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_data(batch_size):
	train_dataset = CatDogDataset(train_files='train_files.csv',
	                                      root_dir='../datasets/Cat-Dog-data/cat-dog-train', transform=data_transform)
	test_dataset = CatDogDataset(train_files='test_files.csv', root_dir='../datasets/Cat-Dog-data/cat-dog-test',
	                                     transform=data_transform)
	
	num_train = len(train_dataset)
	indices = list(range(num_train))
	split = 10000
	train_indices, val_indices = indices[split:], indices[:split]

	train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4,
	                          sampler=SubsetRandomSampler(indices=train_indices))
	val_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4,
	                        sampler=SubsetRandomSampler(indices=val_indices))
	test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)
	
	return train_loader, val_loader, test_loader
