from torch import nn


class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		
		self.features = nn.Sequential(
			nn.Conv2d(3, 32, 5, 1, 1),
			nn.MaxPool2d(5, 5),
			nn.ReLU(True),
			nn.BatchNorm2d(32),
			
			nn.Conv2d(32, 64, 3, 1, 1),
			nn.ReLU(True),
			nn.BatchNorm2d(64),
			
			nn.Conv2d(64, 64, 3, 1, 1),
			nn.MaxPool2d(5, 5),
			nn.ReLU(True),
			nn.BatchNorm2d(64),
			
			nn.Conv2d(64, 128, 3, 1, 1),
			nn.ReLU(True),
			nn.BatchNorm2d(128)
		)
		
		self.classifier = nn.Sequential(
			nn.Linear(128 * 7 * 7, 1024),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(in_features=1024, out_features=1)
		)
	
	def forward(self, x):
		x = self.features(x)
		print(x.shape)
		x = x.view(-1, 128 * 7 * 7)
		x = self.classifier(x)
		return x
