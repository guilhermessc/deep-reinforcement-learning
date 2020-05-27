import numpy as np 
import helper

import torch
from torch import nn, optim
import torch.nn.functional as F 

import helper

class mNetwork(nn.Module):
	def __init__(self):
		super().__init__()

		self.fc1 = nn.Linear(784, 128)
		self.fc2 = nn.Linear(128, 64)
		self.fc3 = nn.Linear(64, 10)

		# initializing weights
		self.fc1.bias.data.fill_(0) # bias is a parameter .data gives access to the numbers
		self.fc1.weight.data.normal_(std=0.01)

		# to train the network
		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = optim.SGD(self.parameters(), lr=0.01)

	def forward(self, x):
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		x = self.fc3(x)
		x = F.softmax(x, dim=1)

		return x

	def train(self, x, y):

		self.optimizer.zero_grad()

		# Forward and backward passes
		output = self.forward(x)
		loss = self.criterion(output, y)
		loss.backward()
		self.optimizer.step()

		return loss.item()

	def fit(self, X, Y, epochs=3, print_every=50):
		steps=0
		for e in range(epochs):
			running_loss = 0
			for i, y in enumerate(Y):
				steps+=1
				# Train step
				running_loss+= self.train(X[i], y)

				if steps % print_every == 0:
					print("Epoch: {}/{}... ".format(e+1, epochs), 
						"Loss: {:.4f}".format(running_loss/print_every))
					running_loss=0



from torchvision import datasets, transforms

def main():

	# Define a transform to normalize the data
	transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=0.5, std=0.5),])
	# Download and load the training data
	trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

	X = []
	Y = []
	for x, y in iter(trainloader):
		x.resize_(x.size()[0], 784)
		X.append(x)
		Y.append(y)

	# Create model
	model = mNetwork()
	# Train model
	model.fit(X, Y)

main()