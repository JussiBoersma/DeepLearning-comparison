import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#download the dataset and initialize train and test sets
transform = transforms.Compose(
    [transforms.ColorJitter(brightness=[0.99,1], contrast=[0.99,1], saturation=0, hue=0),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
		   


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    #img = img * contrast_factor + brighness_factor
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


#get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

#show images
imshow(torchvision.utils.make_grid(images))
#print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

#define the CNN
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        
        # Convolutional layers
                            #Init_channels, channels, kernel_size, padding) 
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2,2)
        
        # FC layers
        # Linear layer (64x4x4 -> 500)
        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        
        # Linear Layer (500 -> 10)
        self.fc2 = nn.Linear(500, 10)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(F.elu(self.conv1(x)))
        x = self.pool(F.elu(self.conv2(x)))
        x = self.pool(F.elu(self.conv3(x)))
        
        # Flatten the image
        x = x.view(-1, 64*4*4)
        x = self.dropout(x)
        x = F.elu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

net = CNNNet()


#define a loss function and optimizer
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#optimizer = optim.Adam(net.parameters(), lr=0.001) 
#optimizer = optim.RMSprop(net.parameters(), lr = 0.001, alpha = 0.9)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01)

if sys.argv[1] == 'train':
	#train the network
	for epoch in range(2):  # loop over the dataset multiple times

		running_loss = 0.0
		for i, data in enumerate(trainloader, 0):
			# get the inputs; data is a list of [inputs, labels]
			inputs, labels = data

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = net(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			# print statistics
			running_loss += loss.item()
			if i % 2000 == 1999:    # print every 2000 mini-batches
				print('[%d, %5d] loss: %.3f' %
					  (epoch + 1, i + 1, running_loss / 2000))
				running_loss = 0.0

	print('Finished Training')

	#save the trained model
	PATH = './cifar_net_nomomentum.pth'
	torch.save(net.state_dict(), PATH)

elif sys.argv[1] == 'test':
	#display some images from the test set to see what you're testing the network on
	dataiter = iter(testloader)
	images, labels = dataiter.next()
	# print images
	#imshow(torchvision.utils.make_grid(images))
	#print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

	#load the trained network
	net = CNNNet()
	net.load_state_dict(torch.load('./cifar_net_nomomentum.pth'))

	#run the network classifier on the selected images
	outputs = net(images)

	_, predicted = torch.max(outputs, 1)

	print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
								  for j in range(4)))
								  
								  
	# this shows how the network performs on the whole dataset
	correct = 0
	total = 0
	with torch.no_grad():
		for data in testloader:
			images, labels = data
			outputs = net(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	print('Accuracy of the network on the 10000 test images: %d %%' % (
		100 * correct / total))

	


