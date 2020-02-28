import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys

#download the dataset and initialize train and test sets
transform = transforms.Compose(
    [transforms.ToTensor(),
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
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


#get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

#show images
#imshow(torchvision.utils.make_grid(images))
#print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


#define the CNN
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


net = VGG('VGG11')

#define a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
a = 'train'
if a == 'train':
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
	PATH = './cifar_net.pth'
	torch.save(net.state_dict(), PATH)

elif a == 'test':
	#display some images from the test set to see what you're testing the network on
	dataiter = iter(testloader)
	images, labels = dataiter.next()
	# print images
	#imshow(torchvision.utils.make_grid(images))
	#print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

	#load the trained network
	net = VGG('VGG11')
	net.load_state_dict(torch.load('./cifar_net.pth'))

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

	


