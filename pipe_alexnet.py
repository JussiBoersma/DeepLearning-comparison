import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import sys

#download the dataset and initialize train and test sets
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	 
transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


transform1 = transforms.Compose([            #[1]
	transforms.Resize(256),                    #[2]
	transforms.CenterCrop(224),                #[3]
	transforms.ToTensor(),                     #[4]
	transforms.Normalize(                      #[5]
	mean=[0.485, 0.456, 0.406],                #[6]
	std=[0.229, 0.224, 0.225]                  #[7]
)])


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform1)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
		   

#alexnettr = models.shufflenet_v2_x1_0(pretrained=True)
#alexnettr.eval()


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
class AlexNet(nn.Module):
	def __init__(self, num_classes=10):
		super(AlexNet, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
			nn.Tanh(),
			nn.MaxPool2d(kernel_size=2),
			nn.Conv2d(64, 192, kernel_size=3, padding=1),
			nn.Tanh(),
			nn.MaxPool2d(kernel_size=2),
			nn.Conv2d(192, 384, kernel_size=3, padding=1),
			nn.Tanh(),
			nn.Conv2d(384, 256, kernel_size=3, padding=1),
			nn.Tanh(),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.Tanh(),
			nn.MaxPool2d(kernel_size=2),
		)
		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(256 * 2 * 2, 4096),
			nn.Tanh(),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.Tanh(),
			nn.Linear(4096, num_classes),
		)

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), 256 * 2 * 2)
		x = self.classifier(x)
		return x


net = AlexNet()
#net = alexnettr
#define a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
a='train'
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
	PATH = './cifar_net_pretr.pth'
	torch.save(net.state_dict(), PATH)
	
	

elif a == 'test':
	#display some images from the test set to see what you're testing the network on
	dataiter = iter(testloader)
	images, labels = dataiter.next()
	# print images
	#imshow(torchvision.utils.make_grid(images))
	#print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(1)))

	#load the trained network
	net = AlexNet()
	#net = alexnettr.eval()
	
	net.load_state_dict(torch.load('./cifar_net_pretr.pth'))

	#run the network classifier on the selected images
	#outputs = net(images)

	#_, predicted = torch.max(outputs, 1)

	#print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
	#							  for j in range(4)))
								  
								  
	# this shows how the network performs on the whole dataset
	correct = 0
	total = 0
	with torch.no_grad():
		for data in testloader:
			images, labels = data
			outputs = net(images)
			#with open('labelsimgnet.txt') as f:
			#	classes1 = [line.strip() for line in f.readlines()]
			_, predicted = torch.max(outputs.data, 1)
				#print(classes1[predicted])
				#print(classes[labels])
				#str = classes1[predicted]
				#if classes[labels] in str:
				#	correct = correct + 1
				#total = total+1
			total += labels.size(0)
				correct += (predicted == labels).sum().item()

	print('Accuracy of the network on the 10000 test images: %d %%' % (
		100 * correct / total))
		
					
#			with open('labelsimgnet.txt') as f:
	#			classes = [line.strip() for line in f.readlines()]
	#			_, index = torch.max(outputs, 1)
	#			percentage = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
	#			print(labels[index[0]], percentage[index[0]].item())
	#			