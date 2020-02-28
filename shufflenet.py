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

class ShuffleBlock(nn.Module):
    def __init__(self, groups=2):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, C//g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)


class SplitBlock(nn.Module):
    def __init__(self, ratio):
        super(SplitBlock, self).__init__()
        self.ratio = ratio

    def forward(self, x):
        c = int(x.size(1) * self.ratio)
        return x[:, :c, :, :], x[:, c:, :, :]


class BasicBlock(nn.Module):
    def __init__(self, in_channels, split_ratio=0.5):
        super(BasicBlock, self).__init__()
        self.split = SplitBlock(split_ratio)
        in_channels = int(in_channels * split_ratio)
        self.conv1 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.conv3 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.shuffle = ShuffleBlock()

    def forward(self, x):
        x1, x2 = self.split(x)
        out = F.relu(self.bn1(self.conv1(x2)))
        out = self.bn2(self.conv2(out))
        out = F.relu(self.bn3(self.conv3(out)))
        out = torch.cat([x1, out], 1)
        out = self.shuffle(out)
        return out


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        mid_channels = out_channels // 2
        # left
        self.conv1 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, mid_channels,
                               kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        # right
        self.conv3 = nn.Conv2d(in_channels, mid_channels,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels)
        self.conv4 = nn.Conv2d(mid_channels, mid_channels,
                               kernel_size=3, stride=2, padding=1, groups=mid_channels, bias=False)
        self.bn4 = nn.BatchNorm2d(mid_channels)
        self.conv5 = nn.Conv2d(mid_channels, mid_channels,
                               kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm2d(mid_channels)

        self.shuffle = ShuffleBlock()

    def forward(self, x):
        # left
        out1 = self.bn1(self.conv1(x))
        out1 = F.relu(self.bn2(self.conv2(out1)))
        # right
        out2 = F.relu(self.bn3(self.conv3(x)))
        out2 = self.bn4(self.conv4(out2))
        out2 = F.relu(self.bn5(self.conv5(out2)))
        # concat
        out = torch.cat([out1, out2], 1)
        out = self.shuffle(out)
        return out


class ShuffleNetV2(nn.Module):
    def __init__(self, net_size):
        super(ShuffleNetV2, self).__init__()
        out_channels = configs[net_size]['out_channels']
        num_blocks = configs[net_size]['num_blocks']

        self.conv1 = nn.Conv2d(3, 24, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.in_channels = 24
        self.layer1 = self._make_layer(out_channels[0], num_blocks[0])
        self.layer2 = self._make_layer(out_channels[1], num_blocks[1])
        self.layer3 = self._make_layer(out_channels[2], num_blocks[2])
        self.conv2 = nn.Conv2d(out_channels[2], out_channels[3],
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels[3])
        self.linear = nn.Linear(out_channels[3], 10)

    def _make_layer(self, out_channels, num_blocks):
        layers = [DownBlock(self.in_channels, out_channels)]
        for i in range(num_blocks):
            layers.append(BasicBlock(out_channels))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # out = F.max_pool2d(out, 3, stride=2, padding=1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


configs = {
    0.5: {
        'out_channels': (48, 96, 192, 1024),
        'num_blocks': (3, 7, 3)
    },

    1: {
        'out_channels': (116, 232, 464, 1024),
        'num_blocks': (3, 7, 3)
    },
    1.5: {
        'out_channels': (176, 352, 704, 1024),
        'num_blocks': (3, 7, 3)
    },
    2: {
        'out_channels': (224, 488, 976, 2048),
        'num_blocks': (3, 7, 3)
    }
}


a = 'train'

net = ShuffleNetV2(net_size=0.5)

#define a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

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
	net = ShuffleNetV2(net_size=0.5)
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

	


