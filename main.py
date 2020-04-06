'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('-m', '--model', default=0, type=int)
parser.add_argument('-o', '--optimizer', default='sgd', type=str)
parser.add_argument('-e', '--epoch', default=50, type=int)
parser.add_argument('-d', '--dir', default='./save', type=str)
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()
print(args)

MODEL = args.model
OPTIMIZER = args.optimizer
EPOCH = args.epoch
SAVE_DIR = args.dir
if not os.path.exists(SAVE_DIR):
	os.makedirs(SAVE_DIR)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
	transforms.RandomCrop(32, padding=4),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
if MODEL == 0:
	net = VGG('VGG19')
elif MODEL == 1:
	net = ResNet18()
elif MODEL == 2:
	net = PreActResNet18()
elif MODEL == 3:
	net = GoogLeNet()
elif MODEL == 4:
	net = DenseNet121()
elif MODEL == 5:
	net = ResNeXt29_2x64d()
elif MODEL == 6:
	net = MobileNet()
elif MODEL == 7:
	net = MobileNetV2()
elif MODEL == 8:
	net = DPN92()
elif MODEL == 9:
	net = ShuffleNetG2()
elif MODEL == 10:
	net = SENet18()
elif MODEL == 11:
	net = ShuffleNetV2(1)
else:
	net = EfficientNetB0()
net = net.to(device)
if device == 'cuda':
	net = torch.nn.DataParallel(net)
	cudnn.benchmark = True

if args.resume:
	# Load checkpoint.
	print('==> Resuming from checkpoint..')
	assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
	checkpoint = torch.load('./checkpoint/ckpt.pth')
	net.load_state_dict(checkpoint['net'])
	best_acc = checkpoint['acc']
	start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
if OPTIMIZER == 'sgd':
	optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
else:
	optimizer = optim.Adam(net.parameters(), lr=args.lr)


# Training
def train(epoch):
	print('\nEpoch: %d' % epoch)
	net.train()
	train_loss = 0
	correct = 0
	total = 0
	for batch_idx, (inputs, targets) in enumerate(trainloader):
		inputs, targets = inputs.to(device), targets.to(device)
		optimizer.zero_grad()
		outputs = net(inputs)
		loss = criterion(outputs, targets)
		loss.backward()
		optimizer.step()

		train_loss += loss.item()
		_, predicted = outputs.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().item()

		progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
		             % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
	return train_loss / len(trainloader), 100. * correct / total


def test(epoch):
	global best_acc
	net.eval()
	test_loss = 0
	correct = 0
	total = 0
	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(testloader):
			inputs, targets = inputs.to(device), targets.to(device)
			outputs = net(inputs)
			loss = criterion(outputs, targets)

			test_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()

			progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

	# Save checkpoint.
	acc = 100. * correct / total
	if acc > best_acc:
		print('Saving..')
		state = {
			'net': net.state_dict(),
			'acc': acc,
			'epoch': epoch,
		}
		if not os.path.isdir('checkpoint'):
			os.mkdir('checkpoint')
		torch.save(state, './checkpoint/ckpt.pth')
		best_acc = acc

	return test_loss / len(testloader), 100. * correct / total


trainAcc = []
testAcc = []
trainLoss = []
testLoss = []
for epoch in range(start_epoch, start_epoch + EPOCH):
	ls, ac = train(epoch)
	trainAcc.append(ac)
	trainLoss.append(ls)
	ls, ac = test(epoch)
	testAcc.append(ac)
	testLoss.append(ls)

plt.figure()
ax1 = plt.subplot()
plt.title("loss and accuracy of training and testing")
ax1.set_xlabel("epochs")
x = range(len(trainLoss))
ax1.set_ylabel("loss")
ax2 = ax1.twinx()
ax2.set_ylabel("accuracy")

kwargs = {
	"marker": None,
	"lw": 2,
}
l1, = ax1.plot(x, trainLoss, color="tab:blue", label="train loss", **kwargs)
l2, = ax2.plot(x, trainAcc, color="tab:orange", label="train accuracy", **kwargs)
l3, = ax1.plot(x, testLoss, color="tab:green", label="test loss", **kwargs)
l4, = ax2.plot(x, testAcc, color="tab:red", label="test accuracy", **kwargs)

plt.legend(handles=[l1, l2, l3, l4], loc="center right")
sv = plt.gcf()
sv.savefig(os.path.join(SAVE_DIR, "lossAndAcc" + str(args.lr) + OPTIMIZER + ".png"), format="png", dpi=300)
with open(os.path.join(SAVE_DIR, "result.txt"), "a+") as f:
	print("model:", MODEL, "lr", args.lr, "optimizer", OPTIMIZER, "finalacc", testAcc[-1], file=f)
