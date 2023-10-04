'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
# from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# Setup learning rate to 0.1 for sgd optimizer, and 0.005 for lamb optimizer
# lr = 0.5 for sgd transfer learning
# lr = 0.001 for adam transfer learning
parser.add_argument('--lr', default=0.1, type=float, choices = [0.5, 0.1, 0.005, 0.001, 0.002], help='learning rate')
# Setup w_decay to 0.0005 for sgd optimizer, and 0.02 for lamb optimizer
parser.add_argument('--w_decay', default=0.0005, type=float, choices = [0.0005, 0.02], help='w decay rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--gpus', type=int, default=0, 
    help='gpu device ids separated by comma. `all` indicates use all gpus.')
parser.add_argument('--pretrained', type=int, default=0, choices = [0, 1],
    help='Load pretrained ImageNet model or not.')
parser.add_argument('--optim', type=str, default='sgd', choices = ['sgd', 'adam'],
    help='gpu device ids separated by comma. `all` indicates use all gpus.')
args = parser.parse_args()

torch.cuda.set_device(args.gpus)
device = 'cuda:' + str(args.gpus) if torch.cuda.is_available() else 'cpu'

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])
transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(p=.40),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])
transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = ResNet34()
# net = ResNet50()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()

### Use ResNet18 Pretrained model from torchvision
print("Load pretrained or not: ", bool(args.pretrained))
print("Learning rate: ", args.lr)

# net = torchvision.models.resnet18(pretrained = args.pretrained)
# inchannel = net.fc.in_features
# net.fc = nn.Linear(inchannel, 10)

class ResNet18_cifar(nn.Module):
    def __init__(self):
        super().__init__()
        base = torchvision.models.resnet18(pretrained = args.pretrained)
        self.base = nn.Sequential(*list(base.children())[:-1])
        in_features = base.fc.in_features
        self.drop = nn.Dropout()
        self.final = nn.Linear(in_features,10)
    
    def forward(self,x):
        x = self.base(x)
        x = self.drop(x.view(-1,self.final.in_features))
        return self.final(x)
net = ResNet18_cifar()



net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint_transfer/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()



if args.optim == 'sgd':
#     param_groups = [
#     {'params':net.base.parameters(),'lr':args.lr/10, 'weight_decay':args.w_decay, 'momentum':0.9},
#     {'params':net.final.parameters(),'lr':args.lr, 'weight_decay':args.w_decay, 'momentum':0.9}
# ]
    param_groups = [
    {'params':net.base.parameters(),'lr':args.lr/10, 'weight_decay':args.w_decay},
    {'params':net.final.parameters(),'lr':args.lr, 'weight_decay':args.w_decay}
]
    # optimizer = optim.SGD(net.parameters(), lr=args.lr,
    #                     momentum=0.9, weight_decay=args.w_decay)
    optimizer = optim.SGD(param_groups)
else:
    param_groups = [
    {'params':net.base.parameters(),'lr':args.lr/10, 'weight_decay':args.w_decay},
    {'params':net.final.parameters(),'lr':args.lr, 'weight_decay':args.w_decay}
]
    # optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.w_decay)
    optimizer = optim.Adam(param_groups)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)



# Training
import time
def train(epoch):
    print('\nEpoch: %d' % epoch)
    current_time = time.ctime()
    print(current_time)

    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        # net.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print('Epoch', epoch, '  Training Acc:', correct/total)
        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


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
    print('Epoch', epoch, '  Test Acc:', correct/total)
            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving with acc:', acc)
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint_transfer'):
            os.mkdir('checkpoint_transfer')
        if args.pretrained:
            ext = ''
        else:
            ext = 'scratch'
        torch.save(state, './checkpoint_transfer/ckpt'+ args.optim + str(args.lr) + ext + '.pth')
        best_acc = acc
    print('Test Best Acc: %.4f%%' % (best_acc))

for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()
