import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import argparse
from torch.utils.tensorboard import SummaryWriter
import time
import ast
import torch.nn.init as init
from tqdm import tqdm
# from utils import progress_bar

parser = argparse.ArgumentParser(description='Transfer from ImageNet pretrain to CIFAR10')

parser.add_argument('--id', default=0, type=int)
parser.add_argument('--epoch', default=50, type=int)
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--w_decay', default=0.000, type=float, help='w decay rate')
parser.add_argument('--optim', type=str, default='adamw', choices = ['sgd', 'adamw'])
parser.add_argument('--batch_size_train', type=int, default=500, help='Batch size for training')
parser.add_argument('--batch_size_test', type=int, default=500, help='Batch size for testing')
parser.add_argument('--hidden_dim', type=int, default=0, help='Hidden dimension in the classifier')
parser.add_argument('--drop', type=float, default=0)
parser.add_argument('--add_bias1', type=ast.literal_eval, default=False)
parser.add_argument('--add_bias2', type=ast.literal_eval, default=False)
parser.add_argument('--add_relu', type=ast.literal_eval, default=False)
parser.add_argument('--unfreeze_mode', type=str, default='none', choices=['none', 'b2_conv2', 'b2'])
parser.add_argument('--data_augment', type=ast.literal_eval, default=False)
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--pbar', type=ast.literal_eval, default=True)
parser.add_argument('--log_root', type=str, default="runs")
args = parser.parse_args()

def print_available_gpus():
    num_devices = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_devices}")

    for i in range(num_devices):
        device_name = torch.cuda.get_device_name(i)
        device_capability = torch.cuda.get_device_capability(i)
        print(f"  GPU {i}: {device_name}, Capability: {device_capability}")

class ResNet_transfer(nn.Module):
    def __init__(self, drop, hidden_dim, add_bias1, add_bias2, add_relu):
        super().__init__()
        base = torchvision.models.resnet18(pretrained = True)

        self.before_layer4 = nn.Sequential(*list(base.children())[:-3])
        self.layer4_block1 = nn.Sequential(*list(base.children())[-3:-2])[0][0]
        
        self.b2_conv1 = nn.Sequential(*list(base.children())[-3:-2])[0][1].conv1

        if add_bias1:
            self.b2_bias1 = nn.Parameter(torch.zeros(1,512,7,7))
        else:
            self.b2_bias1 = None

        self.b2_bn1 = nn.Sequential(*list(base.children())[-3:-2])[0][1].bn1
        self.b2_relu = nn.ReLU()

        self.b2_list1 = [self.b2_conv1, self.b2_bn1, self.b2_relu]

        self.b2_conv2 = nn.Sequential(*list(base.children())[-3:-2])[0][1].conv2

        if add_bias2:
            self.b2_bias2 = nn.Parameter(torch.zeros(1,512,7,7))
        else:
            self.b2_bias2 = None

        self.b2_bn2 = nn.Sequential(*list(base.children())[-3:-2])[0][1].bn2

        if add_relu:
            self.add_relu = nn.ReLU()
        else:
            self.add_relu = None
        self.pool = nn.Sequential(*list(base.children())[-2:-1])

        self.b2_list2 = [self.b2_conv2, self.b2_bn2, self.pool]

        # self.b2_bias3 = nn.Parameter(torch.zeros(1,512,1,1))
        
        if drop > 0:
            part1 = nn.Sequential(nn.Flatten(), nn.Dropout(drop))
        else:
            part1 = nn.Sequential(nn.Flatten())

        if hidden_dim > 0:
            part2 = nn.Sequential(nn.Linear(512, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 10), nn.Softmax(dim=1)) 
        else:
            part2 = nn.Sequential(nn.Linear(512, 10), nn.Softmax(dim=1))

        self.classifier = nn.Sequential(part1, part2)

    def forward(self,x):
        x = self.before_layer4(x)
        x = self.layer4_block1(x)
        
        x = self.b2_conv1(x)
        if self.b2_bias1 is not None:
            x += self.b2_bias1
        x = self.b2_bn1(x)
        x = self.b2_relu(x)

        x = self.b2_conv2(x)
        if self.b2_bias2 is not None:
            x += self.b2_bias2
        x = self.b2_bn2(x)
        if self.add_relu is not None:
            x = self.add_relu(x)
        x = self.pool(x)

        # if self.b2_bias3 is not None:
        #     x += self.b2_bias3
        x = self.classifier(x)
        return x

def main(args):
    t_max = args.epoch

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    if args.data_augment:
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=.40),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size = args.batch_size_train, shuffle=True, num_workers=args.num_workers)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size = args.batch_size_test, shuffle=False, num_workers=args.num_workers)

    model = ResNet_transfer(args.drop, args.hidden_dim, args.add_bias1, args.add_bias2, args.add_relu)
    
    for param in model.before_layer4.parameters():
        param.requires_grad = False
    for param in model.layer4_block1.parameters():
        param.requires_grad = False
    if args.unfreeze_mode == "none":
        for layer in model.b2_list1 + model.b2_list2:
            for param in layer.parameters():
                param.requires_grad = False
        
    elif args.unfreeze_mode == "b2_conv2":
        for layer in model.b2_list1:
            for param in layer.parameters():
                param.requires_grad = False

    print("gpu count =", torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model = model.cuda()
    model.eval()

    criterion = nn.CrossEntropyLoss()

    if args.optim == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr = args.lr, weight_decay=args.w_decay)
    else:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr = args.lr, weight_decay=args.w_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)

    log_dir = f"{args.log_root}/{args.id}_ep{args.epoch}_lr{args.lr}_wd{args.w_decay}_opt{args.optim}"\
        f"_btrain{args.batch_size_train}_btest{args.batch_size_test}_hdim{args.hidden_dim}_dp{args.drop}"\
            f"_b1{args.add_bias1}_b2{args.add_bias2}_re{args.add_relu}_unfreeze{args.unfreeze_mode}_aug{args.data_augment}"
    writer = SummaryWriter(log_dir=log_dir)

    print_prefix = f"{args.id} {args.epoch} {args.lr} {args.w_decay} {args.optim} {args.batch_size_train} {args.batch_size_test} "\
        f"{args.hidden_dim} {args.drop} {args.add_bias1} {args.add_bias2} {args.add_relu} {args.unfreeze_mode} {args.data_augment}"


    def train(epoch):
        # print(print_prefix, 'Epoch', epoch, 'start ...')
        if isinstance(model, torch.nn.DataParallel):
            _model = model.module
        else:
            _model = model
        _model.classifier.train()
        if args.unfreeze_mode == "b2_conv2":
            for layer in _model.b2_list2:
                layer.train()
        elif args.unfreeze_mode == "b2":
            for layer in _model.b2_list1 + _model.b2_list2:
                layer.train()

        train_loss = 0
        correct = 0
        total = 0
        if args.pbar:
            pbar = tqdm(trainloader, total=len(trainloader), desc=f"Epo {epoch} Training Lr {optimizer.param_groups[0]['lr']:.1e}", ncols=100)
        else:
            pbar = trainloader
        # for batch_idx, (inputs, targets) in enumerate(trainloader):
        for x, y in pbar:
            x, y = x.cuda(), y.cuda()
            if args.pbar:
                pbar.set_description_str(f"Epo {epoch} Training Lr {optimizer.param_groups[0]['lr']:.1e}", refresh=True)
            
            optimizer.zero_grad()
            fx = model(x)
            loss = criterion(fx, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            total += y.size(0)
            correct += torch.argmax(fx, 1).eq(y).float().sum().item()
            if args.pbar:
                pbar.set_postfix_str(f"Acc {100*correct/total:.2f}%")
        train_acc = correct / total
        print(print_prefix, 'Epoch', epoch, 'Training Acc:', train_acc)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
            
    def test(epoch, best_acc):
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        if args.pbar:
            pbar = tqdm(testloader, total=len(testloader), desc=f"Epo {epoch} Testing", ncols=100)
        else:
            pbar = testloader

        # for batch_idx, (inputs, targets) in enumerate(testloader):
        for x, y in pbar:
            x, y = x.cuda(), y.cuda()
            with torch.no_grad():
                fx = model(x)
            # loss = criterion(outputs, targets)
            total += y.size(0)
            correct += torch.argmax(fx, 1).eq(y).float().sum().item()
            acc = correct/total
            if args.pbar:
                pbar.set_postfix_str(f"Acc {100*acc:.2f}%")
            # test_loss += loss.item()
        
        test_acc = correct / total
        writer.add_scalar('Accuracy/test', test_acc, epoch)
                
        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            best_acc = acc
        with open(log_dir + "/best_accuracy.txt", "a") as f:
            f.write(f"best_acc={best_acc}\n")
        # print(print_prefix, 'Epoch', epoch, 'Test Acc:', test_acc, 'Test Best Acc: %.4f%%' % (best_acc))
        print(print_prefix, 'Epoch', epoch, 'Test Acc:', test_acc, 'Test Best:', best_acc)
        return best_acc

    for epoch in range(start_epoch, start_epoch + t_max):
        train(epoch)
        scheduler.step()
        best_acc = test(epoch, best_acc)
        
    writer.close()

if __name__ == "__main__":
    # print_available_gpus()

    main(args)