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
from model import ResNet18Poly
import numpy as np

from datetime import datetime
from utils import *

parser = argparse.ArgumentParser(description='Transfer from ImageNet pretrain to CIFAR10')

parser.add_argument('--id', default=0, type=int)
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
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
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--pbar', type=ast.literal_eval, default=True)
parser.add_argument('--log_root', type=str)
args = parser.parse_args()

torch.manual_seed(10)
torch.cuda.manual_seed_all(10)


def main(args):
    t_max = args.epoch
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    trainset = torchvision.datasets.ImageFolder(
        root='/home/uconn/xiexi/poly_replace/subset2' , transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size = args.batch_size_train, shuffle=True, num_workers=args.num_workers)
    testset = torchvision.datasets.ImageFolder(
        root='/home/uconn/dataset/imagenet/val', transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size = args.batch_size_test, shuffle=False, num_workers=args.num_workers)

    criterion = nn.CrossEntropyLoss()

    if args.log_root:
        log_root = args.log_root
    else:
        current_datetime = datetime.now().strftime('%Y%m%d%H%M%S')
        log_root = 'runs' + current_datetime

    log_dir = f"{log_root}/{args.id}_ep{args.epoch}_lr{args.lr}_wd{args.w_decay}_opt{args.optim}"\
        f"_btrain{args.batch_size_train}_btest{args.batch_size_test}_hdim{args.hidden_dim}_dp{args.drop}"\
            f"_b1{args.add_bias1}_b2{args.add_bias2}_re{args.add_relu}_unfreeze{args.unfreeze_mode}_aug{args.data_augment}"
    
    print("log_dir = ", log_dir)

    writer = SummaryWriter(log_dir=log_dir)

    print_prefix = f"{args.id} {args.epoch} {args.lr} {args.w_decay} {args.optim} {args.batch_size_train} {args.batch_size_test} "\
        f"{args.hidden_dim} {args.drop} {args.add_bias1} {args.add_bias2} {args.add_relu} {args.unfreeze_mode} {args.data_augment}"

    def accuracy(output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
                res.append(correct_k.mul_(1.0 / batch_size))
            return res

    def train(model_s, model_t, optimizer, epoch, mask):
        # model_s.train_fz_bn()
        model_s.train()
        model_t.eval()
        # model_t.train()

        train_loss = 0
        # correct = 0
        if args.pbar:
            pbar = tqdm(trainloader, total=len(trainloader), desc=f"Epo {epoch} Training Lr {optimizer.param_groups[0]['lr']:.1e}", ncols=100)
        else:
            pbar = trainloader

        top1_total = 0
        top5_total = 0
        total = 0
        loss_fun = nn.MSELoss()
        # loss_fun = at_loss

        for x, y in pbar:
            x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            with torch.no_grad():
                out_t, fm1_t, fm2_t, fm3_t, fm4_t = model_t(x)
            out_s, fm1_s, fm2_s, fm3_s, fm4_s = model_s.forward_with_fm(x, mask)
            loss = 0
            loss += loss_fun(fm1_t, fm1_s)*500
            loss += loss_fun(fm2_t, fm2_s)*500
            loss += loss_fun(fm3_t, fm3_s)*500
            loss += loss_fun(fm4_t, fm4_s)*500
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # correct += torch.argmax(fx, 1).eq(y).float().sum().item()

            top1, top5 = accuracy(out_s, y, topk=(1, 5))
            top1_total += top1[0] * x.size(0)
            top5_total += top5[0] * x.size(0)
            total += x.size(0)
            pbar.set_postfix_str(f"Loss{train_loss/total:.2f}, 1a {100*top1_total/total:.2f}, 5a {100*top5_total/total:.2f}")

        train_acc = (top1_total / total).item()
        print('Epoch', epoch, 'Training Acc:', train_acc)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/train', train_loss/total, epoch)
            
    def test(model, epoch, best_acc, mask):
        model.eval()
        # correct = 0
        top1_total = 0
        top5_total = 0
        total = 0
        if args.pbar:
            pbar = tqdm(testloader, total=len(testloader), desc=f"Epo {epoch} Testing", ncols=100)
        else:
            pbar = testloader

        for x, y in pbar:
            x, y = x.cuda(), y.cuda()
            with torch.no_grad():
                if mask is not None:
                    out = model(x, mask)
                else:
                    out = model(x)
            top1, top5 = accuracy(out, y, topk=(1, 5))
            top1_total += top1[0] * x.size(0)
            top5_total += top5[0] * x.size(0)
            total += x.size(0)
            pbar.set_postfix_str(f"Top-1 Acc {100*top1_total/total:.2f}%, Top-5 Acc {100*top5_total/total:.2f}%")

        test_acc = (top1_total / total).item()
        writer.add_scalar('Accuracy/test', test_acc, epoch)

        # Save checkpoint.
        
        if test_acc > best_acc:
            best_acc = test_acc

        print('Epoch', epoch, 'Test Acc:', test_acc, 'Test Best:', best_acc)
        return best_acc

    model = ResNet18Poly()

    pretrain_model = torchvision.models.resnet18(pretrained=True)

    def copy_parameters(model1, model2):
        for name1 in model1.state_dict():
            param1 = model1.state_dict()[name1]
            if not isinstance(param1, torch.Tensor):
                continue
            if name1.startswith("layer"):
                name2 = name1[:6] + "_" + name1[7:]
            elif name1.startswith("fc"):
                name2 = name1.replace("fc", "linear", 1)
            else:
                name2 = name1
            
            name2 = name2.replace("downsample", "shortcut", 1)

            assert(name2 in model2.state_dict())    
            model2.state_dict()[name2].copy_(param1.data)
        
    copy_parameters(pretrain_model, model)   

    for param in pretrain_model.parameters():
        param.requires_grad = False 
    
    # tmp_test_acc = 0
    # test(pretrain_model.cuda(), 0, tmp_test_acc, None)
    # tmp_test_acc = 0
    # test(model.cuda(), 0, tmp_test_acc, 1)

    print("gpu count =", torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        pretrain_model = torch.nn.DataParallel(pretrain_model)

    pretrain_model = pretrain_model.cuda()
    model = model.cuda()
    model.eval()
    
    optimizer = optim.AdamW(model.parameters(), lr = args.lr, weight_decay=args.w_decay)

    # optimizer = Lookahead(optimizer)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)

    mask_x = np.linspace(0, np.pi / 2, 80)[1:]
    mask_y = 1 - np.sin(mask_x)

    for epoch in range(start_epoch, start_epoch + t_max):
        # mask = torch.tensor(1 - epoch / (start_epoch + t_max), dtype=torch.float).cuda()
        # mask = 1 - ((epoch + 1) / (start_epoch + 80))

        if epoch <= 78:
            mask = mask_y[epoch]
        else:
            mask = 0

        # mask = 0
        if mask < 0:
            mask = 0

        print("mask = ", mask)
        writer.add_scalar('Mask value', mask, epoch)
        train(model, pretrain_model, optimizer, epoch, mask)
        # scheduler.step()
        best_acc = test(model, epoch, best_acc, mask)

        # Save the model after each epoch
        torch.save(model.state_dict(), f"{log_dir}/model_epoch_{epoch}.pth")
        
    writer.close()

if __name__ == "__main__":
    # print_available_gpus()

    main(args)