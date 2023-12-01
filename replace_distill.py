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
from model import ResNet18Poly, general_relu_poly
from model_relu import ResNet18Relu
import numpy as np
import shutil
import sys

from datetime import datetime
from utils import *

parser = argparse.ArgumentParser(description='Fully poly replacement on ResNet for ImageNet')

parser.add_argument('--id', default=0, type=int)
parser.add_argument('--total_epochs', default=100, type=int)
parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
parser.add_argument('--w_decay', default=0.000, type=float, help='w decay rate')
parser.add_argument('--optim', type=str, default='adamw', choices = ['sgd', 'adamw'])
parser.add_argument('--batch_size_train', type=int, default=500, help='Batch size for training')
parser.add_argument('--batch_size_test', type=int, default=500, help='Batch size for testing')
parser.add_argument('--data_augment', type=ast.literal_eval, default=True)
parser.add_argument('--train_subset', type=ast.literal_eval, default=True, help='if train on the subset of ImageNet or the full ImageNet')
parser.add_argument('--pixel_wise', type=ast.literal_eval, default=True, help='if use pixel-wise poly replacement')
parser.add_argument('--channel_wise', type=ast.literal_eval, default=True, help='if use channel-wise relu_poly class')

parser.add_argument('--poly_weight_inits', nargs=3, type=float, default=[0, 1, 0], help='relu_poly weights initial values')
parser.add_argument('--poly_weight_factors', nargs=3, type=float, default=[1, 1, 1], help='adjust the learning rate of the three weights in relu_poly')

parser.add_argument('--mask_decrease', type=str, default='1-sinx', choices = ['1-sinx', 'e^(-x/10)', 'linear'], help='how the relu replacing mask decreases')
parser.add_argument('--mask_epochs', default=80, type=int, help='the epoch that the relu replacing mask will decrease to 0')

parser.add_argument('--loss_fm_type', type=str, default='at', choices = ['at', 'mse', 'custom_mse'], help='the type for the feature map loss')
parser.add_argument('--loss_fm_factor', default=100, type=float, help='the factor of the feature map loss, set to 0 to disable')
parser.add_argument('--loss_ce_factor', default=1, type=float, help='the factor of the cross-entropy loss, set to 0 to disable')
parser.add_argument('--loss_kd_factor', default=1, type=float, help='the factor of the knowledge distillation loss, set to 0 to disable')

parser.add_argument('--lookahead', type=ast.literal_eval, default=True, help='if enable look ahead for the optimizer')
parser.add_argument('--lr_anneal', type=str, default='None', choices = ['None', 'cos'])

parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--pbar', type=ast.literal_eval, default=True)
parser.add_argument('--log_root', type=str)
args = parser.parse_args()

torch.manual_seed(10)
torch.cuda.manual_seed_all(10)


def main(args):
    t_max = args.total_epochs
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    
    if args.data_augment:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    if args.train_subset:
        trainset = torchvision.datasets.ImageFolder(root='/home/uconn/xiexi/poly_replace/subset2' , transform=transform_train)
    else:
        trainset = torchvision.datasets.ImageFolder(root='/home/uconn/dataset/imagenet/train' , transform=transform_train)
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size = args.batch_size_train, shuffle=True, num_workers=args.num_workers)
    testset = torchvision.datasets.ImageFolder(
        root='/home/uconn/dataset/imagenet/val', transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size = args.batch_size_test, shuffle=False, num_workers=args.num_workers)

    model = ResNet18Poly(args.channel_wise, args.pixel_wise, args.poly_weight_inits, args.poly_weight_factors, relu2_extra_factor=1)

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

    def initialize_model(model):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # initialize_model(model)

    model_relu = ResNet18Relu()
    model_relu = model_relu.cuda()
    
    copy_parameters(pretrain_model, model_relu)   
    
    # tmp_test_acc = 0
    # test(pretrain_model.cuda(), 0, tmp_test_acc, None)
    # tmp_test_acc = 0
    # test(model.cuda(), 0, tmp_test_acc, 1)
    
    print("gpu count =", torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        pretrain_model = torch.nn.DataParallel(pretrain_model)

    # pretrain_model = pretrain_model.cuda()
    model = model.cuda()
    model.eval()

    def find_submodule(module, submodule_name):
        names = submodule_name.split('.')
        for name in names:
            module = getattr(module, name, None)
            if module is None:
                return None
        return module

    relu_poly_params = []
    other_params = []

    for name, param in model.named_parameters():
        submodule_name = '.'.join(name.split('.')[:-1])
        submodule = find_submodule(model, submodule_name)
        
        if isinstance(submodule, general_relu_poly):
            relu_poly_params.append(param)
        else:
            other_params.append(param)

    optimizer_params = [
        {'params': relu_poly_params, 'lr': args.lr},
        {'params': other_params, 'lr': args.lr}
    ]

    # i = 0
    # for param_group in optimizer_params:
    #     lr = param_group['lr'] 
    #     for p in param_group['params']:
    #         i += 1
    #         print(f"{i} Param Name: {p.shape}, Learning Rate: {lr}")
    
    optimizer = optim.AdamW(optimizer_params)

    # optimizer = optim.AdamW(param_groups)
    # optimizer = optim.AdamW(model.parameters(), lr = args.lr, weight_decay=args.w_decay)

    if args.lookahead:
        optimizer = Lookahead(optimizer)

    if args.lr_anneal == "None":
        lr_scheduler = None
    elif args.lr_anneal == "cos":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)


    current_datetime = datetime.now().strftime('%Y%m%d%H%M%S')
    if args.log_root:
        log_root = args.log_root
    else:
        log_root = 'runs' + current_datetime

    log_dir = log_root
    
    print("log_dir = ", log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    args_file = os.path.join(log_dir, "args.txt")
    with open(args_file, 'w') as file:
        for key, value in vars(args).items():
            file.write(f'{key}: {value}\n')
    print(f"Arguments saved in {args_file}")

    writer = SummaryWriter(log_dir=log_dir)

    values_list = [str(value) for key, value in vars(args).items()]
    print_prefix = ' '.join(values_list)

    def train(model_s, model_t, optimizer, epoch, mask):
        # model_s.train_fz_bn()
        model_s.train()
        model_t.eval()
        # model_t.train()

        train_loss = 0
        train_loss_kd = 0
        train_loss_ce = 0
        train_loss_fm = 0

        if args.pbar:
            pbar = tqdm(trainloader, total=len(trainloader), desc=f"Epo {epoch} Lr {optimizer.param_groups[0]['lr']:.1e}", ncols=120)
        else:
            pbar = trainloader

        top1_total = 0
        top5_total = 0
        total = 0

        if args.loss_fm_type == "mse":
            loss_fm_fun = nn.MSELoss()
        elif args.loss_fm_type == "custom_mse":
            loss_fm_fun = custom_mse_loss
        else:
            loss_fm_fun = at_loss

        criterion_kd = SoftTarget(4.0).cuda()
        criterion_ce = nn.CrossEntropyLoss()

        for x, y in pbar:
            x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            with torch.no_grad():
                out_t, fms_t = model_t.forward_with_fms(x, 0)
            out_s, fms_s = model_s.forward_with_fms(x, mask)

            loss_fm = sum(loss_fm_fun(x, y) for x, y in zip(fms_s, fms_t))

            loss_kd = criterion_kd(out_s, out_t) 
            loss_ce = criterion_ce(out_s, y) 

            loss = 0
            if args.loss_fm_factor > 0:
                loss += loss_fm * args.loss_fm_factor
            if args.loss_kd_factor > 0:
                loss += loss_kd * args.loss_kd_factor
            if args.loss_ce_factor > 0:
                loss += loss_ce * args.loss_ce_factor
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_loss_fm += loss_fm.item()
            train_loss_kd += loss_kd.item()
            train_loss_ce += loss_kd.item()

            top1, top5 = accuracy(out_s, y, topk=(1, 5))
            top1_total += top1[0] * x.size(0)
            top5_total += top5[0] * x.size(0)
            total += x.size(0)
            pbar.set_postfix_str(f"L{train_loss/total:.2e},fm{train_loss_fm/total:.2e},kd{train_loss_kd/total:.2e},ce{train_loss_ce/total:.2e}, 1a {100*top1_total/total:.1f}, 5a {100*top5_total/total:.1f}")

        train_acc = (top1_total / total).item()
        # print('Epoch', epoch, 'Training Acc:', train_acc*100)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/train', train_loss/total, epoch)
        writer.add_scalar('Loss_fm/train', train_loss_fm/total, epoch)
        writer.add_scalar('Loss_kd/train', train_loss_kd/total, epoch)
        writer.add_scalar('Loss_ce/train', train_loss_ce/total, epoch)
        return train_acc
            
    def test(model, epoch, best_acc, mask):
        model.eval()
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
            pbar.set_postfix_str(f"1a {100*top1_total/total:.2f}, 5a {100*top5_total/total:.2f}, best {100*best_acc:.2f}")

        test_acc = (top1_total / total).item()
        writer.add_scalar('Accuracy/test', test_acc, epoch)
        
        if test_acc > best_acc:
            best_acc = test_acc

        return best_acc

    if args.mask_decrease == "1-sinx":
        mask_x = np.linspace(0, np.pi / 2, args.mask_epochs)[1:]
        mask_y = 1 - np.sin(mask_x)
    elif args.mask_decrease == "e^(-x/10)":
        mask_x = np.linspace(0, 80, args.mask_epochs)[1:]
        mask_y = np.exp(-mask_x / 10)
    else:  # linear decrease
        mask_y = np.linspace(1, 0, args.mask_epochs)[1:]


    for epoch in range(start_epoch, start_epoch + t_max):
        if epoch < start_epoch + args.mask_epochs:
            mask = mask_y[epoch - start_epoch]
        else:
            mask = 0
        if mask < 0:
            mask = 0

        # mask = 1
        # mask = 0

        print("mask = ", mask)
        writer.add_scalar('Mask value', mask, epoch)
        if epoch >= 1:
            total_elements, relu_elements = model.get_relu_density(mask)
            print(f"total_elements {total_elements}, relu_elements {relu_elements}, density = {relu_elements/total_elements}")
        train_acc = train(model, model_relu, optimizer, epoch, mask)

        if lr_scheduler is not None:
            lr_scheduler.step()
        # if train_acc > 0.5:
        if mask < 0.01:
            best_acc = test(model, epoch, best_acc, mask)

        # Save the model after each epoch
        torch.save(model.state_dict(), f"{log_dir}/model_epoch_{epoch}.pth")
        
    writer.close()

if __name__ == "__main__":
    # print_available_gpus()

    main(args)