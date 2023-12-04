import os
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from torch.utils.tensorboard import SummaryWriter
import ast
from model import ResNet18Poly, general_relu_poly, convert_to_bf16_except_bn, find_submodule, copy_parameters
from model_relu import ResNet18Relu
import numpy as np
import re
from training import train, test, MaskProvider
from utils import Lookahead
from datetime import datetime

parser = argparse.ArgumentParser(description='Fully poly replacement on ResNet for ImageNet')

parser.add_argument('--id', default=0, type=int)

parser.add_argument('--total_epochs', default=100, type=int)
parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
parser.add_argument('--w_decay', default=0.000, type=float, help='w decay rate')
parser.add_argument('--optim', type=str, default='adamw', choices = ['sgd', 'adamw'])
parser.add_argument('--batch_size_train', type=int, default=500, help='Batch size for training')
parser.add_argument('--batch_size_test', type=int, default=500, help='Batch size for testing')
parser.add_argument('--data_augment', type=ast.literal_eval, default=True)
parser.add_argument('--train_subset', type=ast.literal_eval, default=True, help='if train on the 1/13 subset of ImageNet or the full ImageNet')
parser.add_argument('--pixel_wise', type=ast.literal_eval, default=True, help='if use pixel-wise poly replacement')
parser.add_argument('--channel_wise', type=ast.literal_eval, default=True, help='if use channel-wise relu_poly class')
parser.add_argument('--poly_weight_inits', nargs=3, type=float, default=[0, 1, 0], help='relu_poly weights initial values')
parser.add_argument('--poly_weight_factors', nargs=3, type=float, default=[0.1, 1, 0.1], help='adjust the learning rate of the three weights in relu_poly')
parser.add_argument('--mask_decrease', type=str, default='e^(-x/10)', choices = ['1-sinx', 'e^(-x/10)', 'linear'], help='how the relu replacing mask decreases')
parser.add_argument('--mask_epochs', default=80, type=int, help='the epoch that the relu replacing mask will decrease to 0')
parser.add_argument('--loss_fm_type', type=str, default='at', choices = ['at', 'mse', 'custom_mse'], help='the type for the feature map loss')
parser.add_argument('--loss_fm_factor', default=100, type=float, help='the factor of the feature map loss, set to 0 to disable')
parser.add_argument('--loss_ce_factor', default=1, type=float, help='the factor of the cross-entropy loss, set to 0 to disable')
parser.add_argument('--loss_kd_factor', default=1, type=float, help='the factor of the knowledge distillation loss, set to 0 to disable')
parser.add_argument('--lookahead', type=ast.literal_eval, default=True, help='if enable look ahead for the optimizer')
parser.add_argument('--lr_anneal', type=str, default='None', choices = ['None', 'cos'])
parser.add_argument('--bf16', type=ast.literal_eval, default=False, help='if enable training with bf16 precision')
parser.add_argument('--fp16', type=ast.literal_eval, default=False, help='if enable training with float16 precision')

parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--pbar', type=ast.literal_eval, default=True)
parser.add_argument('--log_root', type=str)

parser.add_argument('--resume', type=ast.literal_eval, default=False)
parser.add_argument('--resume_dir', type=str)
parser.add_argument('--resume_epoch', type=int, default=None)

args = parser.parse_args()

torch.manual_seed(10)
torch.cuda.manual_seed_all(10)

def parse_args_line(line):
    key, value = line.split(": ", 1)
    try:
        value = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        pass
    return key, value

if args.resume:
    args_file = args.resume_dir + '/args.txt'
    with open(args_file, 'r') as file:
        for line in file:
            key, value = parse_args_line(line.strip())
            if hasattr(args, key) and not key.startswith('resume'):
                setattr(args, key, value)

def main(args):
    t_max = args.total_epochs
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    mask_provider = MaskProvider(args.mask_decrease, args.mask_epochs)

    if args.bf16 or args.fp16:
        if args.bf16:
            raise NotImplementedError("BF16 support is not yet available.")
        if args.fp16:
            print("FP16 enabled.")

    if args.bf16 or args.fp16:
        if args.bf16:
            print("enable bf16")
        if args.fp16:
            print("enable fp16")
    
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

    dummy_input = torch.rand(1, 3, 224, 224) 
    model(dummy_input, 0)

    pretrain_model = torchvision.models.resnet18(pretrained=True)

    def find_latest_epoch(resume_dir):
        max_epoch = -1
        for filename in os.listdir(resume_dir):
            match = re.match(r'model_epoch_(\d+)\.pth', filename)
            if match:
                epoch = int(match.group(1))
                if epoch > max_epoch:
                    max_epoch = epoch
        return max_epoch

    if args.resume:
        if args.resume_epoch is None:
            start_epoch = find_latest_epoch(args.resume_dir) + 1
        else:
            start_epoch = args.resume_epoch
        t_max = args.total_epochs - start_epoch
        checkpoint_path = os.path.join(args.resume_dir, f'model_epoch_{start_epoch - 1}.pth')
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path)
            model.load_state_dict(state_dict)
        else:
            print(f"No checkpoint found at {checkpoint_path}")
    else:
        copy_parameters(pretrain_model, model)  

    # model = model.cuda()

    # writer = SummaryWriter(log_dir=args.resume_dir)

    # # test(args, testloader, model, 80, 0, 0, writer)
    
    # mask_x = np.linspace(0, 80, args.mask_epochs)[1:]
    # mask_y = np.exp(-mask_x / 10)
    
    # for epoch in range(22, 100):
    #     mask = mask_y[epoch - 0]
    #     model.load_state_dict(torch.load(os.path.join(args.resume_dir, f'model_epoch_{epoch}.pth')))
    #     test(args, testloader, model, epoch, 0, mask, writer)

    # initialize_resnet(model)

    model_relu = ResNet18Relu()
    
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

    model_relu = model_relu.cuda()

    if args.bf16:
        model = convert_to_bf16_except_bn(model)
        model_relu = convert_to_bf16_except_bn(model_relu)

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

    if args.lr_anneal is None or args.lr_anneal == "None":
        lr_scheduler = None
    elif args.lr_anneal == "cos":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)

    scaler = torch.cuda.amp.GradScaler(enabled=args.bf16 or args.fp16)

    if args.resume and args.resume_dir:
        log_root = args.resume_dir
    else:
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

    for epoch in range(start_epoch, start_epoch + t_max):
        mask = mask_provider.get_mask(epoch)
        # mask = 1
        # mask = 0

        print("mask = ", mask)
        writer.add_scalar('Mask value', mask, epoch)
        if args.pixel_wise:
            total_elements, relu_elements = model.get_relu_density(mask)
            print(f"total_elements {total_elements}, relu_elements {relu_elements}, density = {relu_elements/total_elements}")
        train_acc = train(args, trainloader, model, model_relu, optimizer, scaler, epoch, mask, writer)

        if lr_scheduler is not None:
            lr_scheduler.step()
        # if train_acc > 0.5:
        if mask < 0.01 or True:
            best_acc = test(args, testloader, model, epoch, best_acc, mask, writer)

        # Save the model after each epoch
        torch.save(model.state_dict(), f"{log_dir}/model_epoch_{epoch}.pth")
        
    writer.close()

if __name__ == "__main__":
    # print_available_gpus()

    main(args)