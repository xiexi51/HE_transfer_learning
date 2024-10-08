import os
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights
import argparse
from torch.utils.tensorboard import SummaryWriter
import ast
from model_poly_avg import ResNet18FullPoly, relu_fullpoly, copy_parameters
from model_relu import ResNet18Relu
import numpy as np
import re
from ddp_training import ddp_train, ddp_test
from utils import Lookahead, MaskProvider
from datetime import datetime
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

def process(pn, args):
    torch.cuda.set_device(pn)
    torch.manual_seed(10)
    torch.cuda.manual_seed_all(10)

    process_group = torch.distributed.init_process_group(backend="nccl", init_method='env://', world_size=args.total_gpus, rank=pn)

    t_max = args.total_epochs
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    mask_provider = MaskProvider(args.mask_decrease, args.mask_epochs)

    if args.bf16 or args.fp16:
        if args.bf16:
            raise KeyError("BF16 support is not yet available.")
        if args.fp16:
            raise KeyError("FP16 support is not yet available.")
            # print("FP16 enabled.")

    if pn == 0:
        print("gpu count =", torch.cuda.device_count())
    
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
    

    train_sampler = DistributedSampler(trainset, num_replicas=args.total_gpus, rank=pn)
    trainloader = torch.utils.data.DataLoader(trainset, sampler=train_sampler, batch_size=args.batch_size_train, num_workers=args.num_workers, pin_memory=True, shuffle=False)

    
    testset = torchvision.datasets.ImageFolder(
        root='/home/uconn/dataset/imagenet/val', transform=transform_test)
    
    test_sampler = DistributedSampler(testset, num_replicas=args.total_gpus, rank=pn)
    testloader = torch.utils.data.DataLoader(testset, sampler=test_sampler, batch_size=args.batch_size_test, num_workers=args.num_workers, pin_memory=True, shuffle=False)

    model = ResNet18FullPoly(args.poly_weight_factors, relu2_extra_factor=1)

    dummy_input = torch.rand(1, 3, 224, 224) 
    model((dummy_input, 0))

    pretrain_model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)

    def find_latest_epoch(resume_dir):
        max_epoch = -1
        for filename in os.listdir(resume_dir):
            match = re.match(r'model_epoch_(\d+)\.pth', filename)
            if match:
                epoch = int(match.group(1))
                if epoch > max_epoch:
                    max_epoch = epoch
        return max_epoch

    if args.resume or args.reload:
        if args.resume:
            if args.resume_epoch is None:
                start_epoch = find_latest_epoch(args.resume_dir) + 1
            else:
                start_epoch = args.resume_epoch    
            t_max = args.total_epochs - start_epoch
            checkpoint_path = os.path.join(args.resume_dir, f'model_epoch_{start_epoch - 1}.pth')
        else:
            checkpoint_path = os.path.join(args.resume_dir, f'model_epoch_{args.resume_epoch}.pth')

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path)
            state_dict = {key: value for key, value in state_dict.items() if not key.endswith('rand_mask')}
            model.load_state_dict(state_dict, strict=False)
        else:
            print(f"No checkpoint found at {checkpoint_path}")
    else:
        raise KeyError("The replacement of avgpool should only be conducted when args.reload = True")
        

    model_teacher = ResNet18FullPoly([0.1, 1, 0.1])
    torch.load("/home/uconn/xiexi/poly_replace_success_backup/runs20231225035445")

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    model = model.cuda()
    model.eval()

    model_relu = model_relu.cuda()

    for param in model_relu.parameters():
        param.requires_grad = False
    
    model = DistributedDataParallel(model, device_ids=[pn])
    optimizer = optim.AdamW(model.parameters(), lr = args.lr, weight_decay=args.w_decay)

    if args.lookahead:
        optimizer = Lookahead(optimizer)

    if args.lr_anneal is None or args.lr_anneal == "None":
        lr_scheduler = None
    elif args.lr_anneal == "cos":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)

    if args.resume and args.resume_dir:
        log_root = args.resume_dir
    else:
        current_datetime = datetime.now().strftime('%Y%m%d%H%M%S')
        if args.log_root:
            log_root = args.log_root
        else:
            log_root = 'runs' + current_datetime

    log_dir = log_root
    
    if pn == 0:
        print("log_dir = ", log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        args_file = os.path.join(log_dir, "args.txt")
        with open(args_file, 'w') as file:
            for key, value in vars(args).items():
                file.write(f'{key}: {value}\n')
        print(f"Arguments saved in {args_file}")

        writer = SummaryWriter(log_dir=log_dir)
    else:
        writer = None

    values_list = [str(value) for key, value in vars(args).items()]
    print_prefix = ' '.join(values_list)

    # if pn == 0:
    #     print(start_epoch, t_max)
    #     for arg, value in vars(args).items():
    #         print(f"{arg}: {value}")
    if args.reload:
        test_acc, best_acc = ddp_test(args, testloader, model, args.resume_epoch, best_acc, 0, writer, pn)

    for epoch in range(start_epoch, start_epoch + t_max):
        train_sampler.set_epoch(epoch)
        avgmask = mask_provider.get_mask(epoch)

        if pn == 0:
            print("mask = ", avgmask)
            writer.add_scalar('Mask value', avgmask, epoch)
            if isinstance(model, DistributedDataParallel):
                total_elements, maxpool_elements = model.module.get_maxpool_remain_density(avgmask)
            else:
                total_elements, maxpool_elements = model.get_maxpool_remain_density(avgmask)
            print(f"total_elements {total_elements}, maxpool_elements {maxpool_elements}, density = {maxpool_elements/total_elements}")
        
        omit_fms = 5
        train_acc = ddp_train(args, trainloader, model, model_relu, optimizer, epoch, avgmask, writer, pn, omit_fms)

        if avgmask < 0.01 or False:
            test_acc, best_acc = ddp_test(args, testloader, model, epoch, best_acc, avgmask, writer, pn)

        if lr_scheduler is not None:
            lr_scheduler.step()

        if pn == 0:
            if isinstance(model, DistributedDataParallel):
                torch.save(model.module.state_dict(), f"{log_dir}/model_epoch_{epoch}.pth")
            else:
                torch.save(model.state_dict(), f"{log_dir}/model_epoch_{epoch}.pth")

    if writer is not None:
        writer.close()

if __name__ == "__main__":

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29506'

    parser = argparse.ArgumentParser(description='Fully poly replacement on ResNet for ImageNet')
    parser.add_argument('--id', default=0, type=int)
    parser.add_argument('--total_epochs', default=200, type=int)
    parser.add_argument('--lr', default=0.00005, type=float, help='learning rate')
    parser.add_argument('--w_decay', default=0.000, type=float, help='w decay rate')
    parser.add_argument('--optim', type=str, default='adamw', choices = ['sgd', 'adamw'])
    parser.add_argument('--batch_size_train', type=int, default=300, help='Batch size for training')
    parser.add_argument('--batch_size_test', type=int, default=300, help='Batch size for testing')
    parser.add_argument('--data_augment', type=ast.literal_eval, default=True)
    parser.add_argument('--train_subset', type=ast.literal_eval, default=False, help='if train on the 1/13 subset of ImageNet or the full ImageNet')
    parser.add_argument('--pixel_wise', type=ast.literal_eval, default=True, help='if use pixel-wise poly replacement')
    parser.add_argument('--channel_wise', type=ast.literal_eval, default=True, help='if use channel-wise relu_poly class')
    parser.add_argument('--poly_weight_inits', nargs=3, type=float, default=[0, 1, 0], help='relu_poly weights initial values')
    parser.add_argument('--poly_weight_factors', nargs=3, type=float, default=[0.1, 1, 0.1], help='adjust the learning rate of the three weights in relu_poly')
    parser.add_argument('--mask_decrease', type=str, default='0', choices = ['0', '1-sinx', 'e^(-x/10)', 'linear'], help='how the relu replacing mask decreases')
    parser.add_argument('--mask_epochs', default=30, type=int, help='the epoch that the relu replacing mask will decrease to 0')
    parser.add_argument('--loss_fm_type', type=str, default='at', choices = ['irg', 'at', 'mse', 'custom_mse'], help='the type for the feature map loss')
    parser.add_argument('--loss_fm_factor', default=100, type=float, help='the factor of the feature map loss, set to 0 to disable')
    parser.add_argument('--loss_ce_factor', default=1, type=float, help='the factor of the cross-entropy loss, set to 0 to disable')
    parser.add_argument('--loss_kd_factor', default=0.1, type=float, help='the factor of the knowledge distillation loss, set to 0 to disable')
    parser.add_argument('--lookahead', type=ast.literal_eval, default=True, help='if enable look ahead for the optimizer')
    parser.add_argument('--lr_anneal', type=str, default='None', choices = ['None', 'cos'])
    parser.add_argument('--bf16', type=ast.literal_eval, default=False, help='if enable training with bf16 precision')
    parser.add_argument('--fp16', type=ast.literal_eval, default=False, help='if enable training with float16 precision')
    
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--pbar', type=ast.literal_eval, default=True)
    parser.add_argument('--log_root', type=str)

    parser.add_argument('--resume', type=ast.literal_eval, default=False)
    parser.add_argument('--resume_dir', type=str)
    parser.add_argument('--resume_epoch', type=int, default=None)

    parser.add_argument('--reload', type=ast.literal_eval, default=False)

    # parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)

    args = parser.parse_args()

    def parse_args_line(line):
        key, value = line.split(": ", 1)
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
        return key, value

    if args.reload:
        args_file = args.resume_dir + '/args.txt'
        with open(args_file, 'r') as file:
            for line in file:
                key, value = parse_args_line(line.strip())
                if key in ['pixel_wise', 'channel_wise', 'poly_weight_factors']:
                    setattr(args, key, value)
    elif args.resume:
        args_file = args.resume_dir + '/args.txt'
        with open(args_file, 'r') as file:
            for line in file:
                key, value = parse_args_line(line.strip())
                if hasattr(args, key) and not key.startswith('resume') and not key.startswith('reload'):
                    setattr(args, key, value)

    args.total_gpus = torch.cuda.device_count()

    mp.spawn(process, nprocs=args.total_gpus, args=(args, ))
    