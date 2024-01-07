import os
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import argparse
from torch.utils.tensorboard import SummaryWriter
import ast
from model import ResNet18Poly, general_relu_poly, convert_to_bf16_except_bn, find_submodule, copy_parameters
from model_relu_avg import ResNet18ReluAvg
from model_relu import ResNet18Relu
import numpy as np
import re
from ddp_training import ddp_train, ddp_test, single_test
from utils import Lookahead, MaskProvider
from datetime import datetime
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist

def adjust_learning_rate(optimizer, epoch, init_lr, lr_step_size, lr_gamma):
    lr = init_lr * (lr_gamma ** (epoch // lr_step_size))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def process(pn, args):
    torch.cuda.set_device(pn)
    process_group = torch.distributed.init_process_group(backend="nccl", init_method='env://', world_size=args.total_gpus, rank=pn)

    torch.manual_seed(10)
    torch.cuda.manual_seed_all(10)

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
            transforms.RandomResizedCrop(224, antialias=True),
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
        transforms.Resize(256, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    if args.train_subset:
        trainset = torchvision.datasets.ImageFolder(root='/home/uconn/xiexi/poly_replace/subset2' , transform=transform_train)
    else:
        trainset = torchvision.datasets.ImageFolder(root='/home/uconn/dataset/imagenet/train' , transform=transform_train)
    

    train_sampler = DistributedSampler(trainset, num_replicas=args.total_gpus, rank=pn)
    trainloader = torch.utils.data.DataLoader(trainset, sampler=train_sampler, batch_size=args.batch_size_train, num_workers=args.num_train_loader_workers, pin_memory=True, shuffle=False)

    
    testset = torchvision.datasets.ImageFolder(
        root='/home/uconn/dataset/imagenet/val', transform=transform_test)
    
    test_sampler = DistributedSampler(testset, num_replicas=args.total_gpus, rank=pn)
    testloader = torch.utils.data.DataLoader(testset, sampler=test_sampler, batch_size=args.batch_size_test, num_workers=args.num_test_loader_workers, pin_memory=True, shuffle=False)
    testloader_single = torch.utils.data.DataLoader(testset, batch_size = args.batch_size_test, shuffle=False, num_workers=args.num_test_loader_workers, pin_memory=True)

    model = ResNet18Poly(args.channel_wise, args.pixel_wise, args.poly_weight_inits, args.poly_weight_factors, relu2_extra_factor=1)

    dummy_input = torch.rand(1, 3, 224, 224) 
    model((dummy_input, 0))

    # pretrain_model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)

    model_relu_avg = ResNet18ReluAvg(True)
    checkpoint_relu_avg = torch.load("/home/uconn/xiexi/HE_transfer_learning/runs20240101163300/checkpoint_epoch_79.pth")
    model_relu_avg.load_state_dict(checkpoint_relu_avg['model_state_dict'])
    model_relu_avg = model_relu_avg.cuda()

    # model_relu = ResNet18Relu()
    # copy_parameters(pretrain_model, model_relu)
    # model_relu = model_relu.cuda()

    # model_relu_avg = model_relu

    checkpoint = None

    def find_latest_epoch(resume_dir):
        max_epoch = -1
        for filename in os.listdir(resume_dir):
            match = re.match(r'checkpoint_epoch_(\d+)\.pth', filename)
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
                start_epoch = args.resume_epoch + 1
            checkpoint_path = os.path.join(args.resume_dir, f'checkpoint_epoch_{start_epoch - 1}.pth')
        else:
            checkpoint_path = os.path.join(args.resume_dir, f'checkpoint_epoch_{args.resume_epoch}.pth')

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        else:
            print(f"No checkpoint found at {checkpoint_path}")
    else:
        model.load_state_dict(checkpoint_relu_avg['model_state_dict'], strict=False)
        # copy_parameters(pretrain_model, model) 
        # model.load_state_dict(torch.load("default_poly_model.pth"))
        # initialize_resnet(model)


    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # tmp_test_acc = 0
    # if pn == 0:
    #     single_test(args, testloader_single, model.cuda(), 0, tmp_test_acc, 1)
    # dist.barrier()
    
    model = model.cuda()
    model.eval()

    for param in model_relu_avg.parameters():
        param.requires_grad = False
    
    model = DistributedDataParallel(model, device_ids=[pn])

    # if args.bf16:
    #     model = convert_to_bf16_except_bn(model)
    #     model_relu = convert_to_bf16_except_bn(model_relu)

    if args.optim == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr = args.lr, weight_decay=args.w_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum=args.momentum, weight_decay=args.w_decay)

    if args.lookahead:
        optimizer = Lookahead(optimizer)

    if args.lr_anneal is None or args.lr_anneal == "None":
        lr_scheduler = None
    elif args.lr_anneal == "cos":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_epochs)

    assert not(lr_scheduler is not None and args.lr_step_size > 0), "should not use both lr_anneal and lr_step"

    # scaler = torch.cuda.amp.GradScaler(enabled=args.bf16 or args.fp16)

    current_datetime = datetime.now().strftime('%Y%m%d%H%M%S')
    if args.log_root:
        log_root = args.log_root
    else:
        log_root = 'runs' + current_datetime
    if args.resume and args.resume_dir:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        assert checkpoint['epoch'] + 1 == start_epoch
        if args.resume_log_root:
            log_root = args.resume_dir    

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
    #     print(start_epoch)
    #     for arg, value in vars(args).items():
    #         print(f"{arg}: {value}")

    test_acc = 0
    best_acc = 0
    if args.reload or args.resume:
        if start_epoch == 0:
            _test_epoch = 0
        else:
            _test_epoch = start_epoch - 1
        test_acc, best_acc = ddp_test(args, testloader, model, _test_epoch, best_acc, mask_provider.get_mask(_test_epoch), writer, pn)

    for epoch in range(start_epoch, args.total_epochs):
        if args.lr_step_size > 0:
            adjust_learning_rate(optimizer, epoch, args.lr, args.lr_step_size, args.lr_gamma)

        train_sampler.set_epoch(epoch)
        mask = mask_provider.get_mask(epoch)

        if pn == 0:
            print("mask = ", mask)
            writer.add_scalar('Mask value', mask, epoch)
            if args.pixel_wise:
                if isinstance(model, DistributedDataParallel):
                    total_elements, relu_elements = model.module.get_relu_density(mask)
                else:
                    total_elements, relu_elements = model.get_relu_density(mask)
                print(f"total_elements {total_elements}, relu_elements {relu_elements}, density = {relu_elements/total_elements}")
        
        omit_fms = 0
        train_acc = ddp_train(args, trainloader, model, model_relu_avg, optimizer, epoch, mask, writer, pn, omit_fms)

        if mask < 0.01 or False:
            test_acc, best_acc = ddp_test(args, testloader, model, epoch, best_acc, mask, writer, pn)

        if lr_scheduler is not None:
            lr_scheduler.step()
        
        if pn == 0:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()
            if lr_scheduler is None:
                lr_scheduler_state_dict = None
            else:
                lr_scheduler_state_dict = lr_scheduler.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler_state_dict,
            }, f"{log_dir}/checkpoint_epoch_{epoch}.pth")
            with open(f"{log_dir}/acc.txt", 'a') as file:
                file.write(f'{epoch} train {train_acc*100:.2f} test {test_acc*100:.2f} best {best_acc*100:.2f}\n')

    if writer is not None:
        writer.close()

if __name__ == "__main__":

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29505'

    parser = argparse.ArgumentParser(description='Fully poly replacement on ResNet for ImageNet')
    parser.add_argument('--id', default=0, type=int)
    parser.add_argument('--total_epochs', default=200, type=int)
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--w_decay', default=1e-4, type=float, help='w decay rate')
    parser.add_argument('--optim', type=str, default='adamw', choices = ['sgd', 'adamw'])
    parser.add_argument('--batch_size_train', type=int, default=100, help='Batch size for training')
    parser.add_argument('--batch_size_test', type=int, default=500, help='Batch size for testing')
    parser.add_argument('--data_augment', type=ast.literal_eval, default=True)
    parser.add_argument('--train_subset', type=ast.literal_eval, default=False, help='if train on the 1/13 subset of ImageNet or the full ImageNet')
    parser.add_argument('--pixel_wise', type=ast.literal_eval, default=True, help='if use pixel-wise poly replacement')
    parser.add_argument('--channel_wise', type=ast.literal_eval, default=True, help='if use channel-wise relu_poly class')
    parser.add_argument('--poly_weight_inits', nargs=3, type=float, default=[0, 1, 0], help='relu_poly weights initial values')
    parser.add_argument('--poly_weight_factors', nargs=3, type=float, default=[0.03, 1, 0.1], help='adjust the learning rate of the three weights in relu_poly')
    parser.add_argument('--mask_decrease', type=str, default='1-sinx', choices = ['0', '1-sinx', 'e^(-x/10)', 'linear'], help='how the relu replacing mask decreases')
    parser.add_argument('--mask_epochs', default=10, type=int, help='the epoch that the relu replacing mask will decrease to 0')
    parser.add_argument('--loss_fm_type', type=str, default='at', choices = ['at', 'mse', 'custom_mse'], help='the type for the feature map loss')
    parser.add_argument('--loss_fm_factor', default=100, type=float, help='the factor of the feature map loss, set to 0 to disable')
    parser.add_argument('--loss_ce_factor', default=1, type=float, help='the factor of the cross-entropy loss, set to 0 to disable')
    parser.add_argument('--loss_kd_factor', default=0.1, type=float, help='the factor of the knowledge distillation loss, set to 0 to disable')
    parser.add_argument('--lookahead', type=ast.literal_eval, default=True, help='if enable look ahead for the optimizer')
    parser.add_argument('--lr_anneal', type=str, default='None', choices = ['None', 'cos'])
    parser.add_argument('--lr_step_size', type=int, default=100, help="decrease lr every step-size epochs")
    parser.add_argument('--lr_gamma', type=float, default=0.3, help="decrease lr by a factor of lr-gamma")

    parser.add_argument('--bf16', type=ast.literal_eval, default=False, help='if enable training with bf16 precision')
    parser.add_argument('--fp16', type=ast.literal_eval, default=False, help='if enable training with float16 precision')
    
    parser.add_argument('--num_train_loader_workers', type=int, default=9)
    parser.add_argument('--num_test_loader_workers', type=int, default=5)
    parser.add_argument('--pbar', type=ast.literal_eval, default=True)
    parser.add_argument('--log_root', type=str)

    parser.add_argument('--resume', type=ast.literal_eval, default=False)
    parser.add_argument('--resume_dir', type=str)
    parser.add_argument('--resume_epoch', type=int, default=None)
    parser.add_argument('--resume_log_root', type=ast.literal_eval, default=False)

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
                    if not key.startswith('batch_size') and not key == 'lr' and not key.startswith('num_train_loader'):
                        setattr(args, key, value)

    args.total_gpus = torch.cuda.device_count()

    mp.spawn(process, nprocs=args.total_gpus, args=(args, ))
    