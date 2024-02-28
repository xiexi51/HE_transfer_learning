import os
import torch
import torch.optim as optim
import argparse
from torch.utils.tensorboard import SummaryWriter
import ast
from model import ResNet18Poly, general_relu_poly, convert_to_bf16_except_bn, find_submodule, copy_parameters
import numpy as np
import re
from ddp_vanilla_training import ddp_vanilla_train, ddp_test, single_test
from utils import MaskProvider
from datetime import datetime
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist
from utils_dataset import build_imagenet_dataset
from timm.data import Mixup
# from vanillanet_deploy import vanillanet_6_poly
from vanillanet import vanillanet_5

import timm
from timm.utils import ModelEma
from optim_factory import create_optimizer
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy
import math


def adjust_learning_rate(optimizer, epoch, init_lr, lr_step_size, lr_gamma):
    lr = init_lr * (lr_gamma ** (epoch // lr_step_size))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def process(pn, args):
    torch.cuda.set_device(pn)
    process_group = torch.distributed.init_process_group(backend="nccl", init_method='env://', world_size=args.total_gpus, rank=pn)

    torch.manual_seed(10)
    torch.cuda.manual_seed_all(10)
    
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    mask_provider = MaskProvider(args.mask_decrease, args.mask_epochs)

    if pn == 0:
        print("gpu count =", torch.cuda.device_count())
    
    trainset = build_imagenet_dataset(True, args)
    train_sampler = DistributedSampler(trainset, num_replicas=args.total_gpus, rank=pn)
    trainloader = torch.utils.data.DataLoader(trainset, sampler=train_sampler, batch_size=args.batch_size_train, num_workers=args.num_train_loader_workers, pin_memory=True, shuffle=False, drop_last=True)
    testset = build_imagenet_dataset(False, args) 
    test_sampler = DistributedSampler(testset, num_replicas=args.total_gpus, rank=pn)
    single_test_sampler = torch.utils.data.SequentialSampler(testset)
    testloader = torch.utils.data.DataLoader(testset, sampler=test_sampler, batch_size=args.batch_size_test, num_workers=args.num_test_loader_workers, pin_memory=True, shuffle=False, drop_last=False)
    single_testloader = torch.utils.data.DataLoader(testset, sampler=single_test_sampler, batch_size=args.batch_size_test, num_workers=args.num_test_loader_workers, drop_last=False)

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=1000 )

    # model = vanillanet_6_poly(args.poly_weight_inits, args.poly_weight_factors)

    # model_t = vanillanet_6_poly([0, 0, 0], [0, 0, 0])
        
    model = vanillanet_5()

    model_t = None

    # dummy_input = torch.rand(1, 3, 224, 224) 
    # model((dummy_input, 0))
    
    if pn == 0:
        if args.deploy:
            print("create deploy model")
        else:
            print("create full model")

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

    if args.reload:
        checkpoint_path = args.reload_dir
    elif args.resume:
        if args.resume_epoch is None:
            start_epoch = find_latest_epoch(args.resume_dir) + 1
        else:
            start_epoch = args.resume_epoch + 1
        checkpoint_path = os.path.join(args.resume_dir, f'checkpoint_epoch_{start_epoch - 1}.pth')

    if args.reload or args.resume:
        if checkpoint_path and os.path.exists(checkpoint_path):
            if pn == 0:
                print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            # model.load_state_dict(checkpoint['model'], strict=True)
            
            # model.load_state_dict(checkpoint, strict=True)
            
            # model_t.load_state_dict(checkpoint, strict=False)
        else:
            if pn == 0:
                print(f"No checkpoint found at {checkpoint_path}")
    
    
    # for param in model.cls.parameters():
    #     param.requires_grad = False

    model = model.cuda()

    if model_t is not None:
        model_t = model_t.cuda()

    if args.switch_to_deploy:
        raise ValueError("don't switch to deploy here")
        model.switch_to_deploy()
        if pn == 0:
            torch.save(model.state_dict(), "/home/uconn/xiexi/HE_transfer_learning/runs_vanilla6/deploy_vanilla6_acc74.pth")
        # model_ckpt = dict()
        # model_ckpt['model'] = model.state_dict()
        # torch.save(model_ckpt, args.switch_to_deploy)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = []
        for ema_decay in args.model_ema_decay:
            model_ema.append(
                ModelEma(model, decay=ema_decay, device='cpu' if args.model_ema_force_cpu else '', resume='')
            )
        if pn == 0:
            print("Using EMA with decay = %s" % args.model_ema_decay)

    model = DistributedDataParallel(model, device_ids=[pn])

    # if args.bf16:
    #     model = convert_to_bf16_except_bn(model)
    #     model_relu = convert_to_bf16_except_bn(model_relu)

    assigner = None

    optimizer = create_optimizer(
        args, model.module, skip_list=None,
        get_num_layer=assigner.get_layer_id if assigner is not None else None, 
        get_layer_scale=assigner.get_scale if assigner is not None else None)

    if args.lookahead:
        optimizer = timm.optim.lookahead.Lookahead(optimizer)
    
    if args.lr_anneal is None or args.lr_anneal == "None":
        lr_scheduler = None
    else:
        if args.lr_anneal_tmax is None:
            args.lr_anneal_tmax = args.total_epochs
        if args.lr_anneal == "cos":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.lr_anneal_tmax)

    assert not(lr_scheduler is not None and args.lr_step_size > 0), "should not use both lr_anneal and lr_step"

    if mixup_fn is not None:
        if args.bce_loss:
            criterion_ce = BinaryCrossEntropy(target_threshold=args.bce_target_thresh)
        else:
            # smoothing is handled with mixup label transform
            criterion_ce = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion_ce = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion_ce = torch.nn.CrossEntropyLoss()

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

    if args.model_ema and args.model_ema_eval:
        max_accuracy_ema = 0.0
        max_accuracy_ema_epoch = 0
        best_ema_decay = args.model_ema_decay[0]

    if args.reload or args.resume:
        if start_epoch == 0:
            _test_epoch = 0
        else:
            _test_epoch = start_epoch - 1

        _mask = mask_provider.get_mask(0)
        # if pn == 0:            
        #     total_elements, relu_elements = model.module.get_relu_density(_mask)    
        #     print(f"total_elements {total_elements}, relu_elements {relu_elements}, relu density = {relu_elements/total_elements}")
        
        # test_acc = ddp_test(args, testloader, model, _test_epoch, best_acc, _mask, writer, pn)

        # test_acc = ddp_test(args, testloader, model, _test_epoch, best_acc, None, writer, pn)

        # return

    # if pn == 0:
    #     print("test model_t:")
    #     _, _ = single_test(args, single_testloader, model_t, 0, 0, -1)
    # dist.barrier()
        
    # start_epoch = 300
    # args.total_epochs = 500

    recent_checkpoints = []

    for epoch in range(start_epoch, args.total_epochs):
        if args.lr_step_size > 0:
            adjust_learning_rate(optimizer, epoch, args.lr, args.lr_step_size, args.lr_gamma)

        train_sampler.set_epoch(epoch)
        mask = mask_provider.get_mask(epoch)

        mask = None 

        if epoch < args.decay_epochs:
            if args.decay_linear:
                act_learn = epoch / args.decay_epochs * 1.0
            else:
                act_learn = 0.5 * (1 - math.cos(math.pi * epoch / args.decay_epochs)) * 1.0
        else:
            act_learn = 1
        
        model.module.change_act(act_learn)

        # mask = 0

        if pn == 0:
            if mask is not None:
                print("mask = ", mask)
                writer.add_scalar('Mask value', mask, epoch)
            # if args.pixel_wise:
            #     if isinstance(model, DistributedDataParallel):
            #         total_elements, relu_elements = model.module.get_relu_density(mask)
            #     else:
            #         total_elements, relu_elements = model.get_relu_density(mask)
            #     print(f"total_elements {total_elements}, relu_elements {relu_elements}, density = {relu_elements/total_elements}")
        
        omit_fms = 0
        train_acc = ddp_vanilla_train(args=args, trainloader=trainloader, model_s=model, model_t=model_t, optimizer=optimizer, epoch=epoch, 
                                      mask=mask, writer=writer, pn=pn, omit_fms=omit_fms, mixup_fn=mixup_fn, criterion_ce=criterion_ce, 
                                      max_norm=None, update_freq=args.update_freq, model_ema=None, act_learn=act_learn)

        if True or mask < 0.01:
            test_acc = ddp_test(args, testloader, model, epoch, best_acc, mask, writer, pn)

        if pn == 0:
            with open(f"{log_dir}/acc.txt", 'a') as file:
                file.write(f"{epoch} train {train_acc*100:.2f} test {test_acc*100:.2f} best {best_acc*100:.2f} Lr {optimizer.param_groups[0]['lr']:.2e} act_learn {act_learn:.2f}\n")

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
            
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler_state_dict,
                    'best_acc': best_acc,
                }, f"{log_dir}/best_model.pth")

            checkpoint_path = f"{log_dir}/checkpoint_epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler_state_dict,
                'test_acc': test_acc,
            }, checkpoint_path)

            recent_checkpoints.append(checkpoint_path)

            if args.keep_checkpoints != -1:
                if len(recent_checkpoints) > args.keep_checkpoints:
                    oldest_checkpoint = recent_checkpoints.pop(0)
                    os.remove(oldest_checkpoint)

        dist.barrier()
            

    if writer is not None:
        writer.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fully poly replacement on ResNet for ImageNet')

    # general settings
    parser.add_argument('--id', default=0, type=int)
    parser.add_argument('--batch_size_train', type=int, default=200, help='Batch size for training')
    parser.add_argument('--batch_size_test', type=int, default=200, help='Batch size for testing')
    parser.add_argument('--num_train_loader_workers', type=int, default=6)
    parser.add_argument('--num_test_loader_workers', type=int, default=5)
    parser.add_argument('--pbar', type=ast.literal_eval, default=True)
    parser.add_argument('--log_root', type=str)

    parser.add_argument("--master_port", type=int, default=None)

    parser.add_argument('--switch_to_deploy', type=ast.literal_eval, default=False)

    parser.add_argument('--update_freq', default=1, type=int, help='gradient accumulation steps')

    
    # imagenet dataset arguments
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT', help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME', help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic', help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT', help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')

    # Mixup params
    parser.add_argument('--mixup', type=float, default=0.8, help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0, help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None, help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0, help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5, help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch', help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')

    # model params
    parser.add_argument('--act_num', default=3, type=int)
    parser.add_argument('--drop_rate', type=float, default=0, metavar='PCT', help='Drop rate (default: 0.0)')
    parser.add_argument('--deploy', type=ast.literal_eval, default=False)


    # training params
    parser.add_argument('--total_epochs', default=300, type=int)
    parser.add_argument('--lr', default=5e-3, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--w_decay', default=1e-4, type=float, help='w decay rate')
    parser.add_argument('--optim', type=str, default='adamw', choices = ['sgd', 'adamw', 'lamb'])
    parser.add_argument('--decay_epochs', default=100, type=int, help='for deep training strategy')
    parser.add_argument('--decay_linear', type=ast.literal_eval, default=True, help='cos/linear for decay manner')

    parser.add_argument('--use_amp', type=ast.literal_eval, default=False, help="Use PyTorch's AMP (Automatic Mixed Precision) or not")
    
    parser.add_argument('--bce_loss', type=ast.literal_eval, default=False, help='Enable BCE loss w/ Mixup/CutMix use.')
    parser.add_argument('--bce_target_thresh', type=float, default=None, help='Threshold for binarizing softened BCE targets (default: None, disabled)')

    # EMA related parameters
    parser.add_argument('--model_ema', type=ast.literal_eval, default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, nargs='+')
    parser.add_argument('--model_ema_force_cpu', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--model_ema_eval', type=ast.literal_eval, default=False, help='Using ema to eval during training.')

    parser.add_argument('--train_subset', type=ast.literal_eval, default=False, help='if train on the 1/13 subset of ImageNet or the full ImageNet')
    parser.add_argument('--pixel_wise', type=ast.literal_eval, default=True, help='if use pixel-wise poly replacement')
    parser.add_argument('--channel_wise', type=ast.literal_eval, default=True, help='if use channel-wise relu_poly class')
    parser.add_argument('--poly_weight_inits', nargs=3, type=float, default=[0, 0.0, 0], help='relu_poly weights initial values')
    parser.add_argument('--poly_weight_factors', nargs=3, type=float, default=[0.05, 0.5, 0.1], help='adjust the learning rate of the three weights in relu_poly')
    parser.add_argument('--mask_decrease', type=str, default='1-sinx', choices = ['0', '1-sinx', 'e^(-x/10)', 'linear'], help='how the relu replacing mask decreases')
    parser.add_argument('--mask_epochs', default=6, type=int, help='the epoch that the relu replacing mask will decrease to 0')
    parser.add_argument('--loss_fm_type', type=str, default='at', choices = ['at', 'mse', 'custom_mse'], help='the type for the feature map loss')
    parser.add_argument('--loss_fm_factor', default=0, type=float, help='the factor of the feature map loss, set to 0 to disable')
    parser.add_argument('--loss_ce_factor', default=1, type=float, help='the factor of the cross-entropy loss, set to 0 to disable')
    parser.add_argument('--loss_kd_factor', default=0, type=float, help='the factor of the knowledge distillation loss, set to 0 to disable')
    parser.add_argument('--lookahead', type=ast.literal_eval, default=True, help='if enable look ahead for the optimizer')
    parser.add_argument('--lr_anneal', type=str, default='cos', choices = ['None', 'cos'])
    parser.add_argument('--lr_anneal_tmax', type=int, default=None)
    parser.add_argument('--lr_step_size', type=int, default=0, help="decrease lr every step-size epochs")
    parser.add_argument('--lr_gamma', type=float, default=0.1, help="decrease lr by a factor of lr-gamma")

    # parser.add_argument('--bf16', type=ast.literal_eval, default=False, help='if enable training with bf16 precision')
    # parser.add_argument('--fp16', type=ast.literal_eval, default=False, help='if enable training with float16 precision')
    
    
    parser.add_argument('--resume', type=ast.literal_eval, default=False)
    parser.add_argument('--resume_dir', type=str)
    parser.add_argument('--resume_epoch', type=int, default=None)
    parser.add_argument('--resume_log_root', type=ast.literal_eval, default=False)

    parser.add_argument('--reload', type=ast.literal_eval, default=False)
    parser.add_argument('--reload_file', type=str)

    parser.add_argument('--keep_checkpoints', type=int, default=-1, help="Specify the number of recent checkpoints to keep. Set to -1 to keep all checkpoints.")

    # parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)

    args = parser.parse_args()

    os.environ['NCCL_DEBUG'] = 'WARN'
    os.environ['MASTER_ADDR'] = 'localhost'
    if args.master_port is None:
        os.environ['MASTER_PORT'] = '9501'
    else:
        os.environ['MASTER_PORT'] = str(args.master_port)

    print("master port = " + os.environ['MASTER_PORT'])

    def parse_args_line(line):
        key, value = line.split(": ", 1)
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
        return key, value


    if args.reload:
        args_file = args.reload_dir + '/args.txt'
        try:
            with open(args_file, 'r') as file:
                for line in file:
                    key, value = parse_args_line(line.strip())
                    if key in ['pixel_wise', 'channel_wise', 'poly_weight_factors']:
                        setattr(args, key, value)
        except Exception:
            print(f"Warning: Unable to open {args_file}. Continuing without reloading arguments.")

    elif args.resume:
        args_file = args.resume_dir + '/args.txt'
        with open(args_file, 'r') as file:
            for line in file:
                key, value = parse_args_line(line.strip())
                if hasattr(args, key) and not key.startswith('resume') and not key.startswith('reload'):
                    if (not key.startswith('batch_size') and not key == 'lr' and not key.startswith('num_train_loader') 
                        and not key.startswith('num_test_loader') and not key == 'total_epochs' and not key == 'master_port'):
                        setattr(args, key, value)

    args.total_gpus = torch.cuda.device_count()

    mp.spawn(process, nprocs=args.total_gpus, args=(args, ))
    