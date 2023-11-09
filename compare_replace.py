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
from torch.optim.optimizer import Optimizer
from collections import defaultdict
from datetime import datetime

# from utils import progress_bar

parser = argparse.ArgumentParser(description='Transfer from ImageNet pretrain to CIFAR10')

parser.add_argument('--id', default=0, type=int)
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--lr', default=0.00001, type=float, help='learning rate')
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
parser.add_argument('--log_root', type=str)
args = parser.parse_args()

torch.manual_seed(10)
torch.cuda.manual_seed_all(10)

def print_available_gpus():
    num_devices = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_devices}")

    for i in range(num_devices):
        device_name = torch.cuda.get_device_name(i)
        device_capability = torch.cuda.get_device_capability(i)
        print(f"  GPU {i}: {device_name}, Capability: {device_capability}")

class Lookahead(Optimizer):
    def __init__(self, base_optimizer, alpha=0.5, k=6):
        # NOTE super().__init__() not called on purpose
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        defaults = dict(lookahead_alpha=alpha, lookahead_k=k, lookahead_step=0)
        self._base_optimizer = base_optimizer
        self.param_groups = base_optimizer.param_groups
        self.defaults = base_optimizer.defaults
        self.defaults.update(defaults)
        self.state = defaultdict(dict)
        # manually add our defaults to the param groups
        for name, default in defaults.items():
            for group in self._base_optimizer.param_groups:
                group.setdefault(name, default)

    @torch.no_grad()
    def update_slow(self, group):
        for fast_p in group["params"]:
            if fast_p.grad is None:
                continue
            param_state = self._base_optimizer.state[fast_p]
            if 'lookahead_slow_buff' not in param_state:
                param_state['lookahead_slow_buff'] = torch.empty_like(fast_p)
                param_state['lookahead_slow_buff'].copy_(fast_p)
            slow = param_state['lookahead_slow_buff']
            slow.add_(fast_p - slow, alpha=group['lookahead_alpha'])
            fast_p.copy_(slow)

    def sync_lookahead(self):
        for group in self._base_optimizer.param_groups:
            self.update_slow(group)

    @torch.no_grad()
    def step(self, closure=None):
        loss = self._base_optimizer.step(closure)
        for group in self._base_optimizer.param_groups:
            group['lookahead_step'] += 1
            if group['lookahead_step'] % group['lookahead_k'] == 0:
                self.update_slow(group)
        return loss

    def state_dict(self):
        return self._base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self._base_optimizer.load_state_dict(state_dict)
        self.param_groups = self._base_optimizer.param_groups

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
    
    # writer = SummaryWriter(log_dir=log_dir)

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
                    fx = model(x, mask)
                else:
                    fx = model(x)
            top1, top5 = accuracy(fx, y, topk=(1, 5))
            top1_total += top1[0] * x.size(0)
            top5_total += top5[0] * x.size(0)
            total += x.size(0)
            pbar.set_postfix_str(f"Top-1 Acc {100*top1_total/total:.2f}%, Top-5 Acc {100*top5_total/total:.2f}%")
            break

        test_acc = (top1_total / total).item()
        
        if test_acc > best_acc:
            best_acc = test_acc

        print('Epoch', epoch, 'Test Acc:', test_acc, 'Test Best:', best_acc)
        return best_acc

    def test_save(model, epoch, mask, save_dir):
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
                    fx = model.forward_save(x, mask, save_dir)
                else:
                    fx = model.forward_save(x, save_dir)
            top1, top5 = accuracy(fx, y, topk=(1, 5))
            top1_total += top1[0] * x.size(0)
            top5_total += top5[0] * x.size(0)
            total += x.size(0)
            pbar.set_postfix_str(f"Top-1 Acc {100*top1_total/total:.2f}%, Top-5 Acc {100*top5_total/total:.2f}%")
            break

        test_acc = (top1_total / total).item()
        print('Epoch', epoch, 'Test Acc:', test_acc)
        return best_acc

    def save_feature_map(ep, mask, save_dir):
        model = ResNet18Poly()
        # mask = 1 - ((ep + 1) / 50)*0.7
        # if mask < 0:
        #     mask = 0
        print("mask = ", mask)
        pth_dir = "./runs_gpu411/0_ep100_lr0.0001_wd0.0_optadamw_btrain500_btest500_hdim0_dp0_b1False_b2False_reFalse_unfreezenone_augFalse/"
        model.load_state_dict(torch.load(pth_dir + f"model_epoch_{ep}.pth"))
        model = model.cuda()
        # save_dir = f"feature_map_{ep}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        test_save(model, ep, mask, save_dir)
    
    ep1 = 70
    mask1 = 1 - ((ep1 + 1) / 50)*0.7

    save_feature_map(70, 0, f"feature_map_70_mask_0")
    # save_feature_map(71)


    

if __name__ == "__main__":
    # print_available_gpus()

    main(args)