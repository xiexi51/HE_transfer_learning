import torch
from tqdm import tqdm
import torch.nn as nn
from utils import custom_mse_loss, at_loss, irg_loss, SoftTarget, accuracy
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import warnings
import math
from argparse import Namespace
from typing import Iterable, Optional, List
from torch.utils.tensorboard import SummaryWriter
from timm.data import Mixup
from timm.utils import ModelEma
from vanillanet_deploy_poly import VanillaNet_deploy_poly
from typing import Tuple

def set_forward_with_fms(model, if_forward_with_fms):
    if isinstance(model, DistributedDataParallel):
        model.module.if_forward_with_fms = if_forward_with_fms
    else:
        model.if_forward_with_fms = if_forward_with_fms


def ddp_vanilla_train(args: Namespace, trainloader: Iterable, model_s: torch.nn.Module, model_t: torch.nn.Module, optimizer: torch.optim.Optimizer, 
              epoch: int, mask: Tuple[float, float], writer: SummaryWriter, world_pn: int, omit_fms: int, mixup_fn: Mixup, criterion_ce: torch.nn.Module, 
              max_norm: float, update_freq: int, model_ema: List[ModelEma], act_learn: float):
    #model_s.eval()

    model_s.train()

    if model_t is not None:
        model_t.eval()

    train_loss = 0
    train_loss_kd = 0
    train_loss_ce = 0
    train_loss_fm = 0

    total_l2_norm = 0.0
    param_count = 0

    if args.pbar and world_pn == 0:
        if act_learn is not None:
            desc = f"{epoch} Lr{optimizer.param_groups[0]['lr']:.2e} act{act_learn:.2f}"
        else:
            desc = f"{epoch} Lr{optimizer.param_groups[0]['lr']:.2e}"
        pbar = tqdm(trainloader, total=len(trainloader), desc=desc, ncols=140)
    else:
        pbar = trainloader

    top1_total = 0
    top5_total = 0
    total = 0

    reduced_total = 0
    reduced_top1_total = 0
    reduced_top5_total = 0

    if args.loss_fm_type == "mse":
        loss_fm_fun = nn.MSELoss()
    elif args.loss_fm_type == "custom_mse":
        loss_fm_fun = custom_mse_loss
    else:
        loss_fm_fun = at_loss
    
    criterion_kd = SoftTarget(4.0).cuda()

    total_batches = len(pbar)
    effective_batches = total_batches - total_batches % update_freq
    effective_num = effective_batches // update_freq
    # assert effective_num == 1281167 // args.effective_batch_size

    if mask is not None:
        if args.mask_mini_batch:
            mask_iter = (mask[1] - mask[0]) / effective_num
            mask_current = mask[0] + mask_iter
        else:
            mask_iter = 0
            mask_current = mask[1]
    else:
        mask_iter = 0
        mask_current = 0

    accumulated_batches = 0

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    optimizer.zero_grad()

    if args.bf16:
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16

    for iter, (x, y) in enumerate(pbar):
        if iter >= effective_batches:
            break

        x, y = x.cuda(), y.cuda()

        # if args.bf16:
        #     x = x.to(dtype=torch.bfloat16)

        if mixup_fn is not None:
            x, y = mixup_fn(x, y)
        
        with torch.cuda.amp.autocast(enabled=args.use_amp, dtype=amp_dtype):
            if model_t is not None and (args.loss_fm_factor > 0 or args.loss_kd_factor > 0):
                with torch.no_grad():
                    set_forward_with_fms(model_t, True)
                    out_t, fms_t = model_t((x, -1))

            if args.loss_fm_factor > 0:
                set_forward_with_fms(model_s, True)
                out_s, fms_s = model_s((x, mask_current))
            else:
                set_forward_with_fms(model_s, False)
                if mask is not None:
                    out_s = model_s((x, mask_current))
                else:
                    out_s = model_s(x)
        
            loss = 0

            if args.loss_fm_factor > 0:
                loss_fm = sum(loss_fm_fun(x, y) for x, y in zip(fms_s[omit_fms:], fms_t[omit_fms:])) * args.loss_fm_factor
                loss += loss_fm
                train_loss_fm += loss_fm.item()
            if args.loss_kd_factor > 0:
                loss_kd = criterion_kd(out_s, out_t) * args.loss_kd_factor
                loss += loss_kd
                train_loss_kd += loss_kd.item()

            if args.loss_ce_factor > 0:
                loss_ce = criterion_ce(out_s, y) * args.loss_ce_factor
                loss += loss_ce
                train_loss_ce += loss_ce.item()

        loss /= update_freq
        # assert math.isfinite(loss)
        scaler.scale(loss).backward(create_graph=hasattr(optimizer, 'is_second_order') and optimizer.is_second_order)
        accumulated_batches += 1
        train_loss += loss.item()

        if mixup_fn is not None:                
            top1_num = (out_s.argmax(dim=1) == y.argmax(dim=1)).float().sum().item()
        else:
            top1_num = (out_s.argmax(dim=1) == y).float().sum().item()

        top1_total += top1_num

        total += x.size(0)

        reduced_total = torch.tensor(total, dtype=torch.float).cuda().detach()
        reduced_top1_total = torch.tensor(top1_total, dtype=torch.float).cuda().detach()
        
        # dist.reduce(reduced_total, dst=0)
        # dist.reduce(reduced_top1_total, dst=0)

        dist.all_reduce(reduced_total)
        dist.all_reduce(reduced_top1_total)

        reduced_total = reduced_total.cpu().numpy()
        reduced_top1_total = reduced_top1_total.cpu().numpy()
        
        if args.pbar and world_pn == 0:
            # print(total, top1_total, top5_total)
            # print(reduced_total, reduced_top1_total, reduced_top5_total)
            pbar.set_postfix_str(f"L{train_loss/total:.2e},fm{train_loss_fm/total:.2e},kd{train_loss_kd/total:.2e},ce{train_loss_ce/total:.2e}, 1a {100*reduced_top1_total/reduced_total:.1f}, 5a -, m{mask_current:.4f}")

        if accumulated_batches == update_freq:
            mask_current += mask_iter
            
            for name, param in model_s.module.named_parameters():
                if name.endswith('.relu.weight') and param.grad is not None:
                    norm = torch.norm(param.grad.data, p=2).item()
                    if args.relu_grad_max_norm != -1 and norm > args.relu_grad_max_norm :
                        param.grad.data = param.grad.data * args.relu_grad_max_norm / norm
                        norm = args.relu_grad_max_norm
                    total_l2_norm += norm
                    param_count += 1

            scaler.step(optimizer) 
            scaler.update() 

            if args.clamp_poly_weight_ge0:
                for name, param in model_s.module.named_parameters():
                    if name.endswith('.relu.weight'):
                        with torch.no_grad():
                            param.data[:, 0] = torch.clamp(param.data[:, 0], min=0)
                            param.data[:, 1] = torch.clamp(param.data[:, 1], min=0)        

            optimizer.zero_grad() 
            accumulated_batches = 0 
            if model_ema is not None:
                for iter_model_ema in model_ema:
                    iter_model_ema.update(model_s)
                    for i in range(len(iter_model_ema.ema.stages)):
                        if hasattr(iter_model_ema.ema.stages[i], 'act_learn'):
                            iter_model_ema.ema.stages[i].act_learn = model_s.module.stages[i].act_learn
                        if hasattr(iter_model_ema.ema, 'act_learn'):
                            iter_model_ema.ema.act_learn = model_s.module.act_learn

    # print(mask_current, mask[1])

    train_acc = (reduced_top1_total / reduced_total).item()

    if writer is not None:
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/train', train_loss/total, epoch)
        writer.add_scalar('Loss_fm/train', train_loss_fm/total, epoch)
        writer.add_scalar('Loss_kd/train', train_loss_kd/total, epoch)
        writer.add_scalar('Loss_ce/train', train_loss_ce/total, epoch)

    avg_l2_norm = total_l2_norm / param_count if param_count > 0 else 0
    return train_acc, avg_l2_norm
        
def ddp_test(args, testloader, model, epoch, best_acc, mask, writer, world_pn):
    model.eval()
    top1_total = 0
    top5_total = 0
    total = 0
    if args.pbar and world_pn == 0:
        pbar = tqdm(testloader, total=len(testloader), desc=f"Epo {epoch} Testing", ncols=100)
    else:
        pbar = testloader

    test_acc = 0

    if args.bf16:
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16

    for x, y in pbar:
        x, y = x.cuda(), y.cuda()
        
        with torch.cuda.amp.autocast(enabled=args.use_amp, dtype=amp_dtype):
            with torch.no_grad():
                set_forward_with_fms(model, False)
                if mask is not None:
                    out = model((x, mask))
                else:
                    out = model(x)
        top1, top5 = accuracy(out, y, topk=(1, 5))
        top1_total += top1[0] * x.size(0)
        top5_total += top5[0] * x.size(0)
        total += x.size(0)

        
        reduced_total = torch.tensor(total, dtype=torch.float).cuda().detach()
        reduced_top1_total = top1_total.clone().detach()
        reduced_top5_total = top5_total.clone().detach()
        
        # dist.reduce(reduced_total, dst=0)
        # dist.reduce(reduced_top1_total, dst=0)
        # dist.reduce(reduced_top5_total, dst=0)

        dist.all_reduce(reduced_total)
        dist.all_reduce(reduced_top1_total)
        dist.all_reduce(reduced_top5_total)

        reduced_total = reduced_total.cpu().numpy()
        reduced_top1_total = reduced_top1_total.cpu().numpy()
        reduced_top5_total = reduced_top5_total.cpu().numpy()

        test_acc = reduced_top1_total/reduced_total
        if args.pbar and world_pn == 0:
            pbar.set_postfix_str(f"1a {100*reduced_top1_total/reduced_total:.2f}, 5a {100*reduced_top5_total/reduced_total:.2f}, best {100*best_acc:.2f}")
        
    # test_acc = (top1_total / total).item()
    if writer is not None:
        writer.add_scalar('Accuracy/test', test_acc, epoch)

    return test_acc

def single_test(args, testloader, model, epoch, best_acc, mask):
    model.eval()
    top1_total = 0
    top5_total = 0
    total = 0
    if args.pbar:
        pbar = tqdm(testloader, total=len(testloader), desc=f"Epo {epoch} Testing", ncols=100)
    else:
        pbar = testloader

    test_acc = 0
    for x, y in pbar:
        x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            if isinstance(model, DistributedDataParallel):
                raise TypeError("should not use DistributedDataParallel model in single_test")
            else:
                model.if_forward_with_fms = False
            if mask is not None:
                out = model((x, mask))
            else:
                out= model(x)
        top1, top5 = accuracy(out, y, topk=(1, 5))
        top1_total += top1[0] * x.size(0)
        top5_total += top5[0] * x.size(0)
        total += x.size(0)

        test_acc = top1_total/total
        if args.pbar:
            pbar.set_postfix_str(f"1a {100*top1_total/total:.2f}, 5a {100*top5_total/total:.2f}, best {100*best_acc:.2f}")
        
        

    # if writer is not None:
    #     writer.add_scalar('Accuracy/test', test_acc, epoch)
    

    return test_acc


def ddp_train_transfer(args, trainloader, model_s, optimizer, epoch, mask, writer, world_pn):
    model_s.eval()

    train_loss = 0

    if args.pbar and world_pn == 0:
        pbar = tqdm(trainloader, total=len(trainloader), desc=f"Epo {epoch} Lr {optimizer.param_groups[0]['lr']:.1e}", ncols=125)
    else:
        pbar = trainloader

    top1_total = 0
    top5_total = 0
    total = 0

    reduced_total = 0
    reduced_top1_total = 0
    reduced_top5_total = 0
    
    criterion_ce = nn.CrossEntropyLoss()

    for x, y in pbar:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        
        if isinstance(model_s, DistributedDataParallel):
            model_s.module.if_forward_with_fms = False
        else:
            model_s.if_forward_with_fms = False
        
        out_s = model_s((x, 0))

        loss = criterion_ce(out_s, y)

        train_loss += loss.item()
        
        loss.backward()
        
        optimizer.step()

        top1, top5 = accuracy(out_s, y, topk=(1, 5))
        top1_total += top1[0] * x.size(0)
        top5_total += top5[0] * x.size(0)
        total += x.size(0)

        reduced_total = torch.tensor(total, dtype=torch.float).cuda().detach()
        reduced_top1_total = top1_total.clone().detach()
        reduced_top5_total = top5_total.clone().detach()
        dist.reduce(reduced_total, dst=0)
        dist.reduce(reduced_top1_total, dst=0)
        dist.reduce(reduced_top5_total, dst=0)

        reduced_total = reduced_total.cpu().numpy()
        reduced_top1_total = reduced_top1_total.cpu().numpy()
        reduced_top5_total = reduced_top5_total.cpu().numpy()
        
        if args.pbar and world_pn == 0:
            pbar.set_postfix_str(f"L{train_loss/total:.2e}, ce{train_loss/total:.2e}, 1a {100*reduced_top1_total/reduced_total:.1f}, 5a {100*reduced_top5_total/reduced_total:.1f}")

    train_acc = (reduced_top1_total / reduced_total).item()

    if writer is not None:
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/train', train_loss/total, epoch)
        
    return train_acc
