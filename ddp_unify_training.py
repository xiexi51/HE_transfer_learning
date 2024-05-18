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
from my_layer_norm import MyLayerNorm

def set_forward_with_fms(model, if_forward_with_fms):
    if isinstance(model, DistributedDataParallel):
        model.module.if_forward_with_fms = if_forward_with_fms
    else:
        model.if_forward_with_fms = if_forward_with_fms

def ddp_unify_train(args: Namespace, trainloader: Iterable, model_s: torch.nn.Module, model_t: torch.nn.Module, optimizer: torch.optim.Optimizer, 
              epoch: int, mask: Tuple[float, float], writer: SummaryWriter, world_pn: int, omit_fms: int, mixup_fn: Mixup, criterion_ce: torch.nn.Module, 
              max_norm: float, update_freq: int, model_ema: List[ModelEma], act_learn: float, threshold_end: float, undo_grad: bool):
    if args.student_eval:
        model_s.eval()
    else:
        model_s.train()

    if model_t is not None:
        model_t.eval()

    train_loss = 0
    train_loss_kd = 0
    train_loss_ce = 0
    train_loss_fm = 0
    train_loss_conv = 0
    train_loss_var = 0

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

    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    
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
    
    prev_iter_train_acc = 0
    prev_model_state_dict = None
    prev_optimizer_state_dict = None
    undo_grad_signal = "-"

    for iter, (x, y) in enumerate(pbar):
        if iter >= 20 and not args.copy_to_a6000:
            break

        if iter >= effective_batches:
            break
        if undo_grad:
            # Save the current model and optimizer states
            prev_model_state_dict = model_s.state_dict()
            prev_optimizer_state_dict = optimizer.state_dict()

        x, y = x.cuda(), y.cuda()
        if mixup_fn is not None:
            x, y = mixup_fn(x, y)
        
        with torch.cuda.amp.autocast(enabled=args.use_amp, dtype=amp_dtype):
            if model_t is not None and (args.loss_conv_prune_factor > 0 or args.loss_fm_factor > 0 or args.loss_kd_factor > 0):
                with torch.no_grad():
                    set_forward_with_fms(model_t, True)
                    if args.loss_conv_prune_factor > 0:
                        out_t, fms_t, featuremap_t = model_t((x, -1, 1))
                    else:
                        out_t, fms_t, featuremap_t = model_t((x, -1, 1))
            if args.v_type != "demo":
                set_forward_with_fms(model_s, True)
                if mask is not None:
                    out_s, fms_s, featuremap_s = model_s((x, mask_current, threshold_end))
                else:
                    out_s, featuremap_s = model_s(x)
            else:
                out_s = model_s(x)

            loss_var = 0
            for name, module in model_s.module.named_modules():
                if isinstance(module, MyLayerNorm):
                    saved_var_mean = module.saved_var.mean()
                    loss_var += saved_var_mean
                    

            # if iter == 100:
            #     for name, module in model_s.module.named_modules():
            #         if isinstance(module, MyLayerNorm):
            #             max_val = float('-inf')
            #             min_val = float('inf')
            #             sum_val = 0
            #             count = 0

            #             for var in module.train_var_list:
            #                 max_val = max(max_val, var.max().item())
            #                 min_val = min(min_val, var.min().item())
            #                 sum_val += var.sum().item()
            #                 count += var.numel()

            #             mean_val = sum_val / count if count > 0 else float('nan')

            #             print(f"For module {name}: var_list max {max_val}, min {min_val}, mean {mean_val}")
            #     break
           
            loss = 0

            if args.loss_var_factor > 0:
                loss_var = loss_var * args.loss_var_factor
                loss += loss_var
                train_loss_var += loss_var.item()

            if args.v_type != "demo":
                total_conv, active_conv = model_s.module.get_conv_density()
                active_conv_rate = active_conv / total_conv
            else:
                active_conv_rate = 1

            if args.loss_conv_prune_factor > 0:    
                loss_conv = active_conv_rate * args.loss_conv_prune_factor
                loss += loss_conv
                train_loss_conv += loss_conv.item()

            if args.loss_fm_factor > 0:
                loss_fm = sum(loss_fm_fun(x, y) for x, y in zip(fms_s[omit_fms:], fms_t[omit_fms:])) * args.loss_fm_factor
                loss += loss_fm
                train_loss_fm += loss_fm.item()
            if args.loss_kd_factor > 0:
                # loss_kd = criterion_kd(out_s, out_t) * args.loss_kd_factor
                loss_kd = criterion_kd(featuremap_s, featuremap_t) * args.loss_kd_factor
                loss += loss_kd
                train_loss_kd += loss_kd.item()

            if args.loss_ce_factor > 0:
                loss_ce = criterion_ce(out_s, y) * args.loss_ce_factor
                loss += loss_ce
                train_loss_ce += loss_ce.item()

        if mixup_fn is not None:                
            top1_num = (out_s.argmax(dim=1) == y.argmax(dim=1)).float().sum().item()
        else:
            top1_num = (out_s.argmax(dim=1) == y).float().sum().item()
        top1_total += top1_num
        total += x.size(0)
        reduced_total = torch.tensor(total, dtype=torch.float).cuda().detach()
        reduced_top1_total = torch.tensor(top1_total, dtype=torch.float).cuda().detach()
        dist.all_reduce(reduced_total)
        dist.all_reduce(reduced_top1_total)
        reduced_total = reduced_total.cpu().numpy()
        reduced_top1_total = reduced_top1_total.cpu().numpy()
        iter_train_acc = reduced_top1_total / reduced_total

        if undo_grad:
            if iter > 0 and prev_iter_train_acc - iter_train_acc > args.undo_grad_threshold:
                undo_grad_signal = "u"
                model_s.load_state_dict(prev_model_state_dict)
                optimizer.load_state_dict(prev_optimizer_state_dict)
            else:
                undo_grad_signal = "-"
                prev_iter_train_acc = iter_train_acc

        loss /= update_freq
        # assert math.isfinite(loss)
        scaler.scale(loss).backward(create_graph=hasattr(optimizer, 'is_second_order') and optimizer.is_second_order)
        accumulated_batches += 1
        train_loss += loss.item()
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
                if name.endswith('weight_aux') and param.grad is not None:
                    param.grad.data *= 1
            scaler.step(optimizer) 
            scaler.update() 
            optimizer.zero_grad() 
            accumulated_batches = 0 
        
        if args.pbar and world_pn == 0:
            pbar.set_postfix_str(f"L{train_loss/total:.2e},fm{train_loss_fm/total:.2e},kd{train_loss_kd/total:.2e},ce{train_loss_ce/total:.2e},conv{active_conv_rate:.3f},var{train_loss_var/total:.2e} 1a {100*iter_train_acc:.1f}")

    # print(mask_current, mask[1])

    train_acc = (reduced_top1_total / reduced_total).item()

    if writer is not None:
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/train', train_loss/total, epoch)
        writer.add_scalar('Loss_fm/train', train_loss_fm/total, epoch)
        writer.add_scalar('Loss_kd/train', train_loss_kd/total, epoch)
        writer.add_scalar('Loss_ce/train', train_loss_ce/total, epoch)
        writer.add_scalar('Loss_conv/train', train_loss_conv/total, epoch)

    avg_l2_norm = total_l2_norm / param_count if param_count > 0 else 0
    return train_acc, avg_l2_norm
        
def ddp_test(args, testloader, model, epoch, best_acc, mask, writer, world_pn, threshold):
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

    for iter, (x, y) in enumerate(pbar):
        if iter >= 20 and not args.copy_to_a6000:
            break

        x, y = x.cuda(), y.cuda()
        
        with torch.cuda.amp.autocast(enabled=args.use_amp, dtype=amp_dtype):
            with torch.no_grad():
                if args.v_type != "demo":
                    set_forward_with_fms(model, False)
                    if mask is not None:
                        out, _ = model((x, mask, threshold))
                    else:
                        out, _ = model(x)
                else:
                    out = model(x)

        # if iter == 100:
        #     for name, module in model.module.named_modules():
        #         if isinstance(module, MyLayerNorm):
        #             max_val = float('-inf')
        #             min_val = float('inf')
        #             sum_val = 0
        #             count = 0

        #             for var in module.test_var_list:
        #                 max_val = max(max_val, var.max().item())
        #                 min_val = min(min_val, var.min().item())
        #                 sum_val += var.sum().item()
        #                 count += var.numel()

        #             mean_val = sum_val / count if count > 0 else float('nan')

        #             print(f"For module {name}: var_list max {max_val}, min {min_val}, mean {mean_val}")
        #     break
        
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
                out, _, _ = model((x, mask))
            else:
                out, _, _ = model(x)
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
