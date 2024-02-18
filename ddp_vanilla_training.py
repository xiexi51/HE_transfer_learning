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


# def ddp_train(args, trainloader, model_s, model_t, optimizer, epoch, mask, writer, pn, omit_fms, mixup_fn, criterion_ce, loss_scaler):
def ddp_vanilla_train(args: Namespace, trainloader: Iterable, model_s: torch.nn.Module, model_t: torch.nn.Module, optimizer: torch.optim.Optimizer, 
              epoch: int, mask: float, writer: SummaryWriter, pn: int, omit_fms: int, mixup_fn: Mixup, criterion_ce: torch.nn.Module, 
              max_norm: float, update_freq: int, model_ema: List[ModelEma]):
    model_s.train()
    if model_t is not None:
        model_t.eval()

    train_loss = 0
    train_loss_kd = 0
    train_loss_ce = 0
    train_loss_fm = 0

    if args.pbar and pn == 0:
        pbar = tqdm(trainloader, total=len(trainloader), desc=f"Epo {epoch} Lr {optimizer.param_groups[0]['lr']:.1e}", ncols=125)
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
    accumulated_batches = 0

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    optimizer.zero_grad()

    for iter, (x, y) in enumerate(pbar):
        if iter >= effective_batches:
            break

        # print(pn, x[0,0,0,0].numpy())
        # break
        x, y = x.cuda(), y.cuda()

        # if args.bf16:
        #     x = x.to(dtype=torch.bfloat16)

        if mixup_fn is not None:
            x, y = mixup_fn(x, y)
        
        with torch.cuda.amp.autocast(enabled=args.use_amp):
            if model_t is not None:
                if args.loss_fm_factor > 0 or args.loss_kd_factor > 0:
                    with torch.no_grad():
                        if isinstance(model_t, DistributedDataParallel):
                            model_t.module.if_forward_with_fms = True
                        else:
                            model_t.if_forward_with_fms = True
                        out_t, fms_t = model_t(x)

            # if args.loss_fm_factor > 0:
            #     out_s, fms_s = model_s((x, mask))
            # else:
            #     if mask is not None:
            #         out_s = model_s((x, mask))
            #     else:
            #         out_s = model_s(x)

            output_s = model_s(x)
        
            loss = 0

            # if args.loss_fm_factor > 0:
            #     loss_fm = sum(loss_fm_fun(x, y) for x, y in zip(fms_s[omit_fms:], fms_t[omit_fms:])) * args.loss_fm_factor
            #     loss += loss_fm
            #     train_loss_fm += loss_fm.item()
            # if args.loss_kd_factor > 0:
            #     loss_kd = criterion_kd(out_s, out_t) * args.loss_kd_factor
            #     loss += loss_kd
            #     train_loss_kd += loss_kd.item()

            if args.loss_ce_factor > 0:
                loss_ce = criterion_ce(output_s, y) * args.loss_ce_factor
                loss += loss_ce
                train_loss_ce += loss_ce.item()

        loss /= update_freq
        assert math.isfinite(loss)
        scaler.scale(loss).backward(create_graph=hasattr(optimizer, 'is_second_order') and optimizer.is_second_order)
        accumulated_batches += 1
        train_loss += loss.item()

        if accumulated_batches == update_freq:
            scaler.step(optimizer) 
            scaler.update() 
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
                        
        top1_num = (output_s.argmax(dim=1) == y.argmax(dim=1)).float().sum().item()
        top1_total += top1_num

        total += x.size(0)

        reduced_total = torch.tensor(total, dtype=torch.float).cuda().detach()
        reduced_top1_total = torch.tensor(top1_total, dtype=torch.float).cuda().detach()
        
        dist.reduce(reduced_total, dst=0)
        dist.reduce(reduced_top1_total, dst=0)

        reduced_total = reduced_total.cpu().numpy()
        reduced_top1_total = reduced_top1_total.cpu().numpy()
        
        if args.pbar and pn == 0:
            # print(total, top1_total, top5_total)
            # print(reduced_total, reduced_top1_total, reduced_top5_total)
            pbar.set_postfix_str(f"L{train_loss/total:.2e},fm{train_loss_fm/total:.2e},kd{train_loss_kd/total:.2e},ce{train_loss_ce/total:.2e}, 1a {100*reduced_top1_total/reduced_total:.1f}, 5a --")


    train_acc = (reduced_top1_total / reduced_total).item()

    if writer is not None:
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/train', train_loss/total, epoch)
        writer.add_scalar('Loss_fm/train', train_loss_fm/total, epoch)
        writer.add_scalar('Loss_kd/train', train_loss_kd/total, epoch)
        writer.add_scalar('Loss_ce/train', train_loss_ce/total, epoch)
    return train_acc
        
def ddp_test(args, testloader, model, epoch, best_acc, mask, writer, pn):
    model.eval()
    top1_total = 0
    top5_total = 0
    total = 0
    if args.pbar and pn == 0:
        pbar = tqdm(testloader, total=len(testloader), desc=f"Epo {epoch} Testing", ncols=100)
    else:
        pbar = testloader

    test_acc = 0
    for x, y in pbar:
        x, y = x.cuda(), y.cuda()
        
        with torch.cuda.amp.autocast(enabled=args.use_amp):
            with torch.no_grad():
                # if isinstance(model, DistributedDataParallel):
                #     model.module.if_forward_with_fms = False
                # else:
                #     model.if_forward_with_fms = False
                if mask is not None:
                    out = model((x, mask))
                else:
                    out = model(x)
        top1, top5 = accuracy(out, y, topk=(1, 5))
        top1_total += top1[0] * x.size(0)
        top5_total += top5[0] * x.size(0)
        total += x.size(0)

        
        reduced_total = torch.tensor(total, dtype=torch.float).cuda().detach()
        reduced_top1_total = torch.tensor(top1_total, dtype=torch.float).cuda().detach()
        reduced_top5_total = torch.tensor(top5_total, dtype=torch.float).cuda().detach()
        dist.reduce(reduced_total, dst=0)
        dist.reduce(reduced_top1_total, dst=0)
        dist.reduce(reduced_top5_total, dst=0)

        reduced_total = reduced_total.cpu().numpy()
        reduced_top1_total = reduced_top1_total.cpu().numpy()
        reduced_top5_total = reduced_top5_total.cpu().numpy()

        if args.pbar and pn == 0:
            pbar.set_postfix_str(f"1a {100*reduced_top1_total/reduced_total:.2f}, 5a {100*reduced_top5_total/reduced_total:.2f}, best {100*best_acc:.2f}")
        
        test_acc = reduced_top1_total/reduced_total

    # test_acc = (top1_total / total).item()
    if writer is not None:
        writer.add_scalar('Accuracy/test', test_acc, epoch)
    
    if test_acc > best_acc:
        best_acc = test_acc

    return test_acc, best_acc

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
            # else:
            #     model.if_forward_with_fms = True
            if mask is not None:
                out, fms = model((x, mask))
            else:
                # out, fms = model(x)
                out= model(x)
        top1, top5 = accuracy(out, y, topk=(1, 5))
        top1_total += top1[0] * x.size(0)
        top5_total += top5[0] * x.size(0)
        total += x.size(0)

        if args.pbar:
            pbar.set_postfix_str(f"1a {100*top1_total/total:.2f}, 5a {100*top5_total/total:.2f}, best {100*best_acc:.2f}")
        
        test_acc = top1_total/total

    # if writer is not None:
    #     writer.add_scalar('Accuracy/test', test_acc, epoch)
    
    if test_acc > best_acc:
        best_acc = test_acc

    return test_acc, best_acc


def ddp_train_transfer(args, trainloader, model_s, optimizer, epoch, mask, writer, pn):
    model_s.eval()

    train_loss = 0

    if args.pbar and pn == 0:
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
        # print(pn, x[0,0,0,0].numpy())
        # break
        x, y = x.cuda(), y.cuda()
        # x = torch.load(f'x{pn}.pt').cuda()
        # y = torch.load(f'y{pn}.pt').cuda()

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
        reduced_top1_total = torch.tensor(top1_total, dtype=torch.float).cuda().detach()
        reduced_top5_total = torch.tensor(top5_total, dtype=torch.float).cuda().detach()
        dist.reduce(reduced_total, dst=0)
        dist.reduce(reduced_top1_total, dst=0)
        dist.reduce(reduced_top5_total, dst=0)

        reduced_total = reduced_total.cpu().numpy()
        reduced_top1_total = reduced_top1_total.cpu().numpy()
        reduced_top5_total = reduced_top5_total.cpu().numpy()
        
        if args.pbar and pn == 0:
            pbar.set_postfix_str(f"L{train_loss/total:.2e}, ce{train_loss/total:.2e}, 1a {100*reduced_top1_total/reduced_total:.1f}, 5a {100*reduced_top5_total/reduced_total:.1f}")

    train_acc = (reduced_top1_total / reduced_total).item()

    if writer is not None:
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/train', train_loss/total, epoch)
        
    return train_acc