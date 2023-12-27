import torch
from tqdm import tqdm
import torch.nn as nn
from utils import custom_mse_loss, at_loss, SoftTarget, accuracy
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

def ddp_train(args, trainloader, model_s, model_t, optimizer, epoch, mask, writer, pn):
    # model_s.train_fz_bn()
    model_s.train()
    model_t.eval()
    # model_t.train()

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

    if args.loss_fm_type == "mse":
        loss_fm_fun = nn.MSELoss()
    elif args.loss_fm_type == "custom_mse":
        loss_fm_fun = custom_mse_loss
    else:
        loss_fm_fun = at_loss

    criterion_kd = SoftTarget(4.0).cuda()
    criterion_ce = nn.CrossEntropyLoss()

    for x, y in pbar:
        # print(pn, x[0,0,0,0].numpy())
        # break
        x, y = x.cuda(), y.cuda()
        # x = torch.load(f'x{pn}.pt').cuda()
        # y = torch.load(f'y{pn}.pt').cuda()

        if args.bf16:
            x = x.to(dtype=torch.bfloat16)
        optimizer.zero_grad()
        if args.loss_fm_factor > 0 or args.loss_kd_factor > 0:
            with torch.no_grad():
                if isinstance(model_t, DistributedDataParallel):
                    model_t.module.if_forward_with_fms = True
                else:
                    model_t.if_forward_with_fms = True
                out_t, fms_t = model_t(x)

        if args.loss_fm_factor > 0:
            if isinstance(model_s, DistributedDataParallel):
                model_s.module.if_forward_with_fms = True
            else:
                model_s.if_forward_with_fms = True
            out_s, fms_s = model_s((x, mask))
        else:
            if isinstance(model_s, DistributedDataParallel):
                model_s.module.if_forward_with_fms = False
            else:
                model_s.if_forward_with_fms = False
            out_s = model_s((x, mask))

        # torch.save(out_t, f'{pn}_out_t.pt')
        # torch.save(fms_t, f'{pn}_fms_t.pt')
        # torch.save(out_s, f'{pn}_out_s.pt')
        # torch.save(fms_s, f'{pn}_fms_s.pt')

        loss = 0

        if args.loss_fm_factor > 0:
            loss_fm = sum(loss_fm_fun(x, y) for x, y in zip(fms_s, fms_t)) * args.loss_fm_factor
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

        train_loss += loss.item()
        
        loss.backward()

        # gradients = {}
        # weights = {}
        # for name, parameter in model_s.module.named_parameters():
        
        #     if parameter.grad is not None:
        #         gradients[name] = parameter.grad.clone()
        #     weights[name] = parameter.data.clone()

        # torch.save(gradients, f'{pn}_2_gradients.pt')
        # torch.save(weights, f'{pn}_weights.pt')

        # for param in model_s.parameters():
        #     if param.requires_grad and param.grad is not None:
        #         param.grad *= args.total_gpus

        # torch.nn.utils.clip_grad_norm_(model_s.parameters(), 5)
        
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
            # print(total, top1_total, top5_total)
            # print(reduced_total, reduced_top1_total, reduced_top5_total)
            pbar.set_postfix_str(f"L{train_loss/total:.2e},fm{train_loss_fm/total:.2e},kd{train_loss_kd/total:.2e},ce{train_loss_ce/total:.2e}, 1a {100*reduced_top1_total/reduced_total:.1f}, 5a {100*reduced_top5_total/reduced_total:.1f}")


    train_acc = (top1_total / total).item()

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
        if args.bf16:
            x = x.to(dtype=torch.bfloat16)
        with torch.no_grad():
            if isinstance(model, DistributedDataParallel):
                # raise TypeError("should not use DistributedDataParallel model when testing")
                model.module.if_forward_with_fms = False
            else:
                model.if_forward_with_fms = False
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


def ddp_train_avg(args, trainloader, model_s, model_t, optimizer, epoch, avgmask, writer, pn):
    model_s.train()
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

        if args.bf16:
            x = x.to(dtype=torch.bfloat16)
        optimizer.zero_grad()
        if args.loss_fm_factor > 0 or args.loss_kd_factor > 0:
            with torch.no_grad():
                if isinstance(model_t, DistributedDataParallel):
                    model_t.module.if_forward_with_fms = True
                else:
                    model_t.if_forward_with_fms = True
                out_t, fms_t = model_t(x)

        if args.loss_fm_factor > 0:
            if isinstance(model_s, DistributedDataParallel):
                model_s.module.if_forward_with_fms = True
            else:
                model_s.if_forward_with_fms = True
            out_s, fms_s = model_s((x, avgmask))
        else:
            if isinstance(model_s, DistributedDataParallel):
                model_s.module.if_forward_with_fms = False
            else:
                model_s.if_forward_with_fms = False
            out_s = model_s((x, avgmask))

        loss = 0

        if args.loss_fm_factor > 0:
            loss_fm = sum(loss_fm_fun(x, y) for x, y in zip(fms_s[5:], fms_t[5:])) * args.loss_fm_factor
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
            pbar.set_postfix_str(f"L{train_loss/total:.2e},fm{train_loss_fm/total:.2e},kd{train_loss_kd/total:.2e},ce{train_loss_ce/total:.2e}, 1a {100*reduced_top1_total/reduced_total:.1f}, 5a {100*reduced_top5_total/reduced_total:.1f}")

    train_acc = (top1_total / total).item()

    if writer is not None:
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/train', train_loss/total, epoch)
        writer.add_scalar('Loss_fm/train', train_loss_fm/total, epoch)
        writer.add_scalar('Loss_kd/train', train_loss_kd/total, epoch)
        writer.add_scalar('Loss_ce/train', train_loss_ce/total, epoch)
    return train_acc