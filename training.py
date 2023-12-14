import torch
from tqdm import tqdm
import torch.nn as nn
from utils import custom_mse_loss, at_loss, SoftTarget, accuracy

def train(args, trainloader, model_s, model_t, optimizer, epoch, mask, writer):
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
    
    cnt = 0
    # optimizer.zero_grad()

    for x, y in pbar:
        # torch.save(x, f'x{cnt}.pt')
        # torch.save(y, f'y{cnt}.pt')
        cnt += 1
        
        # print(x[0,0,0,0].numpy())
        # continue
        x, y = x.cuda(), y.cuda()
        if args.bf16:
            x = x.to(dtype=torch.bfloat16)
        optimizer.zero_grad()
        with torch.no_grad():
            if isinstance(model_t, torch.nn.DataParallel):
                model_t.module.if_forward_with_fms = True
            else:
                model_t.if_forward_with_fms = True
            out_t, fms_t = model_t(x)        
        if isinstance(model_s, torch.nn.DataParallel):
            model_s.module.if_forward_with_fms = True
        else:
            model_s.if_forward_with_fms = True
        out_s, fms_s = model_s((x, mask))

        # torch.save(out_t, f'origin_out_t.pt')
        # torch.save(fms_t, f'origin_fms_t.pt')
        # torch.save(out_s, f'origin_out_s.pt')
        # torch.save(fms_s, f'origin_fms_s.pt')

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

        # gradients = {}
        # weights = {}
        # for name, parameter in model_s.named_parameters():
        #     # 保存梯度
        #     if parameter.grad is not None:
        #         gradients[name] = parameter.grad.clone()
        #     # 保存权重
        #     weights[name] = parameter.data.clone()

        
        # if cnt > 1:
        #     torch.save(gradients, f'origin_2_gradients.pt')
        #     return
        
        # 保存权重
        # torch.save(weights, f'origin_weights.pt')

        torch.nn.utils.clip_grad_norm_(model_s.parameters(), 5)
        optimizer.step()

        # if args.lookahead:
        #     optimizer.sync_lookahead()
        
        train_loss += loss.item()
        train_loss_fm += loss_fm.item()
        train_loss_kd += loss_kd.item()
        train_loss_ce += loss_kd.item()

        top1, top5 = accuracy(out_s, y, topk=(1, 5))
        top1_total += top1[0] * x.size(0)
        top5_total += top5[0] * x.size(0)
        total += x.size(0)
        
        if args.pbar:
            pbar.set_postfix_str(f"L{train_loss/total:.2e},fm{train_loss_fm/total:.2e},kd{train_loss_kd/total:.2e},ce{train_loss_ce/total:.2e}, 1a {100*top1_total/total:.1f}, 5a {100*top5_total/total:.1f}")


    
    train_acc = (top1_total / total).item()
    # print('Epoch', epoch, 'Training Acc:', train_acc*100)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Loss/train', train_loss/total, epoch)
    writer.add_scalar('Loss_fm/train', train_loss_fm/total, epoch)
    writer.add_scalar('Loss_kd/train', train_loss_kd/total, epoch)
    writer.add_scalar('Loss_ce/train', train_loss_ce/total, epoch)
    return train_acc
        
def test(args, testloader, model, epoch, best_acc, mask, writer):
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
        if args.bf16:
            x = x.to(dtype=torch.bfloat16)
        with torch.no_grad():
            if isinstance(model, torch.nn.DataParallel):
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

        if args.pbar:
            pbar.set_postfix_str(f"1a {100*top1_total/total:.2f}, 5a {100*top5_total/total:.2f}, best {100*best_acc:.2f}")

    test_acc = (top1_total / total).item()
    writer.add_scalar('Accuracy/test', test_acc, epoch)
    
    if test_acc > best_acc:
        best_acc = test_acc

    return best_acc

