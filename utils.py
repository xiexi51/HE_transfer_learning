import torch
from torch.optim.optimizer import Optimizer
from collections import defaultdict
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
import torch.distributed as dist
import subprocess
import glob
import os

cse_gateway_login = "xix22010@137.99.0.102"
a6000_login = "xix22010@192.168.10.16"
# ssh_options = f"-o ProxyJump={cse_gateway_login} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
ssh_options = f"-o ProxyJump={cse_gateway_login} -o StrictHostKeyChecking=no "

def slience_cmd(cmd, silent=True):
    try:
        if silent:
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

def copy_to_a6000(source, destination, silent=True):
    if not os.path.exists(source):
        print(f"Error: {source} does not exist")
        return
    slience_cmd(f"scp {ssh_options} {source} {a6000_login}:{destination}", silent=silent)

def copy_tensorboard_logs(log_dir, a6000_log_dir):
    tb_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))
    for tb_file in tb_files:
        destination = os.path.join(a6000_log_dir, os.path.basename(tb_file))
        copy_to_a6000(tb_file, destination)

def print_available_gpus():
    num_devices = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_devices}")

    for i in range(num_devices):
        device_name = torch.cuda.get_device_name(i)
        device_capability = torch.cuda.get_device_capability(i)
        print(f"  GPU {i}: {device_name}, Capability: {device_capability}")

def change_print_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print
    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    __builtin__.print = print

class CustomSettings:
    def __init__(self, relu_type, poly_weight_inits, poly_factors, prune_type, prune_1_1_kernel, norm_type, cheb_params, 
                 training_use_cheb, var_norm_boundary, ln_momentum, ln_use_quad, ln_trainable_quad_finetune, ln_quad_coeffs, 
                 ln_quad_finetune_factors, ln_x_scaler):
        self.relu_type = relu_type
        self.poly_weight_inits = poly_weight_inits
        self.poly_factors = poly_factors
        self.prune_type = prune_type
        self.prune_1_1_kernel = prune_1_1_kernel
        self.norm_type = norm_type
        self.cheb_params = cheb_params
        self.training_use_cheb = training_use_cheb
        self.var_norm_boundary = var_norm_boundary
        self.ln_momentum = ln_momentum
        self.ln_use_quad = ln_use_quad
        self.ln_trainable_quad_finetune = ln_trainable_quad_finetune
        self.ln_quad_coeffs = ln_quad_coeffs
        self.ln_quad_finetune_factors = ln_quad_finetune_factors
        self.ln_x_scaler = ln_x_scaler

class Lookahead(Optimizer):
    def __init__(self, base_optimizer, alpha=0.5, k=1):
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


def at(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

def at_loss(x, y):
    return (at(x) - at(y)).pow(2).mean()

class SoftTarget(nn.Module):
	'''
	Distilling the Knowledge in a Neural Network
	https://arxiv.org/pdf/1503.02531.pdf
	'''
	def __init__(self, T):
		super(SoftTarget, self).__init__()
		self.T = T

	def forward(self, out_s, out_t):
		loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
						F.softmax(out_t/self.T, dim=1),
						reduction='batchmean') * self.T * self.T
		return loss

class STEFunction(torch.autograd.Function):
    """ define straight through estimator with overrided gradient (gate) """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return torch.mul(F.softplus(input), grad_output)

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
    
def custom_mse_loss(x, y):
    """
    Compute the mean squared error loss.
    When y > 0, the loss for those elements is divided by 3.
    """
    mse_loss = F.mse_loss(x, y, reduction='none')  # Compute element-wise MSE loss
    adjust_factor = torch.where(y > 0, 1/3, 1.0)  # Create a factor, 1/3 where y > 0, else 1
    adjusted_loss = mse_loss * adjust_factor  # Adjust the loss
    return adjusted_loss.mean()  # Return the mean loss

class MaskProvider:
    def __init__(self, decrease_type, mask_epochs):
        self.mask_epochs = mask_epochs
        if decrease_type == "1-sinx":
            mask_x = np.linspace(0, np.pi / 2, mask_epochs + 1)
            self.mask_y = 1 - np.sin(mask_x)
        elif decrease_type == "e^(-x/10)":
            mask_x = np.linspace(0, 80, mask_epochs + 1)
            self.mask_y = np.exp(-mask_x / 10)
        elif decrease_type == "linear": 
            self.mask_y = np.linspace(1, 0, mask_epochs + 1)
        elif decrease_type == "1":
            self.mask_y = np.ones(mask_epochs + 1)
        else:
            self.mask_y = np.zeros(mask_epochs + 1)

    def get_mask(self, epoch):
        if epoch < self.mask_epochs:
            return (self.mask_y[epoch], self.mask_y[epoch + 1])
        else:
            return (0, 0)
        

def fp16_compress_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """
    This DDP communication hook implements a simple gradient compression
    approach that casts ``GradBucket`` tensor to half-precision floating-point format (``torch.float16``)
    and then divides it by the process group size.
    It allreduces those ``float16`` gradient tensors. Once compressed gradient
    tensors are allreduced, the chained callback ``decompress`` casts it back to the input data type (such as ``float32``).

    Example::
        >>> # xdoctest: +SKIP
        >>> ddp_model.register_comm_hook(process_group, fp16_compress_hook)
    """
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()

    compressed_tensor = bucket.buffer().to(torch.float16).div_(world_size)
    fut = dist.all_reduce(
        compressed_tensor, group=group_to_use, async_op=True
    ).get_future()

    def decompress(fut):
        decompressed_tensor = bucket.buffer()
        # Decompress in place to reduce the peak memory.
        # See: https://github.com/pytorch/pytorch/issues/45968
        decompressed_tensor.copy_(fut.value()[0])
        return decompressed_tensor

    return fut.then(decompress)

def irg_loss(fms_s, fms_t, w_irg_vert=0.1, w_irg_edge=5.0, w_irg_tran=5.0):
    def euclidean_dist_fms(fm1, fm2, squared=False, eps=1e-12):
        '''
        Calculating the IRG Transformation, where fm1 precedes fm2 in the network.
        '''
        if fm1.size(2) > fm2.size(2):
            fm1 = F.adaptive_avg_pool2d(fm1, (fm2.size(2), fm2.size(3)))
        if fm1.size(1) < fm2.size(1):
            fm2 = (fm2[:,0::2,:,:] + fm2[:,1::2,:,:]) / 2.0

        fm1 = fm1.view(fm1.size(0), -1)
        fm2 = fm2.view(fm2.size(0), -1)
        fms_dist = torch.sum(torch.pow(fm1-fm2, 2), dim=-1).clamp(min=eps)

        if not squared:
            fms_dist = fms_dist.sqrt()

        fms_dist = fms_dist / fms_dist.max()

        return fms_dist

    def euclidean_dist_fm(fm, squared=False, eps=1e-12): 
        '''
        Calculating the IRG edge of feature map. 
        '''
        fm = fm.view(fm.size(0), -1)
        fm_square = fm.pow(2).sum(dim=1)
        fm_prod   = torch.mm(fm, fm.t())
        fm_dist   = (fm_square.unsqueeze(0) + fm_square.unsqueeze(1) - 2 * fm_prod).clamp(min=eps)

        if not squared:
            fm_dist = fm_dist.sqrt()

        fm_dist = fm_dist.clone()
        fm_dist[range(len(fm)), range(len(fm))] = 0
        fm_dist = fm_dist / fm_dist.max()

        return fm_dist

    # Assuming the feature maps are paired correctly in the lists
    loss_irg_vert = sum(F.mse_loss(fm_s, fm_t) for fm_s, fm_t in zip(fms_s, fms_t)) / len(fms_s)

    loss_irg_edge = sum(F.mse_loss(euclidean_dist_fm(fm_s), euclidean_dist_fm(fm_t)) for fm_s, fm_t in zip(fms_s, fms_t)) / len(fms_s)

    # Assuming adjacent feature maps are used for transformation calculation
    loss_irg_tran = sum(F.mse_loss(euclidean_dist_fms(fms_s[i], fms_s[i+1]), euclidean_dist_fms(fms_t[i], fms_t[i+1])) 
                        for i in range(len(fms_s) - 1)) / (len(fms_s) - 1)

    loss = w_irg_vert * loss_irg_vert + w_irg_edge * loss_irg_edge + w_irg_tran * loss_irg_tran

    return loss