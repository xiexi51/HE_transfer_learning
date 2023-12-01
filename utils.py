import torch
from torch.optim.optimizer import Optimizer
from collections import defaultdict
from torch.nn import functional as F
import torch.nn as nn

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