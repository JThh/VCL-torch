from abc import ABC, abstractmethod
import math
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
# from torch._jit_internal import weak_module, weak_script_method 


EMPTY_STATS = {
            'w_mean': torch.zeros(1, dtype=torch.float32),
            'b_mean': torch.zeros(1, dtype=torch.float32),
            'w_var': torch.zeros(1, dtype=torch.float32),
            'b_var': torch.zeros(1, dtype=torch.float32)
        }


class SpatialConvFlow(nn.Module):
    def __init__(self, out_channels, in_channels, H, W):
        super(SpatialConvFlow, self).__init__()
        self.scale_params = nn.Parameter(torch.Tensor(1, 1, H, W))
        nn.init.normal_(self.scale_params, 1.0, 0.02)  # Close to 1, with small variance
        
    def forward(self, z):
        transformed_z = torch.einsum("iohw,cchw->iohw", z, torch.exp(self.scale_params))
        return transformed_z

# @weak_module
class VariationalLayer(torch.nn.Module, ABC):
    """
    Base class for any type of neural network layer that uses variational inference. The
    defining aspect of such a layer are:

    1. The PyTorch forward() method should allow the parametric specification of whether
    the forward pass should be carried out with parameters sampled from the posterior
    parameter distribution, or using the distribution parameters directly as if though
    they were standard neural network parameters.

    2. Must provide a method for resetting the parameter distributions for the next task
    in the online (multi-task) variational inference setting.

    3. Must provide a method for computing the KL divergence between the layer's parameter
    posterior distribution and parameter prior distribution.
    """
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    @abstractmethod
    def forward(self, x, sample_parameters=True):
        pass

    @abstractmethod
    def reset_for_next_task(self):
        pass

    @abstractmethod
    def kl_divergence(self) -> torch.Tensor:
        pass

    @abstractmethod
    def get_statistics(self) -> dict:
        pass


class BasicBlockVCL(VariationalLayer):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockVCL, self).__init__()
        self.conv1 = MeanFieldGaussianConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = MeanFieldGaussianConv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = CustomSequential(
                MeanFieldGaussianConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(self.expansion * planes)
            )
        else:
            self.shortcut = CustomSequential(nn.Identity())

    def forward(self, x, sample_parameters=True):
        out = F.relu(self.bn1(self.conv1(x, sample_parameters)))
        out = self.bn2(self.conv2(out, sample_parameters))
        out += self.shortcut(x, sample_parameters)
        out = F.relu(out)
        return out

    def kl_divergence(self):
        kl_divergence = self.conv1.kl_divergence() + self.conv2.kl_divergence()
        kl_divergence += self.shortcut.kl_divergence(kl_divergence.device) 
        return kl_divergence
        
    def reset_for_next_task(self):
        self.conv1.reset_for_next_task()
        self.conv2.reset_for_next_task()
        for module in self.shortcut.modules_list:
            if isinstance(module, VariationalLayer):
                module.reset_for_next_task()

    def get_statistics(self):
        "Return flattened and concatenated means and variances"
        statistics_list = []
        for layer in [self.conv1, self.conv2, self.shortcut]:
            if isinstance(layer, VariationalLayer):
                statistics_list.append(layer.get_statistics())

        if not statistics_list:
            return EMPTY_STATS
    
        merged_statistics = {key: [] for key in statistics_list[0].keys()}
        
        for key in merged_statistics.keys():
            merged_statistics[key] = torch.cat([torch.flatten(stats[key]).to(torch.device('cpu')) for stats in statistics_list])
        
        return merged_statistics


class CustomSequential(VariationalLayer):
    def __init__(self, *args):
        super(CustomSequential, self).__init__()
        self.modules_list = nn.ModuleList(args)

    def forward(self, x, *args, **kwargs):
        for module in self.modules_list:
           if isinstance(module, VariationalLayer):
               x = module.forward(x, *args, **kwargs)
           else:
               x = module(x)
        return x

    def kl_divergence(self, device=torch.device('cpu')):
        total_kl = torch.zeros(1, device=device, requires_grad=False)

        for module in self.modules_list:
            if isinstance(module, CustomSequential):
                total_kl += module.kl_divergence(self.device)
            if hasattr(module, 'kl_divergence'):
                total_kl += module.kl_divergence().to(device)

        return total_kl
    
    def reset_for_next_task(self):
        for module in self.modules_list:
            if isinstance(module, VariationalLayer):
                module.reset_for_next_task()

    def get_statistics(self):
        statistics_list = []
        for module in self.modules_list:
            if isinstance(module, VariationalLayer):
                statistics_list.append(module.get_statistics())

        if not statistics_list:
            return EMPTY_STATS

        merged_statistics = {key: [] for key in statistics_list[0].keys()}
        
        for key in merged_statistics.keys():
            merged_statistics[key] = torch.cat([torch.flatten(stats[key]).to(torch.device('cpu')) for stats in statistics_list])
        
        return merged_statistics


    
class MeanFieldGaussianConv2d(VariationalLayer):
    """
    A 2D convolutional layer applying variational inference with mean field
    approximation for Gaussian distributions.

    Args:
        in_channels: Number of channels in the input image
        out_channels: Number of channels produced by the convolution
        kernel_size: Size of the convolving kernel
        stride: Stride of the convolution. Default: 1
        padding: Zero-padding added to both sides of the input. Default: 0
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initial_posterior_variance=1e-3, epsilon=1e-8, n_flows=0):
        super().__init__(epsilon)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ipv = initial_posterior_variance

        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)

        # Define the size for the weights and biases
        self.weight_size = (out_channels, in_channels, *self.kernel_size)
        self.bias_size = (out_channels,)

        # Prior distributions: non-trainable, zero mean and log-variance
        self.register_buffer('prior_W_means', torch.zeros(self.weight_size))
        self.register_buffer('prior_W_log_vars', torch.zeros(self.weight_size))
        self.register_buffer('prior_b_means', torch.zeros(self.bias_size))
        self.register_buffer('prior_b_log_vars', torch.zeros(self.bias_size))

        # Posterior distributions: trainable parameters for means and log-variances
        self.posterior_W_means = Parameter(torch.empty_like(self._buffers['prior_W_means'], requires_grad=True))
        self.posterior_b_means = Parameter(torch.empty_like(self._buffers['prior_b_means'], requires_grad=True))
        self.posterior_W_log_vars = Parameter(torch.empty_like(self._buffers['prior_W_log_vars'], requires_grad=True))
        self.posterior_b_log_vars = Parameter(torch.empty_like(self._buffers['prior_b_log_vars'], requires_grad=True))

        self._initialize_posteriors()

        self.flows = nn.ModuleList([SpatialConvFlow(in_channels, out_channels, kernel_size, kernel_size) for _ in range(n_flows)])

    def forward(self, x, sample_parameters=True):
        """Produces module output on an input."""
        if sample_parameters:
            w, b = self._sample_parameters()
            return F.conv2d(x, w, b, self.stride, self.padding)
        else:
            return F.conv2d(x, self.posterior_W_means, self.posterior_b_means, self.stride, self.padding)

    def reset_for_next_task(self):
        """Overwrites the current prior with the current posterior."""
        self.prior_W_means.data.copy_(self.posterior_W_means.data)
        self.prior_W_log_vars.data.copy_(self.posterior_W_log_vars.data)
        self.prior_b_means.data.copy_(self.posterior_b_means.data)
        self.prior_b_log_vars.data.copy_(self.posterior_b_log_vars.data)

    def kl_divergence(self) -> torch.Tensor:
        """Returns KL(posterior, prior) for the parameters of this layer."""
         # obtain flattened means, log variances, and variances of the prior distribution
        prior_means = torch.autograd.Variable(torch.cat(
            (torch.reshape(self._buffers['prior_W_means'], (-1,)),
             torch.reshape(self._buffers['prior_b_means'], (-1,)))),
            requires_grad=False
        )
        prior_log_vars = torch.autograd.Variable(torch.cat(
            (torch.reshape(self._buffers['prior_W_log_vars'], (-1,)),
             torch.reshape(self._buffers['prior_b_log_vars'], (-1,)))),
            requires_grad=False
        )
        prior_vars = torch.exp(prior_log_vars)

        # obtain flattened means, log variances, and variances of the approximate posterior distribution
        posterior_means = torch.cat(
            (torch.reshape(self.posterior_W_means, (-1,)),
             torch.reshape(self.posterior_b_means, (-1,))),
        )
        posterior_log_vars = torch.cat(
            (torch.reshape(self.posterior_W_log_vars, (-1,)),
             torch.reshape(self.posterior_b_log_vars, (-1,))),
        )
        posterior_vars = torch.exp(posterior_log_vars)

        # compute kl divergence (this computation is valid for multivariate diagonal Gaussians)
        kl_elementwise = posterior_vars / (prior_vars + self.epsilon) + \
                         torch.pow(prior_means - posterior_means, 2) / (prior_vars + self.epsilon) - \
                         1 + prior_log_vars - posterior_log_vars

        return 0.5 * kl_elementwise.sum().unsqueeze(-1)

    def _sample_parameters(self):
        """Sample weights and biases with reparameterization trick."""
        w_epsilons = torch.randn_like(self.posterior_W_means)
        b_epsilons = torch.randn_like(self.posterior_b_means)

        # Apply flows to variances to simulate covariances.
        W_log_vars = self.posterior_W_log_vars
        for flow in self.flows:
            W_log_vars = flow(W_log_vars)

        # W_log_vars = W_log_vars.view_as(self.posterior_W_log_vars)
        # b_log_vars = b_log_vars.view_as(self.posterior_b_log_vars)

        w = self.posterior_W_means + torch.mul(w_epsilons, torch.exp(0.5 * W_log_vars))
        b = self.posterior_b_means + torch.mul(b_epsilons, torch.exp(0.5 * self.posterior_b_log_vars))

        return w, b

    def _initialize_posteriors(self):
        """Initialize posterior parameters with small random values for means and set log variances to a small initial value."""
        torch.nn.init.normal_(self.posterior_W_means, mean=0, std=0.1)
        torch.nn.init.uniform_(self.posterior_b_means, -0.1, 0.1)
        torch.nn.init.constant_(self.posterior_W_log_vars, math.log(self.ipv))
        torch.nn.init.constant_(self.posterior_b_log_vars, math.log(self.ipv))

    def get_statistics(self) -> dict:
        statistics = {
            'w_mean': self.posterior_W_means,
            'b_mean': self.posterior_b_means,
            'w_var': torch.exp(self.posterior_W_log_vars),
            'b_var': torch.exp(self.posterior_b_log_vars)
        }

        return statistics


# @weak_module
class MeanFieldGaussianLinear(VariationalLayer):
    """
    A linear transformation on incoming data of the form :math:`y = w^T x + b`,
    where the weights w and the biases b are distributions of parameters
    rather than point estimates. The layer has a prior distribution over its
    parameters, as well as an approximate posterior distribution.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
    """

    def __init__(self, in_features, out_features, initial_posterior_variance=1e-3, epsilon=1e-8):
        super().__init__(epsilon)
        self.in_features = in_features
        self.out_features = out_features
        self.ipv = initial_posterior_variance

        # priors are not optimizable parameters - all means and log-variances are zero
        self.register_buffer('prior_W_means', torch.zeros(out_features, in_features))
        self.register_buffer('prior_W_log_vars', torch.zeros(out_features, in_features))
        self.register_buffer('prior_b_means', torch.zeros(out_features))
        self.register_buffer('prior_b_log_vars', torch.zeros(out_features))

        self.posterior_W_means = Parameter(torch.empty_like(self._buffers['prior_W_means'], requires_grad=True))
        self.posterior_b_means = Parameter(torch.empty_like(self._buffers['prior_b_means'], requires_grad=True))
        self.posterior_W_log_vars = Parameter(torch.empty_like(self._buffers['prior_W_log_vars'], requires_grad=True))
        self.posterior_b_log_vars = Parameter(torch.empty_like(self._buffers['prior_b_log_vars'], requires_grad=True))

        self._initialize_posteriors()

    # @weak_script_method
    def forward(self, x, sample_parameters=True):
        """ Produces module output on an input. """
        if sample_parameters:
            w, b = self._sample_parameters()
            return F.linear(x, w, b)
        else:
            return F.linear(x, self.posterior_W_means, self.posterior_b_means)

    def reset_for_next_task(self):
        """ Overwrites the current prior with the current posterior. """
        self._buffers['prior_W_means'].data.copy_(self.posterior_W_means.data)
        self._buffers['prior_W_log_vars'].data.copy_(self.posterior_W_log_vars.data)
        self._buffers['prior_b_means'].data.copy_(self.posterior_b_means.data)
        self._buffers['prior_b_log_vars'].data.copy_(self.posterior_b_log_vars.data)

    def kl_divergence(self) -> torch.Tensor:
        """ Returns KL(posterior, prior) for the parameters of this layer. """
        # obtain flattened means, log variances, and variances of the prior distribution
        prior_means = torch.autograd.Variable(torch.cat(
            (torch.reshape(self._buffers['prior_W_means'], (-1,)),
             torch.reshape(self._buffers['prior_b_means'], (-1,)))),
            requires_grad=False
        )
        prior_log_vars = torch.autograd.Variable(torch.cat(
            (torch.reshape(self._buffers['prior_W_log_vars'], (-1,)),
             torch.reshape(self._buffers['prior_b_log_vars'], (-1,)))),
            requires_grad=False
        )
        prior_vars = torch.exp(prior_log_vars)

        # obtain flattened means, log variances, and variances of the approximate posterior distribution
        posterior_means = torch.cat(
            (torch.reshape(self.posterior_W_means, (-1,)),
             torch.reshape(self.posterior_b_means, (-1,))),
        )
        posterior_log_vars = torch.cat(
            (torch.reshape(self.posterior_W_log_vars, (-1,)),
             torch.reshape(self.posterior_b_log_vars, (-1,))),
        )
        posterior_vars = torch.exp(posterior_log_vars)

        # compute kl divergence (this computation is valid for multivariate diagonal Gaussians)
        kl_elementwise = posterior_vars / (prior_vars + self.epsilon) + \
                         torch.pow(prior_means - posterior_means, 2) / (prior_vars + self.epsilon) - \
                         1 + prior_log_vars - posterior_log_vars

        return 0.5 * kl_elementwise.sum()

    def get_statistics(self) -> dict:
        statistics = {
            'w_mean': self.posterior_W_means,
            'b_mean': self.posterior_b_means,
            'w_var': torch.exp(self.posterior_W_log_vars),
            'b_var': torch.exp(self.posterior_b_log_vars)
        }

        return statistics

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def _sample_parameters(self):
        # obtained sampled weights and biases using local reparameterization trick
        w_epsilons = torch.randn_like(self.posterior_W_means)
        b_epsilons = torch.randn_like(self.posterior_b_means)

        w = self.posterior_W_means + torch.mul(w_epsilons, torch.exp(0.5 * self.posterior_W_log_vars))
        b = self.posterior_b_means + torch.mul(b_epsilons, torch.exp(0.5 * self.posterior_b_log_vars))

        return w, b

    def _initialize_posteriors(self):
        # posteriors on the other hand are optimizable parameters - means are normally distributed, log_vars
        # have some small initial value
        torch.nn.init.normal_(self.posterior_W_means, mean=0, std=0.1)
        torch.nn.init.uniform_(self.posterior_b_means, -0.1, 0.1)
        torch.nn.init.constant_(self.posterior_W_log_vars, math.log(self.ipv))
        torch.nn.init.constant_(self.posterior_b_log_vars, math.log(self.ipv))
