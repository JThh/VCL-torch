"""
This module contains the less thoroughly tested implementations of VCL models.

In particular, the models in this module are defined in a different manner to the main
models in the models.vcl_nn module. The models in this module are defined in terms of
bayesian layers from the layers.variational module, which abstract the details of
online variational inference. This approach is in line with the standard style in which
PyTorch models are defined.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.variational import VariationalLayer, MeanFieldGaussianLinear, MeanFieldGaussianConv2d, BasicBlockVCL, CustomSequential
from models.deep_models import Encoder
from util.operations import kl_divergence, bernoulli_log_likelihood, normal_with_reparameterization

EPSILON = 1e-8  # Small value to avoid divide-by-zero and log(zero) problems


class VCL(nn.Module, ABC):
    """ Base class for all VCL models """
    def __init__(self, epsilon=EPSILON):
        super().__init__()
        self.epsilon = epsilon

    @abstractmethod
    def reset_for_new_task(self, head_idx):
        pass

    @abstractmethod
    def get_statistics(self) -> (list, dict):
        pass


class ResNetVCL(nn.Module):
    def __init__(self, block, num_blocks, num_tasks, num_classes_per_task, in_channels=3, initial_posterior_variance=1e-6, epsilon=1e-8, mc_sampling_n=10, device=torch.device('cuda:0')):
        super(ResNetVCL, self).__init__()
        self.device = device                
        self.mc_sampling_n = mc_sampling_n
        self.ipv = initial_posterior_variance
        self.epsilon = epsilon
        self.num_tasks = num_tasks
        self.num_classes_per_task = num_classes_per_task
        self.in_planes = 64

        # Initial convolutional layer
        self.conv1 = MeanFieldGaussianConv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, initial_posterior_variance=self.ipv, epsilon=self.epsilon)
        self.bn1 = nn.BatchNorm2d(64)

        # Layers defined by the ResNet block structure
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[1], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[1], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Task-specific heads
        self.task_specific_heads = nn.ModuleList()
        for _ in range(num_tasks):
            head = MeanFieldGaussianLinear(512 * block.expansion, num_classes_per_task, initial_posterior_variance=self.ipv, epsilon=self.epsilon)
            self.task_specific_heads.append(head)

        self.softmax = nn.Softmax(dim=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return CustomSequential(*layers)

    def forward(self, x, task_idx=0, sample_parameters=True):
        out = F.relu(self.bn1(self.conv1(x, sample_parameters)))
        # out = F.relu((self.conv1(x, sample_parameters)))
        out = self.layer1(out, sample_parameters)
        out = self.layer2(out, sample_parameters)
        out = self.layer3(out, sample_parameters)
        out = self.layer4(out, sample_parameters)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.task_specific_heads[task_idx](out, sample_parameters)
        out = self.softmax(out)
        return out

    def _kl_divergence(self, task_idx):
        kl_divergence = self.conv1.kl_divergence()
        kl_divergence += self.layer1.kl_divergence(self.device)
        kl_divergence += self.layer2.kl_divergence(self.device)
        kl_divergence += self.layer3.kl_divergence(self.device)
        kl_divergence += self.layer4.kl_divergence(self.device)
        kl_divergence += self.task_specific_heads[task_idx].kl_divergence()
        return kl_divergence

    def reset_for_new_task(self, task_idx):
        self.conv1.reset_for_next_task()
        self.task_specific_heads[task_idx].reset_for_next_task()
        self.layer1.reset_for_next_task()
        self.layer2.reset_for_next_task()
        self.layer3.reset_for_next_task()
        self.layer4.reset_for_next_task()

    def vcl_loss(self, x, y, task_idx, task_size):
        return self._kl_divergence(task_idx) / task_size + torch.nn.NLLLoss()(self(x, task_idx), y)

    def point_estimate_loss(self, x, y, task_idx):
        return torch.nn.NLLLoss()(self(x, task_idx, sample_parameters=False), y)

    def prediction(self, x, task_idx):
        """ Returns an integer between 0 and self.out_size """
        return torch.argmax(self(x, task_idx), dim=1)

    def get_statistics(self) -> (list, dict):
        pass

    def _mean_posterior_variance(self):
        """
        Return the mean posterior variance for logging purposes.
        Excludes the head layer.
        """
        posterior_vars = []
    
        for layer in [self.conv1, self.layer1, self.layer2, self.layer3, self.layer4]:
            if isinstance(layer, VariationalLayer):
                stats = layer.get_statistics()
                layer_vars = torch.cat([torch.flatten(stats['w_var']).to(torch.device('cpu')), torch.flatten(stats['b_var']).to(torch.device('cpu'))])
                
            posterior_vars.append(layer_vars)
        
        all_variances = torch.cat(posterior_vars)
        return torch.mean(all_variances).item()


class ConvVCL(nn.Module):
    def __init__(self, input_dims=(1,28,28), n_hidden_layers=1, hidden_dim=512, num_tasks=5, num_classes_per_task=10, initial_posterior_variance=1e-6, epsilon=1e-8, mc_sampling_n=10, device=torch.device('cpu')):
        super(ConvVCL, self).__init__()
        self.device = device                
        self.mc_sampling_n = mc_sampling_n
        self.ipv = initial_posterior_variance
        self.epsilon = epsilon
        self.num_tasks = num_tasks
        self.num_classes_per_task = num_classes_per_task

        # Convolutional layers
        self.conv1 = MeanFieldGaussianConv2d(input_dims[0], 64, kernel_size=3, stride=1, padding=1, initial_posterior_variance=self.ipv, epsilon=self.epsilon).to(device)
        self.conv2 = MeanFieldGaussianConv2d(64, 128, kernel_size=3, stride=1, padding=1, initial_posterior_variance=self.ipv, epsilon=self.epsilon).to(device)
        self.conv3 = MeanFieldGaussianConv2d(128, 256, kernel_size=3, stride=1, padding=1, initial_posterior_variance=self.ipv, epsilon=self.epsilon).to(device)
        self.shared_conv_layers = nn.ModuleList([self.conv1, self.conv2, self.conv3])

        # Calculate the output size after convolutional layers
        fake_input = torch.zeros(1, *input_dims).to(device)
        fake_output = self.conv_forward(fake_input)
        conv_output_size = fake_output.view(-1).shape[0]

        # Integrated task-specific linear layers and heads
        self.task_specific_layers = nn.ModuleList()
        for _ in range(num_tasks):
            layers = []
            shared_dims = [conv_output_size] + [hidden_dim for _ in range(n_hidden_layers)] + [num_classes_per_task]
            for i in range(len(shared_dims) - 1):
                layer = MeanFieldGaussianLinear(shared_dims[i], shared_dims[i + 1], initial_posterior_variance=self.ipv, epsilon=self.epsilon).to(device)
                layers.append(layer)
            self.task_specific_layers.append(nn.Sequential(*layers))

        self.softmax = nn.Softmax(dim=1)

    def conv_forward(self, x, sample_parameters=True):
        out = F.relu(self.conv1(x, sample_parameters))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out, sample_parameters))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv3(out, sample_parameters))
        out = F.max_pool2d(out, 2)
        return out

    def forward(self, x, task_idx=0, sample_parameters=True):
        y_out = torch.zeros(size=(x.shape[0], self.num_classes_per_task)).to(self.device)

        for _ in range(self.mc_sampling_n if sample_parameters else 1):
            out = self.conv_forward(x, sample_parameters)
            out = out.view(out.size(0), -1)  # Flatten the output of conv layers
            out = self.task_specific_layers[task_idx](out)
            y_out.add_(self.softmax(out))

        y_out.div_(self.mc_sampling_n if sample_parameters else 1)
        return y_out

    def vcl_loss(self, x, y, task_idx, task_size):
        output = self.forward(x, task_idx)
        return self._kl_divergence(task_idx) / task_size + nn.NLLLoss()(output, y)

    def prediction(self, x, task_idx):
        """Returns the predicted class for the given task."""
        output = self.forward(x, task_idx, sample_parameters=False)
        return torch.argmax(output, dim=1)

    def point_estimate_loss(self, x, y, task_idx):
        output = self.forward(x, task_idx, sample_parameters=False)
        return torch.nn.NLLLoss()(output, y)

    def _kl_divergence(self, task_idx) -> torch.Tensor:
        kl_divergence = torch.zeros(1, device=self.device, dtype=torch.float32)

        # Compute KL divergence for conv layers
        for conv in self.shared_conv_layers:
            kl_divergence += conv.kl_divergence()

        # Compute KL divergence for task-specific layers
        for layer in self.task_specific_layers[task_idx]:
            kl_divergence += layer.kl_divergence()

        return kl_divergence

    def reset_for_new_task(self, task_idx):
        for conv in self.shared_conv_layers:
            conv.reset_for_next_task()
        for layer in self.task_specific_layers[task_idx]:
            layer.reset_for_next_task()

    def get_statistics(self) -> (list, dict):
        pass

    def _mean_posterior_variance(self):
        """
        Return the mean posterior variance for logging purposes.
        Excludes the head layer.
        """
        posterior_vars = []
    
        for layer in self.shared_conv_layers:
            if isinstance(layer, VariationalLayer):
                stats = layer.get_statistics()
                layer_vars = torch.cat([torch.flatten(stats['w_var']).to(torch.device('cpu')), torch.flatten(stats['b_var']).to(torch.device('cpu'))])
                posterior_vars.append(layer_vars)
        
        all_variances = torch.cat(posterior_vars)
        return torch.mean(all_variances).item()



class PartialConvVCL(ConvVCL):
    """
    Make only heads and shared linear layers variational (with backbone based on non-variational convs).
    """
    def __init__(self, input_dims=(1,28,28), n_hidden_layers=1, hidden_dim=512, num_tasks=5, num_classes_per_task=10, initial_posterior_variance=1e-6, epsilon=1e-8, mc_sampling_n=10, device=torch.device('cpu')):
        super(PartialConvVCL, self).__init__(input_dims, n_hidden_layers, hidden_dim, n_heads, num_classes, initial_posterior_variance, epsilon, mc_sampling_n, device)

        self.conv1 = nn.Conv2d(input_dims[0], 64, kernel_size=3, stride=1, padding=1).to(device)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1).to(device)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1).to(device)
    
        self.shared_conv_layers = nn.ModuleList([self.conv1, self.conv2, self.conv3])

    def conv_forward(self, x, sample_parameters=True):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv3(out))
        out = F.max_pool2d(out, 2)
        return out


class DiscriminativeVCL(VCL):
    """
    A Bayesian neural network which is updated using variational inference
    methods, and which is multi-headed at the output end. Suitable for
    continual learning of discriminative tasks.
    """

    def __init__(self, x_dim, h_dim, y_dim, n_heads=1, shared_h_dims=(100, 100),
                 initial_posterior_variance=1e-6, mc_sampling_n=10, device='cpu'):
        super().__init__()
        # check for bad parameters
        if n_heads < 1:
            raise ValueError('Network requires at least one head.')

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.y_dim = y_dim
        self.n_heads = n_heads
        self.ipv = initial_posterior_variance
        self.mc_sampling_n = mc_sampling_n
        self.device = device

        shared_dims = [x_dim] + list(shared_h_dims) + [h_dim]

        # list of layers in shared network
        self.shared_layers = nn.ModuleList([
            MeanFieldGaussianLinear(shared_dims[i], shared_dims[i + 1], self.ipv, EPSILON) for i in
            range(len(shared_dims) - 1)
        ])
        # list of heads, each head is a list of layers
        self.heads = nn.ModuleList([
            MeanFieldGaussianLinear(self.h_dim, self.y_dim, self.ipv, EPSILON) for _ in range(n_heads)
        ])

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, head_idx, sample_parameters=True):
        y_out = torch.zeros(size=(x.size()[0], self.y_dim)).to(self.device)

        # repeat forward pass n times to sample layer params multiple times
        for _ in range(self.mc_sampling_n if sample_parameters else 1):
            h = x
            # shared part
            for layer in self.shared_layers:
                h = F.relu(layer(h, sample_parameters=sample_parameters))

            # head
            h = self.heads[head_idx](h, sample_parameters=sample_parameters)
            h = self.softmax(h)

            y_out.add_(h)

        y_out.div_(self.mc_sampling_n)

        return y_out

    def vcl_loss(self, x, y, head_idx, task_size):
        return self._kl_divergence(head_idx) / task_size + torch.nn.NLLLoss()(self(x, head_idx), y)

    def point_estimate_loss(self, x, y, head_idx):
        return torch.nn.NLLLoss()(self(x, head_idx, sample_parameters=False), y)

    def prediction(self, x, task):
        """ Returns an integer between 0 and self.out_size """
        return torch.argmax(self(x, task), dim=1)

    def reset_for_new_task(self, head_idx):
        for layer in self.shared_layers:
            if isinstance(layer, VariationalLayer):
                layer.reset_for_next_task()

        if isinstance(self.heads[head_idx], VariationalLayer):
            self.heads[head_idx].reset_for_next_task()

    def get_statistics(self) -> (list, dict):
        layer_statistics = []
        model_statistics = {
            'average_w_mean': 0,
            'average_b_mean': 0,
            'average_w_var': 0,
            'average_b_var': 0
        }

        n_layers = 0
        for layer in self.shared_layers:
            n_layers += 1
            layer_statistics.append(layer.get_statistics())
            model_statistics['average_w_mean'] += layer_statistics[-1]['average_w_mean']
            model_statistics['average_b_mean'] += layer_statistics[-1]['average_b_mean']
            model_statistics['average_w_var'] += layer_statistics[-1]['average_w_var']
            model_statistics['average_b_var'] += layer_statistics[-1]['average_b_var']

        for head in self.heads:
            n_layers += 1
            layer_statistics.append(head.get_statistics())
            model_statistics['average_w_mean'] += layer_statistics[-1]['average_w_mean']
            model_statistics['average_b_mean'] += layer_statistics[-1]['average_b_mean']
            model_statistics['average_w_var'] += layer_statistics[-1]['average_w_var']
            model_statistics['average_b_var'] += layer_statistics[-1]['average_b_var']

        # todo averaging averages like this is actually incorrect (assumes equal num of params in each layer)
        model_statistics['average_w_mean'] /= n_layers
        model_statistics['average_b_mean'] /= n_layers
        model_statistics['average_w_var'] /= n_layers
        model_statistics['average_b_var'] /= n_layers

        return layer_statistics, model_statistics

    def _kl_divergence(self, head_idx) -> torch.Tensor:
        kl_divergence = torch.zeros(1, requires_grad=False).to(self.device)

        # kl divergence is equal to sum of parameter-wise divergences since
        # distribution is diagonal multivariate normal (parameters are independent)
        for layer in self.shared_layers:
            kl_divergence = torch.add(kl_divergence, layer.kl_divergence())

        kl_divergence = torch.add(kl_divergence, self.heads[head_idx].kl_divergence())
        return kl_divergence


class GenerativeVCL(VCL):
    """
    A Bayesian neural network which is updated using variational inference
    methods, and which is multi-headed at the input end. Suitable for
    continual learning of generative tasks.
    """

    def __init__(self, z_dim, h_dim, x_dim, n_heads=0, encoder_h_dims=(500, 500), decoder_head_h_dims=(500,),
                 decoder_shared_h_dims=(500,), initial_posterior_variance=1e-6, mc_sampling_n=10, device='cpu'):
        super().__init__()
        # handle bad input
        if n_heads < 1:
            raise ValueError('Network requires at least one head.')

        # internal parameters
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.x_dim = x_dim
        self.encoder_h_dims = encoder_h_dims
        self.n_heads = n_heads
        self.ipv = initial_posterior_variance
        self.mc_sampling_n = mc_sampling_n
        self.device = device
        # prior over z
        self.z_prior_mean = 0.0
        self.z_prior_log_variance = 0.0
        # layer dimensions
        head_dims = [z_dim] + list(decoder_head_h_dims) + [h_dim]
        shared_dims = [h_dim] + list(decoder_shared_h_dims) + [x_dim]

        # encoder produces means and variances for a z-dim diagonal gaussian
        self.encoder = Encoder(x_dim, z_dim * 2, encoder_h_dims)
        # list of heads, each with a list of layers
        self.decoder_heads = nn.ModuleList([
            nn.ModuleList([MeanFieldGaussianLinear(head_dims[i], head_dims[i + 1], self.ipv) for i in range(len(head_dims) - 1)])
            for _ in
            range(n_heads)
        ])
        # list of layers in shared network
        self.decoder_shared = nn.ModuleList([
            MeanFieldGaussianLinear(shared_dims[i], shared_dims[i + 1], self.ipv, EPSILON) for i in
            range(len(shared_dims) - 1)
        ])

    def forward(self, x, head_idx, sample_parameters=True):
        """ Forward pass for the entire VAE, passing through both the encoder and the decoder. """
        z_params = self.forward_encoder_only(x, head_idx).view(len(x), self.z_dim, 2)
        z_means = z_params[:, :, 0]
        z_log_std = z_params[:, :, 1]

        z = normal_with_reparameterization(z_means, torch.exp(z_log_std), self.device).to(self.device)
        x_out = self.forward_decoder_only(z, head_idx)

        return x_out

    def forward_encoder_only(self, x, head_idx):
        """ Forward pass for the encoder. Takes data as input, and returns the mean and variance
        of the log-normal posterior latent distribution p(z | x), for each data point. This
        distribution is used to sample the actual latent representation z for the data point x. """
        return self.encoder(x, head_idx)

    def forward_decoder_only(self, z, head_idx, sample_parameters=True):
        """ Forward pass for the decoder. Takes a latent representation and produces a reconstruction
        of the original data point that z represents. """
        x_out = torch.zeros(size=(z.size()[0], self.x_dim)).to(self.device)

        # repeat forward pass n times to sample layer params multiple times
        for _ in range(self.mc_sampling_n if sample_parameters else 1):
            h = z
            for layer in self.decoder_heads[head_idx]:
                h = F.relu(layer(h))

            for layer in self.decoder_shared:
                h = F.relu(layer(h))

            x_out.add_(h)
        x_out.div_(self.mc_sampling_n if sample_parameters else 1)

        return x_out

    def vae_loss(self, x, task_idx, task_size):
        """ Loss implementing the full variational lower bound from page 5 of the paper """
        elbo = self._elbo(x, task_idx)
        kl = self._kl_divergence(task_idx) / task_size
        return - elbo + kl

    def generate(self, batch_size, task_idx):
        """ Sample new images x from p(x|z)p(z), where z is a gaussian noise distribution. """
        z = torch.randn((batch_size, self.z_dim)).to(self.device)

        for layer in self.decoder_heads[task_idx]:
            z = F.relu(layer(z))

        for layer in self.decoder_shared:
            z = F.relu(layer(z))

        return z

    def reset_for_new_task(self, head_idx):
        """ Creates new encoder and resets the decoder (in the VCL sense). """
        self.encoder.reset_for_new_task()

        for layer in self.decoder_shared:
            if isinstance(layer, VariationalLayer):
                layer.reset_for_next_task()

        for layer in self.decoder_heads[head_idx]:
            if isinstance(layer, VariationalLayer):
                layer.reset_for_next_task()

    def get_statistics(self) -> dict:
        pass

    def _elbo(self, x, head_idx, sample_n=1):
        """ Computes the variational lower bound """
        z_params = self.forward_encoder_only(x, head_idx).view(len(x), self.z_dim, 2)
        z_means = z_params[:, :, 0]
        z_log_std = z_params[:, :, 1]

        kl = kl_divergence(z_means, z_log_std)
        log_likelihood = torch.zeros(size=(x.size()[0],)).to(self.device)
        for _ in range(sample_n):
            z = normal_with_reparameterization(z_means, torch.exp(z_log_std), self.device).to(self.device)
            x_reconstructed = self.forward_decoder_only(z, head_idx)
            # Bernoulli likelihood of data
            log_likelihood.add_(bernoulli_log_likelihood(x, x_reconstructed, self.epsilon))
        log_likelihood = log_likelihood.div_(sample_n)

        return torch.mean(log_likelihood - kl)

    def _kl_divergence(self, head_idx) -> torch.Tensor:
        """ KL divergence of the VCL decoder's posterior distribution from its previous posterior. """
        kl_divergence = torch.zeros(1, requires_grad=False).to(self.device)

        # kl divergence is equal to sum of parameter-wise divergences since
        # distribution is diagonal multivariate normal (parameters are independent)
        for layer in self.decoder_shared:
            kl_divergence = torch.add(kl_divergence, layer.kl_divergence())

        for layer in self.decoder_heads[head_idx]:
            kl_divergence = torch.add(kl_divergence, layer.kl_divergence())

        return kl_divergence
