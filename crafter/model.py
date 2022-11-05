# -*- coding: utf-8 -*-
from __future__ import division
import math
import torch
from torch import nn
from torch.nn import functional as F


# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.empty(out_features, in_features))
        self.register_buffer(
            'weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)


class DQN(nn.Module):
    def __init__(self, args, action_space):
        super(DQN, self).__init__()
        self.atoms = args.atoms
        self.action_space = action_space

        if args.architecture == 'canonical':
            self.convs = nn.Sequential(nn.Conv2d(args.history_length, 32, 8, stride=4, padding=0), nn.ReLU(),
                                       nn.Conv2d(32, 64, 4, stride=2,
                                                 padding=0), nn.ReLU(),
                                       nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU())
            self.conv_output_size = 3136
        elif args.architecture == 'data-efficient':
            self.convs = nn.Sequential(nn.Conv2d(args.history_length, 32, 5, stride=5, padding=0), nn.ReLU(),
                                       nn.Conv2d(32, 64, 5, stride=5, padding=0), nn.ReLU())
            self.conv_output_size = 576
        self.fc_h_v = NoisyLinear(
            self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
        self.fc_h_a = NoisyLinear(
            self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
        self.fc_z_v = NoisyLinear(
            args.hidden_size, self.atoms, std_init=args.noisy_std)
        self.fc_z_a = NoisyLinear(
            args.hidden_size, action_space * self.atoms, std_init=args.noisy_std)

    def forward(self, x, log=False):
        x = self.convs(x)
        x = x.view(-1, self.conv_output_size)
        v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
        a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
        v, a = v.view(-1, 1, self.atoms), a.view(-1,
                                                 self.action_space, self.atoms)
        q = v + a - a.mean(1, keepdim=True)  # Combine streams
        if log:  # Use log softmax for numerical stability
            # Log probabilities with action over second dimension
            q = F.log_softmax(q, dim=2)
        else:
            # Probabilities with action over second dimension
            q = F.softmax(q, dim=2)
        return q

    def reset_noise(self):
        for name, module in self.named_children():
            if 'fc' in name:
                module.reset_noise()


class QRDQN(nn.Module):
    def __init__(self, args, action_space):
        super(QRDQN, self).__init__()
        self.n_quantiles = args.n_quantiles
        self.n_ensemble = args.n_ensemble
        self.action_space = action_space

        if args.architecture == 'canonical':
            self.convs = nn.Sequential(nn.Conv2d(args.history_length, 32, 8, stride=4, padding=0), nn.ReLU(),
                                       nn.Conv2d(32, 64, 4, stride=2,
                                                 padding=0), nn.ReLU(),
                                       nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU())
            self.conv_output_size = 3136
        elif args.architecture == 'data-efficient':
            self.convs = nn.Sequential(nn.Conv2d(args.history_length, 32, 5, stride=5, padding=0), nn.ReLU(),
                                       nn.Conv2d(32, 64, 5, stride=5, padding=0), nn.ReLU())
            self.conv_output_size = 576

        if args.bootstrapped_qrdqn:
            all_fc_h_v, all_fc_z_v = [], []
            for _ in range(self.n_ensemble):
                all_fc_h_v.append(
                    nn.Linear(self.conv_output_size, args.hidden_size))
                all_fc_z_v.append(nn.Linear(args.hidden_size,
                                  self.action_space * self.n_quantiles))
            self.all_fc_h_v = nn.ModuleList(all_fc_h_v)
            self.all_fc_z_v = nn.ModuleList(all_fc_z_v)
            self.quantiles = self._quantile_bootstrap
            self.forward = self._forward_bootstrapped
        else:
            self.fc_h_v = nn.Linear(self.conv_output_size, args.hidden_size)
            self.fc_z_v = nn.Linear(
                args.hidden_size, action_space * self.n_quantiles)
            self.quantiles = self._quantiles
            self.forward = self._forward_regular

    def _forward_regular(self, x, **kwargs):
        quantiles = self.quantiles(x)
        return quantiles.mean(1)

    def _quantiles(self, x, **kwargs):
        x = self.convs(x)
        x = x.view(-1, self.conv_output_size)
        quantiles = self.fc_z_v(F.relu(self.fc_h_v(x))
                                ).view(-1, self.n_quantiles, self.action_space)
        return quantiles

    def _forward_bootstrapped(self, x, **kwargs):
        all_quantiles = self.quantiles(x)  # [K, B, atom, action]
        all_quantiles_mean = torch.stack(
            [q.mean(1) for q in all_quantiles])  # [K, B, action]
        mean = torch.mean(all_quantiles_mean, axis=0)  # [B, action]
        return mean

    def _quantile_bootstrap(self, x, freeze_feature=False):
        x = self.convs(x)
        if freeze_feature:
            x = x.detach()
        x = x.view(-1, self.conv_output_size)
        all_quantiles = []
        for fc_z_v, fc_h_v in zip(self.all_fc_z_v, self.all_fc_h_v):
            quantiles = fc_z_v(F.relu(fc_h_v(x))).view(-1,
                                                       self.n_quantiles, self.action_space)
            all_quantiles.append(quantiles)
        return all_quantiles

    def reset_noise(self):
        return

    # def reset_noise(self):
    #     for name, module in self.named_children():
    #         if 'fc' in name:
    #             module.reset_noise()
