# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import division

import math
from statistics import quantiles

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def init_(m):
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))


def init_relu_(m):
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain("relu"))


def apply_init_(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)


class Conv2d_tf(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(Conv2d_tf, self).__init__(*args, **kwargs)
        self.padding = kwargs.get("padding", "SAME")

    def _compute_padding(self, input, dim):
        input_size = input.size(dim + 2)
        filter_size = self.weight.size(dim + 2)
        effective_filter_size = (filter_size - 1) * self.dilation[dim] + 1
        out_size = (input_size + self.stride[dim] - 1) // self.stride[dim]
        total_padding = max(0, (out_size - 1) * self.stride[dim] + effective_filter_size - input_size)
        additional_padding = int(total_padding % 2 != 0)
        return additional_padding, total_padding

    def forward(self, input):
        if self.padding == "VALID":
            return F.conv2d(
                input,
                self.weight,
                self.bias,
                self.stride,
                padding=0,
                dilation=self.dilation,
                groups=self.groups,
            )
        rows_odd, padding_rows = self._compute_padding(input, dim=0)
        cols_odd, padding_cols = self._compute_padding(input, dim=1)
        if rows_odd or cols_odd:
            input = F.pad(input, [0, cols_odd, 0, rows_odd])

        return F.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            padding=(padding_rows // 2, padding_cols // 2),
            dilation=self.dilation,
            groups=self.groups,
        )


class L2Pool(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.AvgPool2d(*args, **kwargs)
        self.n = self.pool.kernel_size ** 2

    def forward(self, x):
        return torch.sqrt(self.pool(x ** 2) * self.n)


class ResidualBlock(nn.Module):
    def __init__(self, n_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = Conv2d_tf(n_channels, n_channels, kernel_size=3, stride=1, padding=(1, 1))
        self.relu = nn.ReLU()
        self.conv2 = Conv2d_tf(n_channels, n_channels, kernel_size=3, stride=1, padding=(1, 1))
        self.stride = stride

        apply_init_(self.modules())

        self.train()

    def forward(self, x):
        identity = x

        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += identity
        return out


class ImpalaCNN(nn.Module):
    """
    Arguments:
    ----------
    num_inputs : `int`
        Number of channels in the input image.
    """

    def __init__(self, num_inputs, channels=[16, 32, 32]):  # noqa: B006
        super(ImpalaCNN, self).__init__()

        # define Impala CNN
        self.layer1 = self._make_layer(num_inputs, channels[0])
        self.layer2 = self._make_layer(channels[0], channels[1])
        self.layer3 = self._make_layer(channels[1], channels[2])
        self.flatten = Flatten()
        self.relu = nn.ReLU()

        # initialise all conv modules
        apply_init_(self.modules())

        # put model into train mode
        self.train()

    def _make_layer(self, in_channels, out_channels, stride=1):
        layers = list()

        layers.append(Conv2d_tf(in_channels, out_channels, kernel_size=3, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(ResidualBlock(out_channels))
        layers.append(ResidualBlock(out_channels))

        return nn.Sequential(*layers)

    def forward(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu(self.flatten(x))
        return x


class MiniGridCNN(nn.Module):
    def __init__(self, args, env):
        super(MiniGridCNN, self).__init__()
        final_channels = 64

        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, final_channels, (2, 2)),
            nn.ReLU(),
        )
        n = env.observation_space.shape[-2]
        m = env.observation_space.shape[-1]
        self.image_embedding_size = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * final_channels

    def forward(self, x):
        return self.image_conv(x)


# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

        self.weight = self.weight_mu

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

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
            return F.linear(
                input,
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu + self.bias_sigma * self.bias_epsilon,
            )
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)


class DQN(nn.Module):
    def __init__(self, args, env):
        super(DQN, self).__init__()
        self.args = args
        self.action_space = env.action_space.n
        self.dueling = args.dueling
        self.c51 = args.c51
        self.qrdqn = args.qrdqn
        self.qrdqn_bootstrap = args.qrdqn_bootstrap
        self.bootstrap_dqn = args.bootstrap_dqn
        self.noisy_layers = args.noisy_layers
        if self.c51:
            self.atoms = args.atoms
            self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.device)
            self.V_min = args.V_min
            self.V_max = args.V_max
            self.delta_z = float(self.V_max - self.V_min) / (self.atoms - 1)
        if self.qrdqn:
            self.atoms = 200
        if not self.qrdqn and not self.bootstrap_dqn:
            self.forward = (
                self._forward_c51
                if self.c51
                else (self._forward_dueling if self.dueling else self._forward_no_duel)
            )
        elif self.bootstrap_dqn:
            self.n_ensemble = args.n_ensemble
            self.forward = self._forward_bootstrap_dqn
        elif self.qrdqn and self.qrdqn_bootstrap:
            self.quantiles = self._quantile_bootstrap
            self.n_ensemble = args.n_ensemble
            self.forward = self._forward_bootstrap_qrdqn
            taus = torch.arange(0, self.atoms + 1, device=args.device, dtype=torch.float32) / self.atoms
            self.tau_hats = ((taus[1:] + taus[:-1]) / 2.0).view(1, self.atoms)
            self.c51 = False
        else:
            self.quantiles = self._quantiles_dueling if self.dueling else self._quantiles_no_duel
            self.forward = self._forward_qrdqn
            taus = torch.arange(0, self.atoms + 1, device=args.device, dtype=torch.float32) / self.atoms
            self.tau_hats = ((taus[1:] + taus[:-1]) / 2.0).view(1, self.atoms)
            self.c51 = False
        self.make_network(args, env)

        if args.attach_task_id:  # only works for regular dqn for now
            self.task_embedder = nn.Embedding(args.num_train_seeds, 64)

    def make_network(self, args, env):
        if args.env_name.startswith("MiniGrid"):
            self.features = MiniGridCNN(args, env)
            self.conv_output_size = self.features.image_embedding_size
        elif env.observation_space.shape[0] == 84:
            # atari
            self.features = ImpalaCNN(4)
            example_state = torch.randn(
                (
                    1,
                    4,
                )
                + env.observation_space.shape
            )
            self.conv_output_size = self.features(example_state).shape[1]
        else:
            c = env.observation_space.shape[0]
            if self.args.attach_task_id:
                c -= 1
            self.features = ImpalaCNN(c)
            if env.observation_space.shape != (3, 64, 64) and not self.args.attach_task_id:
                example_state = torch.randn((1,) + env.observation_space.shape)
                self.conv_output_size = self.features(example_state).shape[1]
            else:
                self.conv_output_size = 2048
        
        if self.args.attach_task_id:
            self.conv_output_size += 64

        if self.noisy_layers:
            if self.dueling and (self.c51 or self.qrdqn):
                self.fc_h_v = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
                self.fc_h_a = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
                self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
                self.fc_z_a = NoisyLinear(
                    args.hidden_size, self.action_space * self.atoms, std_init=args.noisy_std
                )
            elif self.dueling:
                self.fc_h_v = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
                self.fc_h_a = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
                self.fc_z_v = NoisyLinear(args.hidden_size, 1, std_init=args.noisy_std)
                self.fc_z_a = NoisyLinear(args.hidden_size, self.action_space, std_init=args.noisy_std)
            elif self.c51 or self.qrdqn:
                self.fc_h_v = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
                self.fc_z_v = NoisyLinear(
                    args.hidden_size, self.action_space * self.atoms, std_init=args.noisy_std
                )
            else:
                self.fc_h_v = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
                self.fc_z_v = NoisyLinear(args.hidden_size, self.action_space, std_init=args.noisy_std)
        else:
            if self.dueling and (self.c51 or self.qrdqn):
                self.fc_h_v = nn.Linear(self.conv_output_size, args.hidden_size)
                self.fc_h_a = nn.Linear(self.conv_output_size, args.hidden_size)
                self.fc_z_v = nn.Linear(args.hidden_size, self.atoms)
                self.fc_z_a = nn.Linear(args.hidden_size, self.action_space * self.atoms)
            elif self.dueling:
                self.fc_h_v = nn.Linear(self.conv_output_size, args.hidden_size)
                self.fc_h_a = nn.Linear(self.conv_output_size, args.hidden_size)
                self.fc_z_v = nn.Linear(args.hidden_size, 1)
                self.fc_z_a = nn.Linear(args.hidden_size, self.action_space)
            elif (self.c51 or self.qrdqn) and not self.qrdqn_bootstrap:
                self.fc_h_v = nn.Linear(self.conv_output_size, args.hidden_size)
                self.fc_z_v = nn.Linear(args.hidden_size, self.action_space * self.atoms)
            elif (self.qrdqn and self.qrdqn_bootstrap):
                all_fc_h_v, all_fc_z_v = [], []
                for _ in range(self.n_ensemble):
                    all_fc_h_v.append(nn.Linear(self.conv_output_size, args.hidden_size))
                    all_fc_z_v.append(nn.Linear(args.hidden_size, self.action_space * self.atoms))
                self.all_fc_h_v = nn.ModuleList(all_fc_h_v)
                self.all_fc_z_v = nn.ModuleList(all_fc_z_v)

                self.anchors_all_fc_h_v = nn.ModuleList([nn.ParameterList([nn.parameter.Parameter(p.data.clone(), False) for p in list(f.parameters())]) for f in self.all_fc_h_v])
                self.anchors_all_fc_z_v = nn.ModuleList([nn.ParameterList([nn.parameter.Parameter(p.data.clone(), False) for p in list(f.parameters())]) for f in self.all_fc_z_v])

            elif self.bootstrap_dqn:
                all_fc_h_v, all_fc_z_v = [], []
                for _ in range(self.n_ensemble):
                    all_fc_h_v.append(nn.Linear(self.conv_output_size, args.hidden_size))
                    all_fc_z_v.append(nn.Linear(args.hidden_size, self.action_space))
                self.all_fc_h_v = nn.ModuleList(all_fc_h_v)
                self.all_fc_z_v = nn.ModuleList(all_fc_z_v)
            else:
                self.fc_h_v = nn.Linear(self.conv_output_size, args.hidden_size)
                self.fc_z_v = nn.Linear(args.hidden_size, self.action_space)

        apply_init_(self.modules())
        self.train()

    def _forward_c51(self, x, log=False):
        dist = self.dist(x, log)
        q = (dist * self.support).sum(-1)
        return q

    def _forward_qrdqn(self, x):
        quantiles = self.quantiles(x)
        q = quantiles.mean(1)
        return q

    def _forward_dueling(self, x):
        x = self.features(x)
        x = x.view(-1, self.conv_output_size)
        value = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
        advantage = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
        value, advantage = (
            value.view(
                -1,
                1,
            ),
            advantage.view(-1, self.action_space),
        )
        q = value + advantage - advantage.mean(1, keepdim=True)  # Combine streams
        return q

    def _forward_no_duel(self, x):
        if self.args.attach_task_id:
            task_id = x[:, 0, 0, 0].int()
            task_id = torch.clamp(task_id, 0, self.args.num_processes)
            task_embedding = self.task_embedder(task_id)
            self.task_embedding = task_embedding
            x = x[:, 1:, :, :]
        x = self.features(x)
        if self.args.attach_task_id:
            x = torch.cat([x, task_embedding], axis=1)
        x = x.view(-1, self.conv_output_size)
        q = self.fc_z_v(F.relu(self.fc_h_v(x))).view(-1, self.action_space)
        return q

    def _forward_bootstrap_dqn(self, x, ensemble_i=(), freeze_feature=False, all_head=False):
        x = self.features(x)
        x = x.view(-1, self.conv_output_size)
        if freeze_feature:
            x = x.detach()
        if type(ensemble_i) == int:
            fc_z_v = self.all_fc_z_v[ensemble_i]
            fc_h_v = self.all_fc_h_v[ensemble_i]
            q = fc_z_v(F.relu(fc_h_v(x))).view(-1, self.action_space)
            return q
        all_head_prediction = []
        for fc_z_v, fc_h_v in zip(self.all_fc_z_v, self.all_fc_h_v):
            q = fc_z_v(F.relu(fc_h_v(x))).view(-1, self.action_space)
            all_head_prediction.append(q)
        if all_head:
            return torch.stack(all_head_prediction)
        if len(ensemble_i) == 0:
            return torch.stack(all_head_prediction).mean(axis=0)
        q = []
        for i, j in enumerate(ensemble_i):
            q.append(all_head_prediction[j][i])
        q = torch.stack(q)
        return q

    def _forward_bootstrap_qrdqn(self, x):
        all_quantiles = self.quantiles(x)  # [K, B, atom, action]
        all_quantiles_mean = torch.stack([q.mean(1) for q in all_quantiles])  # [K, B, action]
        mean = torch.mean(all_quantiles_mean, axis=0)
        return mean

    def effective_rank(self, delta=0.01):
        if self.qrdqn and self.qrdqn_bootstrap or self.bootstrap_dqn:
            matrix = self.all_fc_h_v[0]
        else:
            matrix = self.fc_h_v
        _, s, _ = torch.svd(matrix.weight)
        diag_sum = torch.sum(s)
        partial_sum = s[0]
        k = 0
        while (partial_sum / diag_sum) < (1 - delta):
            k += 1
            partial_sum += s[k]
        return k

    def dist(self, x, log=False):
        x = self.features(x)
        x = x.view(-1, self.conv_output_size)
        if self.dueling:
            value = self.fc_z_v(F.relu(self.fc_h_v(x))).view(-1, 1, self.atoms)
            advantage = self.fc_z_a(F.relu(self.fc_h_a(x))).view(-1, self.action_space, self.atoms)
            q_atoms = value + advantage - advantage.mean(1, keepdim=True)
        else:
            q_atoms = self.fc_z_v(F.relu(self.fc_h_v(x))).view(-1, self.action_space, self.atoms)

        if log:
            dist = F.log_softmax(q_atoms, dim=-1)
        else:
            dist = F.softmax(q_atoms, dim=-1)
            dist = dist.clamp(min=1e-4)

        return dist

    def _quantiles_dueling(self, x):
        x = self.features(x)
        x = x.view(-1, self.conv_output_size)
        value = self.fc_z_v(F.relu(self.fc_h_v(x))).view(-1, self.atoms, 1)
        advantage = self.fc_z_a(F.relu(self.fc_h_a(x))).view(-1, self.atoms, self.action_space)
        quantiles = value + advantage - advantage.mean(dim=2, keepdim=True)
        return quantiles

    def _quantiles_no_duel(self, x):
        x = self.features(x)
        x = x.view(-1, self.conv_output_size)
        quantiles = self.fc_z_v(F.relu(self.fc_h_v(x))).view(-1, self.atoms, self.action_space)
        return quantiles

    def _quantile_bootstrap(self, x, freeze_feature=False):
        x = self.features(x)
        if freeze_feature:
            x = x.detach()
        x = x.view(-1, self.conv_output_size)
        all_quantiles = []
        for fc_z_v, fc_h_v in zip(self.all_fc_z_v, self.all_fc_h_v):
            quantiles = fc_z_v(F.relu(fc_h_v(x))).view(-1, self.atoms, self.action_space)
            all_quantiles.append(quantiles)
        return all_quantiles

    def single_quantile(self, x, index):
        x = self.features(x)
        x = x.view(-1, self.conv_output_size)
        fc_h_v = self.all_fc_h_v[index]
        fc_z_v = self.all_fc_z_v[index]
        return fc_z_v(F.relu(fc_h_v(x))).view(-1, self.atoms, self.action_space)

    def reset_noise(self):
        if self.noisy_layers:
            for name, module in self.named_children():
                if "fc" in name:
                    module.reset_noise()
        else:
            pass


class ValueNetwork(nn.Module):
    def __init__(self, args, env):
        super(ValueNetwork, self).__init__()
        self.action_space = env.action_space.n
        self.make_network(args, env)

    def make_network(self, args, env):
        if args.env_name.startswith("MiniGrid"):
            self.features = MiniGridCNN(args, env)
            self.conv_output_size = self.features.image_embedding_size
        else:
            self.features = ImpalaCNN(env.observation_space.shape[0])
            if env.observation_space.shape != (3, 64, 64):
                example_state = torch.randn((1,) + env.observation_space.shape)
                self.conv_output_size = self.features(example_state).shape[1]
            else:
                self.conv_output_size = 2048

        self.fc_h_v = nn.Linear(self.conv_output_size, args.hidden_size)
        self.fc_z_v = nn.Linear(args.hidden_size, 1)

        apply_init_(self.modules())
        self.train()

    def forward(self, x):
        x_v = self.features(x)
        x_v = x_v.view(-1, self.conv_output_size)
        value = self.fc_z_v(F.relu(self.fc_h_v(x_v))).view(-1, 1)
        return value


class AdvantageNetwork(nn.Module):
    def __init__(self, args, env):
        super(AdvantageNetwork, self).__init__()
        self.action_space = env.action_space.n
        self.make_network(args, env)

    def make_network(self, args, env):
        if args.env_name.startswith("MiniGrid"):
            self.features = MiniGridCNN(args, env)
            self.conv_output_size = self.features.image_embedding_size
        else:
            self.features = ImpalaCNN(env.observation_space.shape[0])
            if env.observation_space.shape != (3, 64, 64):
                example_state = torch.randn((1,) + env.observation_space.shape)
                self.conv_output_size = self.features(example_state).shape[1]
            else:
                self.conv_output_size = 2048

        self.fc_h_a = nn.Linear(self.conv_output_size, args.hidden_size)
        self.fc_z_a = nn.Linear(args.hidden_size, 15)

        apply_init_(self.modules())
        self.train()

    def forward(self, x):
        x_a = self.features(x)
        x_a = x_a.view(-1, self.conv_output_size)
        advantage = self.fc_z_a(F.relu(self.fc_h_a(x_a))).view(-1, self.action_space)
        return advantage


class DecoupledDQN(nn.Module):
    def __init__(self, args, env):
        super(DecoupledDQN, self).__init__()
        self.action_space = env.action_space.n
        self.make_network(args, env)

    def make_network(self, args, env):
        self.value = ValueNetwork(args, env)
        self.advantage = AdvantageNetwork(args, env)

    def forward(self, x):
        value = self.value(x)
        advantage = self.advantage(x)
        q = value + advantage
        return advantage, value, q

    def effective_rank(self, delta=0.01):
        _, s, _ = torch.svd(self.value.fc_h_v.weight)
        diag_sum = torch.sum(s)
        partial_sum = s[0]
        k = 0
        while (partial_sum / diag_sum) < (1 - delta):
            k += 1
            partial_sum += s[k]
        return k


class ATCEncoder(nn.Module):
    def __init__(self, env):
        super(ATCEncoder, self).__init__()
        self.conv = ImpalaCNN(env.observation_space.shape[0])
        self.conv_output_size = 2048
        self.head = nn.Linear(self.conv_output_size, 256)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.conv_output_size)
        return self.head(x)

    def encode(self, x):
        x = self.conv(x)
        return x


class ATCContrast(nn.Module):
    def __init__(self):
        super(ATCContrast, self).__init__()
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.W = nn.Linear(256, 256)

    def forward(self, anchor, positive):
        anchor_emb = self.fc2(F.relu(self.fc1(anchor)))
        anchor = anchor + anchor_emb
        pred = self.W(anchor)
        logits = torch.matmul(pred, positive.T)
        logits = logits - torch.max(logits, dim=1, keepdim=True)[0]
        return logits


class ATCDQN(nn.Module):
    def __init__(self, args, env):
        super(ATCDQN, self).__init__()
        self.action_space = env.action_space.n
        self.dueling = args.dueling
        self.c51 = args.c51
        self.qrdqn = args.qrdqn
        self.noisy_layers = False
        if self.c51:
            self.atoms = args.atoms
            self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.device)
            self.V_min = args.V_min
            self.V_max = args.V_max
            self.delta_z = float(self.V_max - self.V_min) / (self.atoms - 1)
        if self.qrdqn:
            self.atoms = 200
        self.make_network(args, env)
        if not self.qrdqn:
            self.forward = (
                self._forward_c51
                if self.c51
                else (self._forward_dueling if self.dueling else self._forward_no_duel)
            )
        else:
            self.quantiles = self._quantiles_dueling if self.dueling else self._quantiles_no_duel
            self.forward = self._forward_qrdqn
            taus = torch.arange(0, self.atoms + 1, device=args.device, dtype=torch.float32) / self.atoms
            self.tau_hats = ((taus[1:] + taus[:-1]) / 2.0).view(1, self.atoms)
            self.c51 = False

    def make_network(self, args, env):
        self.conv_output_size = 2048

        if self.dueling and (self.c51 or self.qrdqn):
            self.fc_h_v = nn.Linear(self.conv_output_size, args.hidden_size)
            self.fc_h_a = nn.Linear(self.conv_output_size, args.hidden_size)
            self.fc_z_v = nn.Linear(args.hidden_size, self.atoms)
            self.fc_z_a = nn.Linear(args.hidden_size, self.action_space * self.atoms)
        elif self.dueling:
            self.fc_h_v = nn.Linear(self.conv_output_size, args.hidden_size)
            self.fc_h_a = nn.Linear(self.conv_output_size, args.hidden_size)
            self.fc_z_v = nn.Linear(args.hidden_size, 1)
            self.fc_z_a = nn.Linear(args.hidden_size, self.action_space)
        elif self.c51 or self.qrdqn:
            self.fc_h_v = nn.Linear(self.conv_output_size, args.hidden_size)
            self.fc_z_v = nn.Linear(args.hidden_size, self.action_space * self.atoms)
        else:
            self.fc_h_v = nn.Linear(self.conv_output_size, args.hidden_size)
            self.fc_z_v = nn.Linear(args.hidden_size, self.action_space)

        apply_init_(self.modules())
        self.train()

    def _forward_c51(self, x, log=False):
        dist = self.dist(x, log)
        q = (dist * self.support).sum(-1)
        return q

    def _forward_qrdqn(self, x):
        quantiles = self.quantiles(x)
        q = quantiles.mean(1)
        return q

    def _forward_dueling(self, x):
        x = x.view(-1, self.conv_output_size)
        value = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
        advantage = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
        value, advantage = (
            value.view(
                -1,
                1,
            ),
            advantage.view(-1, self.action_space),
        )
        q = value + advantage - advantage.mean(1, keepdim=True)  # Combine streams
        return q

    def _forward_no_duel(self, x):
        x = x.view(-1, self.conv_output_size)
        q = self.fc_z_v(F.relu(self.fc_h_v(x))).view(-1, self.action_space)
        return q

    def effective_rank(self, delta=0.01):
        _, s, _ = torch.svd(self.fc_h_v.weight)
        diag_sum = torch.sum(s)
        partial_sum = s[0]
        k = 0
        while (partial_sum / diag_sum) < (1 - delta):
            k += 1
            partial_sum += s[k]
        return k

    def dist(self, x, log=False):
        x = x.view(-1, self.conv_output_size)
        if self.dueling:
            value = self.fc_z_v(F.relu(self.fc_h_v(x))).view(-1, 1, self.atoms)
            advantage = self.fc_z_a(F.relu(self.fc_h_a(x))).view(-1, self.action_space, self.atoms)
            q_atoms = value + advantage - advantage.mean(1, keepdim=True)
        else:
            q_atoms = self.fc_z_v(F.relu(self.fc_h_v(x))).view(-1, self.action_space, self.atoms)

        if log:
            dist = F.log_softmax(q_atoms, dim=-1)
        else:
            dist = F.softmax(q_atoms, dim=-1)
            dist = dist.clamp(min=1e-4)

        return dist

    def _quantiles_dueling(self, x):
        x = x.view(-1, self.conv_output_size)
        value = self.fc_z_v(F.relu(self.fc_h_v(x))).view(-1, self.atoms, 1)
        advantage = self.fc_z_a(F.relu(self.fc_h_a(x))).view(-1, self.atoms, self.action_space)
        quantiles = value + advantage - advantage.mean(dim=2, keepdim=True)
        return quantiles

    def _quantiles_no_duel(self, x):
        x = x.view(-1, self.conv_output_size)
        quantiles = self.fc_z_v(F.relu(self.fc_h_v(x))).view(-1, self.atoms, self.action_space)
        return quantiles

    def reset_noise(self):
        if self.noisy_layers:
            for name, module in self.named_children():
                if "fc" in name:
                    module.reset_noise()
        else:
            pass


class SimpleDQN(nn.Module):
    def __init__(self, args, env):
        super(SimpleDQN, self).__init__()
        self.action_space = env.action_space.n
        self.conv1 = Conv2d_tf(3, 16, kernel_size=7, stride=1, padding="SAME")
        self.pool1 = L2Pool(kernel_size=2, stride=2)
        self.conv2a = Conv2d_tf(16, 32, kernel_size=5, stride=1, padding="SAME")
        self.conv2b = Conv2d_tf(32, 32, kernel_size=5, stride=1, padding="SAME")
        self.pool2 = L2Pool(kernel_size=2, stride=2)
        self.conv3 = Conv2d_tf(32, 32, kernel_size=7, stride=1, padding="SAME")
        self.pool3 = L2Pool(kernel_size=2, stride=2)
        self.conv4 = Conv2d_tf(32, 32, kernel_size=7, stride=1, padding="SAME")
        self.pool4 = L2Pool(kernel_size=2, stride=2)
        self.flat = nn.Flatten()
        self.fc_h1_v = nn.Linear(512, 256)
        self.fc_h1_a = nn.Linear(512, 256)
        self.fc_h2_v = nn.Linear(256, 512)
        self.fc_h2_a = nn.Linear(256, 512)
        self.fc_z_v = nn.Linear(512, 1)
        self.fc_z_a = nn.Linear(512, self.action_space)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2b(F.relu(self.conv2a(x)))))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.flat(x)
        value = F.relu(self.fc_h1_v(x))
        value = F.relu(self.fc_h2_v(value))
        value = self.fc_z_v(value)
        advantage = F.relu(self.fc_h1_a(x))
        advantage = F.relu(self.fc_h2_a(advantage))
        advantage = self.fc_z_a(advantage)
        value, advantage = (
            value.view(
                -1,
                1,
            ),
            advantage.view(-1, self.action_space),
        )
        q = value + advantage - advantage.mean(1, keepdim=True)
        return q

    def effective_rank(self, delta=0.01):
        _, s, _ = torch.svd(self.fc_h_v.weight)
        diag_sum = torch.sum(s)
        partial_sum = s[0]
        k = 0
        while (partial_sum / diag_sum) < (1 - delta):
            k += 1
            partial_sum += s[k]
        return k


class TwinnedDQN(nn.Module):
    def __init__(self, args, env):
        super(TwinnedDQN, self).__init__()
        self.q1 = DQN(args, env)
        self.q2 = DQN(args, env)

    def forward(self, x):
        q1 = self.q1(x)
        q2 = self.q2(x)

        return q1, q2


class SAC(nn.Module):
    def __init__(self, args, env):
        super(SAC, self).__init__()
        self.action_space = env.action_space.n
        self.features = ImpalaCNN(env.observation_space.shape[0])
        self.conv_output_size = 2048
        self.fc1 = nn.Linear(self.conv_output_size, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, self.action_space)
        apply_init_(self.modules())
        self.train()

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.conv_output_size)
        action_logits = self.fc2(F.relu(self.fc1(x)))
        greedy_action = torch.argmax(action_logits, dim=1, keepdim=True)
        return greedy_action

    def act(self, x):
        return self.forward(x)

    def sample(self, x):
        x = self.features(x)
        x = x.view(-1, self.conv_output_size)
        action_logits = self.fc2(F.relu(self.fc1(x)))
        action_probs = F.softmax(action_logits, dim=1)
        action_dist = Categorical(action_probs)
        action = action_dist.sample().view(-1, 1)

        # Avoid numerical instability.
        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return action, action_probs, log_action_probs


class Conv_Q(nn.Module):
    def __init__(self, frames, num_actions):
        super(Conv_Q, self).__init__()
        self.c1 = nn.Conv2d(frames, 32, kernel_size=8, stride=4)
        self.c2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.c3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.l1 = nn.Linear(3136, 512)
        self.l2 = nn.Linear(512, num_actions)

    def forward(self, state):
        q = F.relu(self.c1(state))
        q = F.relu(self.c2(q))
        q = F.relu(self.c3(q))
        q = F.relu(self.l1(q.reshape(-1, 3136)))
        return self.l2(q)


class OrderClassifier(nn.Module):
    def __init__(self):
        super(OrderClassifier, self).__init__()
        self.main = nn.Sequential(
            Flatten(),
            init_relu_(nn.Linear(1024, 16)),
            nn.ReLU(),
            init_(nn.Linear(16, 2)),
            nn.Softmax(dim=1),
        )
        self.train()

    def forward(self, emb):
        x = self.main(emb)
        return x
