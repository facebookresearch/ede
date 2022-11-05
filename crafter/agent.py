# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_

from model import DQN, QRDQN


class Agent():
    def __init__(self, args, env):
        self.action_space = env.action_space()
        self.atoms = args.atoms
        self.Vmin = args.V_min
        self.Vmax = args.V_max
        self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(
            device=args.device)  # Support (range) of z
        self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
        self.batch_size = args.batch_size
        self.n = args.multi_step
        self.discount = args.discount
        self.norm_clip = args.norm_clip

        self.online_net = DQN(args, self.action_space).to(device=args.device)
        if args.model:  # Load pretrained model if provided
            if os.path.isfile(args.model):
                # Always load tensors onto CPU by default, will shift to GPU if necessary
                state_dict = torch.load(args.model, map_location='cpu')
                if 'conv1.weight' in state_dict.keys():
                    for old_key, new_key in (('conv1.weight', 'convs.0.weight'), ('conv1.bias', 'convs.0.bias'), ('conv2.weight', 'convs.2.weight'), ('conv2.bias', 'convs.2.bias'), ('conv3.weight', 'convs.4.weight'), ('conv3.bias', 'convs.4.bias')):
                        # Re-map state dict for old pretrained models
                        state_dict[new_key] = state_dict[old_key]
                        # Delete old keys for strict load_state_dict
                        del state_dict[old_key]
                self.online_net.load_state_dict(state_dict)
                print("Loading pretrained model: " + args.model)
            else:  # Raise error if incorrect model path provided
                raise FileNotFoundError(args.model)

        self.online_net.train()

        self.target_net = DQN(args, self.action_space).to(device=args.device)
        self.update_target_net()
        self.target_net.train()
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimiser = optim.Adam(
            self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)

    # Resets noisy weights in all linear layers (of online net only)
    def reset_noise(self):
        self.online_net.reset_noise()

    # Acts based on single state (no batch)
    def act(self, state):
        with torch.no_grad():
            return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).argmax(1).item()

    def explore(self, state, **kwargs):
        return self.act(state), {}

    # Acts with an ε-greedy policy (used for evaluation only)
    # High ε can reduce evaluation scores drastically
    def act_e_greedy(self, state, epsilon=0.001):
        return np.random.randint(0, self.action_space) if np.random.random() < epsilon else self.act(state)

    def learn(self, mem):
        # Sample transitions
        idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(
            self.batch_size)

        # Calculate current state probabilities (online network noise already sampled)
        # Log probabilities log p(s_t, ·; θonline)
        log_ps = self.online_net(states, log=True)
        # log p(s_t, a_t; θonline)
        log_ps_a = log_ps[range(self.batch_size), actions]

        with torch.no_grad():
            # Calculate nth next state probabilities
            # Probabilities p(s_t+n, ·; θonline)
            pns = self.online_net(next_states)
            # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
            dns = self.support.expand_as(pns) * pns
            # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
            argmax_indices_ns = dns.sum(2).argmax(1)
            self.target_net.reset_noise()  # Sample new target net noise
            # Probabilities p(s_t+n, ·; θtarget)
            pns = self.target_net(next_states)
            # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
            pns_a = pns[range(self.batch_size), argmax_indices_ns]

            # Compute Tz (Bellman operator T applied to z)
            # Tz = R^n + (γ^n)z (accounting for terminal states)
            Tz = returns.unsqueeze(
                1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)
            # Clamp between supported values
            Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)
            # Compute L2 projection of Tz onto fixed support z
            b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
            # Fix disappearing probability mass when l = b = u (b is int)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1

            # Distribute probability of Tz
            m = states.new_zeros(self.batch_size, self.atoms)
            offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(
                1).expand(self.batch_size, self.atoms).to(actions)
            m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a *
                                                             (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a *
                                                             (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

        # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
        loss = -torch.sum(m * log_ps_a, 1)
        self.online_net.zero_grad()
        # Backpropagate importance-weighted minibatch loss
        (weights * loss).mean().backward()
        # Clip gradients by L2 norm
        clip_grad_norm_(self.online_net.parameters(), self.norm_clip)
        self.optimiser.step()

        # Update priorities of sampled transitions
        mem.update_priorities(idxs, loss.detach().cpu().numpy())

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    # Save model parameters on current device (don't move model between devices)
    def save(self, path, name='model.pth'):
        torch.save(self.online_net.state_dict(), os.path.join(path, name))

    # Evaluates Q-value based on single state (no batch)
    def evaluate_q(self, state):
        with torch.no_grad():
            return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).max(1)[0].item()

    def train(self):
        self.online_net.train()

    def eval(self):
        self.online_net.eval()




class QrdqnAgent():
    def __init__(self, args, env):
        self.action_space = env.action_space()
        self.n_quantiles = args.n_quantiles
        self.batch_size = args.batch_size
        self.n_step = args.multi_step
        self.gamma = args.discount
        self.norm_clip = args.norm_clip

        self.ucb_c = args.ucb_c
        self.explore_strat = args.explore_strat
        self.per = args.per
        self.bootstrapped_qrdqn = args.bootstrapped_qrdqn
        self.use_wbb = args.use_wbb
        self.double_qrdqn = args.double_qrdqn
        self.qrdqn_always_train_feat = args.qrdqn_always_train_feat
        self.use_average_target = args.use_average_target
        self.min_priority = 1e-2

        ############ Epsilon greedy ########
        epsilon_start = 1.0
        epsilon_final = args.end_eps
        epsilon_decay = args.eps_decay_period
        def foo(t):
            return epsilon_final + (epsilon_start - epsilon_final) * np.exp(
            -1.0 * (t - args.learn_start) / epsilon_decay
        )
        self.epsilon = foo
        ####################################

        self.online_net = QRDQN(args, self.action_space).to(device=args.device)
        if args.model:  # Load pretrained model if provided
            if os.path.isfile(args.model):
                # Always load tensors onto CPU by default, will shift to GPU if necessary
                state_dict = torch.load(args.model, map_location='cpu')
                if 'conv1.weight' in state_dict.keys():
                    for old_key, new_key in (('conv1.weight', 'convs.0.weight'), ('conv1.bias', 'convs.0.bias'), ('conv2.weight', 'convs.2.weight'), ('conv2.bias', 'convs.2.bias'), ('conv3.weight', 'convs.4.weight'), ('conv3.bias', 'convs.4.bias')):
                        # Re-map state dict for old pretrained models
                        state_dict[new_key] = state_dict[old_key]
                        # Delete old keys for strict load_state_dict
                        del state_dict[old_key]
                self.online_net.load_state_dict(state_dict)
                print("Loading pretrained model: " + args.model)
            else:  # Raise error if incorrect model path provided
                raise FileNotFoundError(args.model)

        self.online_net.train()

        self.target_net = QRDQN(args, self.action_space).to(device=args.device)
        self.update_target_net()
        self.target_net.train()
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimiser = optim.Adam(
            self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)

    # Resets noisy weights in all linear layers (of online net only)
    def reset_noise(self):
        self.online_net.reset_noise()

    # Acts based on single state (no batch)
    def act(self, state):
        with torch.no_grad():
            return self.online_net(state.unsqueeze(0)).argmax(1).item()

    def explore(self, state, t, **kwargs):
        if self.explore_strat == "egreedy":
            cur_epsilon = self.epsilon(t)
            return self.act_e_greedy(state, cur_epsilon), {}
        elif self.explore_strat == "ucb":
            action, mean, eps_var, ale_var = self.act_ucb(state)
            return action, {'mean': mean, 'eps_var': eps_var, 'ale_var': ale_var}
        elif self.explore_strat == "thompson":
            action, mean, eps_var, ale_var = self.act_thompson_sampling(state)
            return action, {'mean': mean, 'eps_var': eps_var, 'ale_var': ale_var}
        else:
            return self.act(state), {}

    # Acts with an ε-greedy policy (used for evaluation only)
    # High ε can reduce evaluation scores drastically
    def act_e_greedy(self, state, epsilon=0.001):
        return np.random.randint(0, self.action_space) if np.random.random() < epsilon else self.act(state)

    def act_ucb(self, state):
        decay = self.ucb_c
        mean, eps_var, ale_var = self.get_bootstrapped_uncertainty(
            state.unsqueeze(0))
        eps_var, ale_var = torch.sqrt(eps_var), torch.sqrt(ale_var)
        mean = mean.mean(axis=1)
        value = mean + decay * eps_var
        action = value.argmax(1).item()
        return action, mean, eps_var, ale_var

    def act_thompson_sampling(self, state):
        decay = self.ucb_c
        mean, eps_var, ale_var = self.get_bootstrapped_uncertainty(
            state.unsqueeze(0))
        eps_var, ale_var = torch.sqrt(eps_var), torch.sqrt(ale_var)
        mean = mean.mean(axis=1)
        epsilon = np.random.normal(size=(mean.size(0), mean.size(1)))
        epsilon = torch.from_numpy(epsilon).to(state.device)
        value = mean + decay * epsilon * eps_var
        action = value.argmax(1).item()
        return action, mean, eps_var, ale_var

    def get_bootstrapped_uncertainty(self, state):
        with torch.no_grad():
            all_quantiles = self.online_net.quantiles(
                state)  # [K, B, atom, action]
            return self._compute_uncertainty(all_quantiles)

    def get_get_bootstrapped_target_uncertainty(self, state):
        with torch.no_grad():
            all_quantiles = self.target_net.quantiles(state)
            action_mean, eps_var, ale_var = self._compute_uncertainty(
                all_quantiles)
            return all_quantiles, action_mean, eps_var, ale_var

    def _compute_uncertainty(self, all_quantiles):
        all_quantiles = torch.permute(torch.stack(
            all_quantiles), (1, 0, 2, 3))  # [B, K, atom, action]
        eps_var = torch.var(all_quantiles, dim=1)  # [B, atom, action]
        eps_var = torch.mean(eps_var, dim=1)  # [B, action]
        ale_var = torch.mean(all_quantiles, dim=1)  # [B, atom, action]
        ale_var = torch.var(ale_var, axis=1)  # [B, action]
        action_mean = torch.mean(all_quantiles, axis=2)  # [B, K, action]
        return action_mean, eps_var, ale_var

    def learn(self, mem):

        if self.bootstrapped_qrdqn:
            idxs, loss, weights = self._loss_qrdqn_bootstrap_multi_head(mem)
        else:
            idxs, loss, weights = self._loss_qrdqn(mem)

        self.online_net.zero_grad()
        # Backpropagate importance-weighted minibatch loss
        # (weights * loss).mean().backward()
        loss.backward()
        # Clip gradients by L2 norm
        clip_grad_norm_(self.online_net.parameters(), self.norm_clip)
        self.optimiser.step()

        # Update priorities of sampled transitions
        if self.per:
            mem.update_priorities(idxs, loss.detach().cpu().numpy())

    def _loss_qrdqn_bootstrap_single_head_from_transition(
        self,
        head_idx,
        state,
        action,
        next_state,
        reward,
        not_done,
        weights,
        train_feature
    ):
        with torch.no_grad():
            if self.use_wbb:
                all_next_quantiles, mean, eps_var, ale_var = self.get_get_bootstrapped_target_uncertainty(
                    next_state)
                mean = mean.mean(axis=1)  # (B, A)
                eps_std = torch.sqrt(eps_var)  # (B, A)
                next_greedy_actions = mean.argmax(
                    dim=1, keepdim=True).expand(self.batch_size, 1)
                target_std = eps_std.gather(
                    dim=1, index=next_greedy_actions)
                weights = torch.sigmoid(-target_std *
                                        self.wbb_temperature) + 0.5
            else:
                all_next_quantiles = self.target_net.quantiles(
                    next_state, freeze_feature=(not train_feature))

            # select the current head index
            next_quantiles = all_next_quantiles[head_idx]

            if self.use_average_target:
                target_quantiles = torch.mean(
                    torch.stack(all_next_quantiles), axis=0)
                next_quantiles = target_quantiles
            else:
                target_quantiles = next_quantiles

            if self.double_qrdqn:
                selection_quantile = self.online_net.quantiles(
                    next_state, freeze_feature=(not train_feature))
                selection_quantile = selection_quantile[head_idx]
            else:
                selection_quantile = next_quantiles

            next_greedy_actions = selection_quantile.mean(
                dim=1, keepdim=True).argmax(dim=2, keepdim=True)
            next_greedy_actions = next_greedy_actions.expand(
                self.batch_size, self.online_net.n_quantiles, 1)
            greedy_next_quantiles = target_quantiles.gather(
                dim=2, index=next_greedy_actions).squeeze(dim=2)
            target_quantiles = reward.unsqueeze(1) + not_done * \
                self.gamma ** self.n_step * greedy_next_quantiles

        current_quantiles = self.online_net.quantiles(
            state, freeze_feature=(not train_feature))[head_idx]
        actions = action[..., None, None].long().expand(
            self.batch_size, self.online_net.n_quantiles, 1)
        current_quantiles = torch.gather(
            current_quantiles, dim=2, index=actions).squeeze(dim=2)

        loss, td = self.quantile_huber_loss(
            current_quantiles, target_quantiles, weights)
        return loss, td

    def _loss_qrdqn_bootstrap_single_head(self, replay_buffer, head_idx, train_feature):
        # Sample transitions
        ind, state, action, reward, next_state, not_done, weights = replay_buffer.sample(
            self.batch_size)

        loss, td = self._loss_qrdqn_bootstrap_single_head_from_transition(
            head_idx, state, action, next_state, reward, not_done, weights, train_feature
        )

        if self.per:
            priority = (
                td.abs()
                .clamp(min=self.min_priority)
                .detach()
                .sum(dim=1)
                .mean(dim=1, keepdim=True)
                .cpu()
                .numpy()
                .flatten()
            )
            return ind, loss, priority

        return ind, loss, weights

    def _loss_qrdqn_bootstrap_multi_head(self, replay_buffer, head_idx=None):
        total_loss = 0
        head_idx = np.random.randint(self.online_net.n_ensemble)
        all_ind, all_priority = [], []
        for i in range(self.target_net.n_ensemble):
            train_head = i == head_idx 
            train_feature = train_head or self.qrdqn_always_train_feat
            ind, loss, priority = self._loss_qrdqn_bootstrap_single_head(
                replay_buffer, i, train_feature)
            total_loss += loss
            all_ind.append(ind)
            all_priority.append(priority)
        if self.per:
            ind, priority = np.concatenate(
                all_ind, axis=0), np.concatenate(all_priority, axis=0)
        return ind, total_loss, priority

    def _loss_qrdqn(self, replay_buffer):
        # Sample transitions
        ind, state, action, reward, next_state, not_done, weights = replay_buffer.sample(
            self.batch_size)
        with torch.no_grad():
            next_quantiles = self.target_net.quantiles(next_state)
            next_greedy_actions = next_quantiles.mean(
                dim=1, keepdim=True).argmax(dim=2, keepdim=True)
            next_greedy_actions = next_greedy_actions.expand(
                self.batch_size, self.online_net.n_quantiles, 1)
            next_quantiles = next_quantiles.gather(
                dim=2, index=next_greedy_actions).squeeze(dim=2)
            target_quantiles = reward.unsqueeze(1) + not_done * \
                self.gamma ** self.n_step * next_quantiles

        current_quantiles = self.online_net.quantiles(state)
        actions = action[..., None, None].long().expand(
            self.batch_size, self.online_net.n_quantiles, 1)
        current_quantiles = torch.gather(
            current_quantiles, dim=2, index=actions).squeeze(dim=2)

        loss, td = self.quantile_huber_loss(
            current_quantiles, target_quantiles, weights)

        priority = (
            td.abs()
            .clamp(min=self.min_priority)
            .detach()
            .sum(dim=1)
            .mean(dim=1, keepdim=True)
            .cpu()
            .numpy()
            .flatten()
        )

        return ind, loss, priority

    def huber(self, td_errors, kappa=1.0):
        return torch.where(
            td_errors.abs() <= kappa, 0.5 * td_errors.pow(2), kappa * (td_errors.abs() - 0.5 * kappa)
        )

    def quantile_huber_loss(self, current_quantiles, target_quantiles, weights):
        weights = weights[..., None]
        n_quantiles = current_quantiles.shape[-1]
        cum_prob = (
            torch.arange(n_quantiles, device=current_quantiles.device,
                         dtype=torch.float) + 0.5
        ) / n_quantiles
        cum_prob = cum_prob.view(1, -1, 1)

        td = target_quantiles.unsqueeze(-2) - current_quantiles.unsqueeze(-1)
        huber_loss = self.huber(td)
        loss = torch.abs(cum_prob - (td.detach() < 0).float()) * huber_loss
        loss = (loss.sum(dim=-2) * weights).mean()

        return loss, td

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    # Save model parameters on current device (don't move model between devices)
    def save(self, path, name='model.pth'):
        torch.save(self.online_net.state_dict(), os.path.join(path, name))

    # Evaluates Q-value based on single state (no batch)
    def evaluate_q(self, state):
        with torch.no_grad():
            return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).max(1)[0].item()

    def train(self):
        self.online_net.train()

    def eval(self):
        self.online_net.eval()
