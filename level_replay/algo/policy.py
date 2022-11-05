# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
from statistics import quantiles
import torch
import torch.nn.functional as F
from level_replay.algo.dqn import DQN, SimpleDQN, Conv_Q, ATCDQN, ATCEncoder, ATCContrast
from torch.nn.utils import clip_grad_norm_

import numpy as np

from level_replay.utils import seed


class DQNAgent(object):
    def __init__(self, args, env):
        self.args = args
        self.device = args.device
        self.action_space = env.action_space.n
        self.batch_size = args.batch_size
        self.norm_clip = args.norm_clip
        self.gamma = args.gamma

        if args.simple_dqn:
            self.Q = SimpleDQN(args, env).to(self.device)
        else:
            self.Q = DQN(args, env).to(self.device)

        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer = getattr(torch.optim, args.optimizer)(
            self.Q.parameters(), **args.optimizer_parameters
        )
        for param in self.Q_target.parameters():
            param.requires_grad = False

        self.PER = args.PER
        self.n_step = args.multi_step

        self.min_priority = args.min_priority

        # Target update rule
        self.maybe_update_target = (
            self.polyak_target_update if args.polyak_target_update else self.copy_target_update
        )
        self.target_update_frequency = int(args.target_update // args.num_processes)
        self.tau = args.tau

        # Evaluation hyper-parameters
        self.state_shape = (-1,) + env.observation_space.shape
        self.eval_eps = args.eval_eps

        # For seed bar chart
        self.track_seed_weights = args.track_seed_weights
        if self.track_seed_weights:
            self.seed_weights = {
                i: 0 for i in range(args.start_level, args.start_level + args.num_train_seeds)
            }

        # Uncertainty weighted bellman backup
        self.use_wbb = args.use_wbb
        self.wbb_temperature = args.wbb_temperature

        # Bootstrap DQN
        self.bootstrap_dqn = args.bootstrap_dqn
        self.current_bootstrap_head = np.random.randint(args.n_ensemble, size=args.num_processes)

        # qrdqn related
        self.double_qrdqn = args.double_qrdqn

        if self.Q.c51:
            self.loss = self._loss_c51
        elif self.Q.qrdqn and self.Q.qrdqn_bootstrap:
            if not args.bootstrap_dqn:
                self.loss = self._loss_qrdqn_bootstrap_multi_head
                self.use_average_target = args.use_average_target
            else:
                self.loss = self._loss_qrdqn_bootstrap_single_head
                self.use_average_target = False
        elif self.Q.qrdqn:
            if args.drq:
                self.loss = self._loss_qrdqn_aug
            else:
                self.loss = self._loss_qrdqn
            self.kappa = 1.0
            self.cumulative_density = np.array((2 * np.arange(self.Q.atoms) + 1) / (2.0 * self.Q.atoms))
            self.cumulative_density = torch.from_numpy(self.cumulative_density).to(self.device)
        elif args.drq or args.autodrq:
            self.loss = self._loss_aug
        else:
            self.loss = self._loss

        # Number of training iterations
        self.iterations = 0

    def select_action(self, state, eps=0.1, eval=False):
        with torch.no_grad():
            if eval:
                self.Q.eval()
            if self.bootstrap_dqn and not eval:
                q = self.Q(state, self.current_bootstrap_head)
            else:
                q = self.Q(state)
            action = q.argmax(1).reshape(-1, 1)
            max_q = q.max(1)[0]
            mean_q = q.mean(1)
            v = (1 - eps) * max_q + eps * mean_q
            if eval:
                self.Q.train()
            return action, v

    # def sample_action(self, state, temperature=0.1):
    #     with torch.no_grad():
    #         mean, eps_var, ale_var = self.get_bootstrapped_uncertainty(state)
    #         mean = mean.mean(axis=1) / temperature
    #         prob = F.softmax(mean, dim=1)
    #         samples = torch.multinomial(prob, 1)
    #         return samples, None

    def sample_action(self, state, c):
        with torch.no_grad():
            mean, eps_var, ale_var = self.get_bootstrapped_uncertainty(state)
            eps_var, ale_var = torch.sqrt(eps_var), torch.sqrt(ale_var)
            mean = mean.mean(axis=1)
            eps_var = eps_var * torch.randn(eps_var.shape, device=eps_var.device)
            value = mean + c * eps_var
            action = value.argmax(1).reshape(-1, 1)
            return action


    def get_quantile(self, state):
        with torch.no_grad():
            quantiles = self.Q.quantiles(state)  # (B, atom, action)
            mean = quantiles.mean(1, keepdim=True)
            var = quantiles.var(1)
            upper_quantile = quantiles[:, self.Q.atoms//2:]
            upper_var = (upper_quantile - mean)**2
            upper_var = upper_var.mean(1)
            return quantiles, mean.squeeze(1), var, upper_var

    def get_bootstrapped_uncertainty(self, state):
        with torch.no_grad():
            all_quantiles = self.Q.quantiles(state)  # [K, B, atom, action]
            return self._compute_uncertainty(all_quantiles)

    def get_get_bootstrapped_target_uncertainty(self, state):
        with torch.no_grad():
            all_quantiles = self.Q_target.quantiles(state)
            action_mean, eps_var, ale_var = self._compute_uncertainty(all_quantiles)
            return all_quantiles, action_mean, eps_var, ale_var

    def _compute_uncertainty(self, all_quantiles):
        all_quantiles = torch.permute(torch.stack(all_quantiles), (1, 0, 2, 3))  # [B, K, atom, action]
        eps_var = torch.var(all_quantiles, dim=1)  # [B, atom, action]
        eps_var = torch.mean(eps_var, dim=1)  # [B, action]
        ale_var = torch.mean(all_quantiles, dim=1)  # [B, atom, action]
        ale_var = torch.var(ale_var, axis=1)  # [B, action]
        action_mean = torch.mean(all_quantiles, axis=2)  # [B, K, action]
        return action_mean, eps_var, ale_var

    def get_bootstrap_dqn_values(self, state):
        with torch.no_grad():
            q = self.Q(state, all_head=True)
            mean = q.mean(axis=0)
            std = q.std(axis=0)
            return mean, std

    def get_value(self, state, eps=0.1):
        with torch.no_grad():
            q = self.Q(state)
            max_q = q.max(1)[0]
            mean_q = q.mean(1)
            v = (1 - eps) * max_q + eps * mean_q
            return v

    def advantage(self, state, eps):
        q = self.Q(state)
        max_q = q.max(1)[0]
        mean_q = q.mean(1)
        v = (1 - eps) * max_q + eps * mean_q
        return q - v.repeat(q.shape[1]).reshape(-1, q.shape[1])

    def train(self, replay_buffer):

        if self.args.noisy_layers:
            self.Q.reset_noise()
            self.Q_target.reset_noise()

        ind, loss, priority = self.loss(replay_buffer)

        self.Q_optimizer.zero_grad()
        loss.backward()  # Backpropagate importance-weighted minibatch loss
        clip_grad_norm_(self.Q.parameters(), self.norm_clip)  # Clip gradients by L2 norm
        grad_magnitude = list(self.Q.named_parameters())[0][1].grad.clone().norm()
        # grad_magnitude = list(self.Q.named_parameters())[-2][1].grad.clone().norm()
        self.Q_optimizer.step()

        # Update target network by polyak or full copy every X iterations.
        self.iterations += 1
        self.maybe_update_target()

        if self.PER:
            replay_buffer.update_priority(ind, priority)

        return loss, grad_magnitude, priority

    def _loss(self, replay_buffer):
        state, action, next_state, reward, not_done, seeds, ind, weights = replay_buffer.sample()

        self.update_seed_weights(seeds, weights)

        with torch.no_grad():
            next_action = self.Q(next_state).argmax(1).reshape(-1, 1)
            target_Q = reward + not_done * (self.gamma ** self.n_step) * self.Q_target(next_state).gather(
                1, next_action
            )
        loss = 0
        current_Q = self.Q(state).gather(1, action)
        if self.args.attach_task_id:
            task_embedding = self.Q.task_embedding
            kl = (task_embedding**2).sum()
            loss += 0.1 * kl

        loss += (weights * F.smooth_l1_loss(current_Q, target_Q, reduction="none")).mean()
        priority = (current_Q - target_Q).abs().clamp(min=self.min_priority).cpu().data.numpy().flatten()

        return ind, loss, priority

    def _loss_c51(self, replay_buffer):
        state, action, next_state, reward, not_done, seeds, ind, weights = replay_buffer.sample()

        self.update_seed_weights(seeds, weights)

        log_prob = self.Q.dist(state, log=True)
        log_prob_a = log_prob[range(self.batch_size), action]

        with torch.no_grad():
            next_prob = self.Q.dist(next_state)
            next_dist = self.Q.support.expand_as(next_prob) * next_prob
            argmax_idx = next_dist.sum(-1).argmax(1)

            if self.Q_target.noisy_layers:
                self.Q_target.reset_noise()

            next_prob = self.Q_target.dist(next_state)
            next_prob_a = next_prob[range(self.batch_size), argmax_idx]

            Tz = reward.unsqueeze(1) + not_done * (self.gamma ** self.n_step) * self.Q.support.unsqueeze(0)
            Tz = Tz.clamp(min=self.Q.V_min, max=self.Q.V_max)

            b = (Tz - self.Q.V_min) / self.Q.delta_z
            lower, upper = b.floor().to(torch.int64), b.ceil().to(torch.int64)

            lower[(upper > 0) * (lower == upper)] -= 1
            upper[(lower < (self.Q.atoms - 1)) * (lower == upper)] += 1

            m = state.new_zeros(self.batch_size, self.Q.atoms)
            offset = (
                torch.linspace(0, ((self.batch_size - 1) * self.Q.atoms), self.batch_size)
                .unsqueeze(1)
                .expand(self.batch_size, self.Q.atoms)
                .to(action)
            )
            m.view(-1).index_add_(0, (lower + offset).view(-1), (next_prob_a * (upper.float() - b)).view(-1))
            m.view(-1).index_add_(0, (upper + offset).view(-1), (next_prob_a * (b - lower.float())).view(-1))

        KL = -torch.sum(m * log_prob_a, 1)

        self.Q_optimizer.zero_grad()
        loss = (weights * KL).mean()
        priority = KL.clamp(min=self.min_priority).cpu().data.numpy().flatten()

        return ind, loss, priority

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
                all_next_quantiles, mean, eps_var, ale_var = self.get_get_bootstrapped_target_uncertainty(next_state)
                mean = mean.mean(axis=1)  # (B, A)
                eps_std = torch.sqrt(eps_var)  # (B, A)
                next_greedy_actions = mean.argmax(dim=1, keepdim=True).expand(self.batch_size, 1)
                target_std = eps_std.gather(dim=1, index=next_greedy_actions)
                weights = torch.sigmoid(-target_std * self.wbb_temperature) + 0.5
            else:
                all_next_quantiles = self.Q_target.quantiles(next_state, freeze_feature=(not train_feature))

            next_quantiles = all_next_quantiles[head_idx]  # select the current head index

            if self.use_average_target:
                target_quantiles = torch.mean(torch.stack(all_next_quantiles), axis=0)
                next_quantiles = target_quantiles
            else:
                target_quantiles = next_quantiles

            if self.double_qrdqn:
                selection_quantile = self.Q.quantiles(next_state, freeze_feature=(not train_feature))
                selection_quantile = selection_quantile[head_idx]
            else:
                selection_quantile = next_quantiles

            next_greedy_actions = selection_quantile.mean(dim=1, keepdim=True).argmax(dim=2, keepdim=True)
            next_greedy_actions = next_greedy_actions.expand(self.batch_size, self.Q.atoms, 1)
            greedy_next_quantiles = target_quantiles.gather(dim=2, index=next_greedy_actions).squeeze(dim=2)
            target_quantiles = reward + not_done * self.gamma ** self.n_step * greedy_next_quantiles

        current_quantiles = self.Q.quantiles(state, freeze_feature=(not train_feature))[head_idx]
        actions = action[..., None].long().expand(self.batch_size, self.Q.atoms, 1)
        current_quantiles = torch.gather(current_quantiles, dim=2, index=actions).squeeze(dim=2)

        loss, td = self.quantile_huber_loss(current_quantiles, target_quantiles, weights)
        return loss, td

    def _loss_qrdqn_bootstrap_single_head(self, replay_buffer, head_idx, train_feature):
        if self.args.drq and train_feature:
            state, action, next_state, reward, not_done, seeds, ind, weights, aug_state, aug_next_state = replay_buffer.sample()
        else:
            state, action, next_state, reward, not_done, seeds, ind, weights = replay_buffer.sample(aug=False)

        self.update_seed_weights(seeds, weights)

        loss, td = self._loss_qrdqn_bootstrap_single_head_from_transition(
            head_idx, state, action, next_state, reward, not_done, weights, train_feature
        )
        if self.args.drq and train_feature:
            aug_loss, _ = self._loss_qrdqn_bootstrap_single_head_from_transition(
                head_idx, aug_state, action, aug_next_state, reward, not_done, weights, train_feature
            )
            loss += aug_loss

        if self.PER:
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

    def _anchor_loss(self):
        total_loss = 0
        for head, anchor in zip(self.Q.all_fc_h_v, self.Q.anchors_all_fc_h_v):
            diff=[]
            for i, p in enumerate(head.parameters()):
                diff.append(torch.sum((p - anchor[i].detach())**2))
            diff = torch.stack(diff).sum()
            total_loss += diff
        for head, anchor in zip(self.Q.all_fc_z_v, self.Q.anchors_all_fc_z_v):
            diff=[]
            for i, p in enumerate(head.parameters()):
                diff.append(torch.sum((p - anchor[i].detach())**2))
            diff = torch.stack(diff).sum()
            total_loss += diff
        return self.args.anchor_loss * total_loss

    def _loss_qrdqn_bootstrap_multi_head(self, replay_buffer, head_idx=None):
        if self.args.uadqn:
            return self._loss_uadqn(replay_buffer)
        total_loss = 0
        head_idx = np.random.randint(self.Q.n_ensemble)
        all_ind, all_priority = [], []
        for i in range(self.Q_target.n_ensemble):
            train_feature = (i == head_idx) or self.args.qrdqn_always_train_feat
            ind, loss, priority = self._loss_qrdqn_bootstrap_single_head(replay_buffer, i, train_feature)
            total_loss += loss
            all_ind.append(ind)
            all_priority.append(priority)
        if self.args.anchor_loss > 0:
            total_loss += self._anchor_loss()
        if self.PER:
            ind, priority = np.concatenate(all_ind, axis=0), np.concatenate(all_priority, axis=0)
        return ind, total_loss, priority

    def _loss_uadqn(self, replay_buffer):
        state, action, next_state, reward, not_done, seeds, ind, weights = replay_buffer.sample(aug=False)
        with torch.no_grad():
            all_next_quantiles = self.Q_target.quantiles(next_state, freeze_feature=False)
            target_quantiles = torch.mean(torch.stack(all_next_quantiles), axis=0)  # average target
            next_quantiles = target_quantiles
            selection_quantile = next_quantiles
            next_greedy_actions = selection_quantile.mean(dim=1, keepdim=True).argmax(dim=2, keepdim=True)
            next_greedy_actions = next_greedy_actions.expand(self.batch_size, self.Q.atoms, 1)
            all_greedy_next_quantiles = 0
            for tq in all_next_quantiles:
                all_greedy_next_quantiles += tq.gather(dim=2, index=next_greedy_actions).squeeze(dim=2)
            greedy_next_quantiles = all_greedy_next_quantiles / len(all_next_quantiles)
            target_quantiles = reward + not_done * self.gamma ** self.n_step * greedy_next_quantiles

        n_head = len(all_next_quantiles)
        total_loss, total_td = 0, 0
        actions = action[..., None].long().expand(self.batch_size, self.Q.atoms, 1)
        all_online_quantiles = self.Q.quantiles(state, freeze_feature=False)
        for head_idx in range(n_head):
            current_quantiles = all_online_quantiles[head_idx]
            current_quantiles = torch.gather(current_quantiles, dim=2, index=actions).squeeze(dim=2)
            loss, td = self.quantile_huber_loss(current_quantiles, target_quantiles, weights)
            total_loss += loss
            total_td += td
        if self.args.anchor_loss > 0:
            total_loss += self._anchor_loss()
        return ind, total_loss, total_td

    def _loss_qrdqn(self, replay_buffer):
        state, action, next_state, reward, not_done, seeds, ind, weights = replay_buffer.sample()

        self.update_seed_weights(seeds, weights)

        with torch.no_grad():
            next_quantiles = self.Q_target.quantiles(next_state)
            next_greedy_actions = next_quantiles.mean(dim=1, keepdim=True).argmax(dim=2, keepdim=True)
            next_greedy_actions = next_greedy_actions.expand(self.batch_size, self.Q.atoms, 1)
            next_quantiles = next_quantiles.gather(dim=2, index=next_greedy_actions).squeeze(dim=2)
            target_quantiles = reward + not_done * self.gamma ** self.n_step * next_quantiles

        current_quantiles = self.Q.quantiles(state)
        actions = action[..., None].long().expand(self.batch_size, self.Q.atoms, 1)
        current_quantiles = torch.gather(current_quantiles, dim=2, index=actions).squeeze(dim=2)

        loss, td = self.quantile_huber_loss(current_quantiles, target_quantiles, weights)

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

    def _loss_qrdqn_aug(self, replay_buffer):
        (
            state,
            action,
            next_state,
            reward,
            not_done,
            seeds,
            ind,
            weights,
            state_aug,
            next_state_aug,
        ) = replay_buffer.sample()

        self.update_seed_weights(seeds, weights)

        with torch.no_grad():
            next_quantiles = self.Q_target.quantiles(next_state)
            next_greedy_actions = next_quantiles.mean(dim=1, keepdim=True).argmax(dim=2, keepdim=True)
            next_greedy_actions = next_greedy_actions.expand(self.batch_size, self.Q.atoms, 1)
            next_quantiles = next_quantiles.gather(dim=2, index=next_greedy_actions).squeeze(dim=2)
            target_quantiles = reward + not_done * self.gamma ** self.n_step * next_quantiles

            next_quantiles_aug = self.Q_target.quantiles(next_state_aug)
            next_greedy_actions_aug = next_quantiles_aug.mean(dim=1, keepdim=True).argmax(dim=2, keepdim=True)
            next_greedy_actions_aug = next_greedy_actions_aug.expand(self.batch_size, self.Q.atoms, 1)
            next_quantiles_aug = next_quantiles_aug.gather(dim=2, index=next_greedy_actions).squeeze(dim=2)
            target_quantiles_aug = reward + not_done * self.gamma ** self.n_step * next_quantiles_aug

            target_quantiles = (target_quantiles + target_quantiles_aug) / 2

        current_quantiles = self.Q.quantiles(state)
        current_quantiles_aug = self.Q.quantiles(state_aug)
        actions = action[..., None].long().expand(self.batch_size, self.Q.atoms, 1)
        current_quantiles = torch.gather(current_quantiles, dim=2, index=actions).squeeze(dim=2)
        current_quantiles_aug = torch.gather(current_quantiles_aug, dim=2, index=actions).squeeze(dim=2)

        loss, td = self.quantile_huber_loss(current_quantiles, target_quantiles, weights)
        loss += self.quantile_huber_loss(current_quantiles_aug, target_quantiles_aug, weights)[0]

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

    def _loss_aug(self, replay_buffer):
        (
            state,
            action,
            next_state,
            reward,
            not_done,
            seeds,
            ind,
            weights,
            state_aug,
            next_state_aug,
        ) = replay_buffer.sample()

        self.update_seed_weights(seeds, weights)

        with torch.no_grad():
            next_action = self.Q(next_state).argmax(1).reshape(-1, 1)
            target_Q = reward + not_done * (self.gamma ** self.n_step) * self.Q_target(next_state).gather(
                1, next_action
            )

            next_action_aug = self.Q(next_state_aug).argmax(1).reshape(-1, 1)
            target_Q_aug = reward + not_done * (self.gamma ** self.n_step) * self.Q_target(
                next_state_aug
            ).gather(1, next_action_aug)

            target_Q = (target_Q + target_Q_aug) / 2

        current_Q = self.Q(state).gather(1, action)
        current_Q_aug = self.Q(state_aug).gather(1, action)

        loss = (weights * F.smooth_l1_loss(current_Q, target_Q, reduction="none")).mean()
        loss += (weights * F.smooth_l1_loss(current_Q_aug, target_Q, reduction="none")).mean()
        priority = (current_Q - target_Q).abs().clamp(min=self.min_priority).cpu().data.numpy().flatten()

        return ind, loss, priority

    def _loss_bootstrap_dqn(self, replay_buffer):
        total_loss = 0
        head_idx = np.random.randint(self.Q.n_ensemble)
        for i in range(self.Q.n_ensemble):
            train_feature = (i == head_idx) or self.args.qrdqn_always_train_feat
            ind, loss, priority = self._loss_bootstrap_dqn_single_head(replay_buffer, i, train_feature)
            total_loss += loss
        return ind, total_loss, priority

    def _loss_bootstrap_dqn_single_head(self, replay_buffer, index, train_feature):
        state, action, next_state, reward, not_done, seeds, ind, weights = replay_buffer.sample()
        with torch.no_grad():
            next_action = self.Q(next_state, ensemble_i=index).argmax(1).reshape(-1, 1)
            target_Q = reward + not_done * (self.gamma ** self.n_step) * \
                self.Q_target(next_state, ensemble_i=index).gather(1, next_action)
        current_Q = self.Q(state, freeze_feature=not train_feature).gather(1, action)
        loss = (weights * F.smooth_l1_loss(current_Q, target_Q, reduction="none")).mean()
        priority = (current_Q - target_Q).abs().clamp(min=self.min_priority).cpu().data.numpy().flatten()
        return ind, loss, priority

    def update_seed_weights(self, seeds, weights):
        if self.track_seed_weights:
            for idx, seed in enumerate(seeds):
                s = seed.cpu().numpy()[0]
                if type(weights) != int and len(weights) > 1:
                    self.seed_weights[s] = self.seed_weights.get(s, 0) + weights[idx].cpu().numpy()[0]
                else:
                    self.seed_weights[s] = self.seed_weights.get(s, 0) + 1
        else:
            pass

    def train_with_online_target(self, replay_buffer, online):
        state, action, next_state, reward, not_done, seeds, ind, weights = replay_buffer.sample()

        self.update_seed_weights(seeds, weights)

        with torch.no_grad():
            target_Q = reward + not_done * (self.gamma ** self.n_step) * online.get_value(next_state, 0, 0)

        current_Q = self.Q(state).gather(1, action)

        loss = (weights * F.smooth_l1_loss(current_Q, target_Q, reduction="none")).mean()

        self.Q_optimizer.zero_grad()
        loss.backward()  # Backpropagate importance-weighted minibatch loss
        clip_grad_norm_(self.Q.parameters(), self.norm_clip)  # Clip gradients by L2 norm
        grad_magnitude = list(self.Q.named_parameters())[-2][1].grad.clone().norm()
        self.Q_optimizer.step()

        # Update target network by polyak or full copy every X iterations.
        self.iterations += 1
        self.maybe_update_target()

        if self.PER:
            priority = ((current_Q - target_Q).abs() + 1e-10).cpu().data.numpy().flatten()
            replay_buffer.update_priority(ind, priority)

        return loss, grad_magnitude

    def huber(self, td_errors, kappa=1.0):
        return torch.where(
            td_errors.abs() <= kappa, 0.5 * td_errors.pow(2), kappa * (td_errors.abs() - 0.5 * kappa)
        )

    def quantile_huber_loss(self, current_quantiles, target_quantiles, weights):
        n_quantiles = current_quantiles.shape[-1]
        cum_prob = (
            torch.arange(n_quantiles, device=current_quantiles.device, dtype=torch.float) + 0.5
        ) / n_quantiles
        cum_prob = cum_prob.view(1, -1, 1)

        td = target_quantiles.unsqueeze(-2) - current_quantiles.unsqueeze(-1)
        huber_loss = self.huber(td)
        loss = torch.abs(cum_prob - (td.detach() < 0).float()) * huber_loss
        loss = (loss.sum(dim=-2) * weights).mean()

        return loss, td

    def polyak_target_update(self):
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def copy_target_update(self):
        if self.iterations % self.target_update_frequency == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())

    def save(self, filename):
        torch.save(self.iterations, filename + "iterations")
        torch.save(self.Q.state_dict(), f"{filename}Q_{self.iterations}")
        torch.save(self.Q_optimizer.state_dict(), filename + "optimizer")

    def load(self, filename):
        self.iterations = torch.load(filename + "iterations")
        self.Q.load_state_dict(torch.load(f"{filename}Q_{self.iterations}"))
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer.load_state_dict(torch.load(filename + "optimizer"))


class ATCAgent(object):
    def __init__(self, args, env):
        self.device = args.device
        self.action_space = env.action_space.n
        self.batch_size = args.batch_size
        self.norm_clip = args.norm_clip
        self.gamma = args.gamma

        self.Q = ATCDQN(args, env).to(self.device)

        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer = getattr(torch.optim, args.optimizer)(
            self.Q.parameters(), **args.optimizer_parameters
        )
        for param in self.Q_target.parameters():
            param.requires_grad = False

        self.encoder = ATCEncoder(env).to(self.device)
        self.contrast = ATCContrast().to(self.device)
        self.target_encoder = copy.deepcopy(self.encoder)
        self.ul_optimizer = getattr(torch.optim, args.optimizer)(
            self.ul_parameters(), **args.optimizer_parameters
        )
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.PER = args.PER
        self.n_step = args.multi_step

        self.min_priority = args.min_priority

        # Target update rule
        self.maybe_update_target = (
            self.polyak_target_update if args.polyak_target_update else self.copy_target_update
        )
        self.target_update_frequency = int(args.target_update // args.num_processes)
        self.tau = args.tau

        # Evaluation hyper-parameters
        self.state_shape = (-1,) + env.observation_space.shape
        self.eval_eps = args.eval_eps

        # For seed bar chart
        self.track_seed_weights = False
        if self.track_seed_weights:
            self.seed_weights = {
                i: 0 for i in range(args.start_level, args.start_level + args.num_train_seeds)
            }

        # Number of training iterations
        self.iterations = 0

    def select_action(self, state, eps=0.1, eval=False):
        with torch.no_grad():
            features = self.encoder.encode(state)
            q = self.Q(features)
            action = q.argmax(1).reshape(-1, 1)
            max_q = q.max(1)[0]
            mean_q = q.mean(1)
            v = (1 - eps) * max_q + eps * mean_q
            return action, v

    def get_value(self, state, eps=0.1):
        with torch.no_grad():
            features = self.encoder.encode(state)
            q = self.Q(features)
            max_q = q.max(1)[0]
            mean_q = q.mean(1)
            v = (1 - eps) * max_q + eps * mean_q
            return v

    def advantage(self, state, eps):
        features = self.encoder.encode(state)
        q = self.Q(features)
        max_q = q.max(1)[0]
        mean_q = q.mean(1)
        v = (1 - eps) * max_q + eps * mean_q
        return q - v.repeat(q.shape[1]).reshape(-1, q.shape[1])

    def train(self, replay_buffer):
        ind, rl_loss, ul_loss, accuracy = self.loss(replay_buffer)

        self.Q_optimizer.zero_grad()
        rl_loss.backward()  # Backpropagate importance-weighted minibatch loss
        clip_grad_norm_(self.Q.parameters(), self.norm_clip)  # Clip gradients by L2 norm
        self.Q_optimizer.step()

        self.ul_optimizer.zero_grad()
        ul_loss.backward()
        clip_grad_norm_(self.ul_parameters(), self.norm_clip)
        self.ul_optimizer.step()

        # Update target network by polyak or full copy every X iterations.
        self.iterations += 1
        self.maybe_update_target()
        self.atc_target_update()

        return rl_loss, accuracy

    def loss(self, replay_buffer):
        (
            state,
            action,
            next_state,
            reward,
            not_done,
            seeds,
            ind,
            weights,
            state_aug,
            next_state_aug,
        ) = replay_buffer.sample_atc()

        with torch.no_grad():
            next_action = self.Q(self.encoder.encode(next_state)).argmax(1).reshape(-1, 1)
            target_Q = reward + not_done * (self.gamma ** self.n_step) * self.Q_target(
                self.target_encoder.encode(next_state)
            ).gather(1, next_action)

        current_Q = self.Q(self.encoder.encode(state)).gather(1, action)

        rl_loss = F.smooth_l1_loss(current_Q, target_Q, reduction="none").mean()

        anchor = self.encoder(state_aug)
        with torch.no_grad():
            positive = self.target_encoder(next_state_aug)

        logits = self.contrast(anchor, positive)
        labels = torch.arange(state.shape[0], dtype=torch.long, device=self.device)

        ul_loss = F.cross_entropy(logits, labels)

        correct = torch.argmax(logits.detach(), dim=1) == labels
        accuracy = torch.mean(correct.float())

        return ind, rl_loss, ul_loss, accuracy

    def ul_parameters(self):
        yield from self.encoder.parameters()
        yield from self.contrast.parameters()

    def polyak_target_update(self):
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def copy_target_update(self):
        if self.iterations % self.target_update_frequency == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())

    def atc_target_update(self):
        for param, target_param in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target_param.data.copy_(0.01 * param.data + 0.99 * target_param.data)


class AtariAgent(object):
    def __init__(self, args, env):
        self.device = args.device
        self.action_space = args.num_actions
        self.batch_size = args.batch_size
        self.norm_clip = args.norm_clip
        self.gamma = args.gamma

        self.Q = Conv_Q(4, self.action_space).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer = getattr(torch.optim, args.optimizer)(
            self.Q.parameters(), **args.optimizer_parameters
        )

        self.PER = args.PER
        self.n_step = args.multi_step

        self.min_priority = args.min_priority

        # Target update rule
        self.maybe_update_target = (
            self.polyak_target_update if args.polyak_target_update else self.copy_target_update
        )
        self.target_update_frequency = args.target_update
        self.tau = args.tau

        # Evaluation hyper-parameters
        self.state_shape = (-1,) + env.observation_space.shape
        self.eval_eps = args.eval_eps

        # Number of training iterations
        self.iterations = 0

    def select_action(self, state, eval=False):
        with torch.no_grad():
            q = self.Q(state)
            action = q.argmax(1).reshape(-1, 1)
            return action, None

    def train(self, replay_buffer):
        state, action, next_state, reward, not_done, seeds, ind, weights = replay_buffer.sample()

        with torch.no_grad():
            next_action = self.Q(next_state).argmax(1).reshape(-1, 1)
            target_Q = reward + not_done * (self.gamma ** self.n_step) * self.Q_target(next_state).gather(
                1, next_action
            )

        current_Q = self.Q(state).gather(1, action)

        loss = (weights * F.smooth_l1_loss(current_Q, target_Q, reduction="none")).mean()

        self.Q_optimizer.zero_grad()
        loss.backward()  # Backpropagate importance-weighted minibatch loss
        grad_magnitude = list(self.Q.named_parameters())[-2][1].grad.clone().norm()
        # clip_grad_norm_(self.Q.parameters(), self.norm_clip)  # Clip gradients by L2 norm
        self.Q_optimizer.step()

        # Update target network by polyak or full copy every X iterations.
        self.iterations += 1
        self.maybe_update_target()

        if self.PER:
            priority = ((current_Q - target_Q).abs() + 1e-10).pow(0.6).cpu().data.numpy().flatten()
            replay_buffer.update_priority(ind, priority)

        return loss, grad_magnitude

    def huber(self, x):
        return torch.where(x < self.min_priority, 0.5 * x.pow(2), self.min_priority * x).mean()

    def polyak_target_update(self):
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def copy_target_update(self):
        if self.iterations % self.target_update_frequency == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())

    def save(self, filename):
        torch.save(self.iterations, filename + "iterations")
        torch.save(self.Q.state_dict(), f"{filename}Q_{self.iterations}")
        torch.save(self.Q_optimizer.state_dict(), filename + "optimizer")

    def load(self, filename):
        self.iterations = torch.load(filename + "iterations")
        self.Q.load_state_dict(torch.load(f"{filename}Q_{self.iterations}"))
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer.load_state_dict(torch.load(filename + "optimizer"))
