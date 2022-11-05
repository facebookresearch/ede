# Copyright (c) 2017 Ilya Kostrikov
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# This file is a modified version of:
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/ppo.py
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import torch
import torch.nn as nn
import torch.optim as optim

from level_replay.model import model_for_env_name


class PPO:
    """
    Vanilla PPO
    """

    def __init__(
        self,
        actor_critic,
        clip_param,
        ppo_epoch,
        num_mini_batch,
        value_loss_coef,
        entropy_coef,
        lr=None,
        eps=None,
        max_grad_norm=None,
        env_name=None,
    ):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

        self.env_name = env_name

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for _ in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                (
                    obs_batch,
                    recurrent_hidden_states_batch,
                    actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                ) = sample

                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch
                )

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (values - return_batch).pow(2)
                value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

                self.optimizer.zero_grad()
                loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch


class MultiWorkerPPO:
    def __init__(self, args, envs):
        self.num_workers = args.num_workers
        self.actor_critic = model_for_env_name(args, envs).to(args.device)

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch

        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef

        self.max_grad_norm = args.max_grad_norm

        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=args.lr)

        self.env_name = args.env_name
        self.device = args.device

    def update(self, rollouts_):
        advantages = list()
        for rollouts in rollouts_:
            adv = rollouts.returns[:-1] - rollouts.value_preds[:-1]
            advantages.append((adv - adv.mean()) / (adv.std() + 1e-5))

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for _ in range(self.ppo_epoch):
            data_generators = [
                rollouts_[i].feed_forward_generator(advantages[i], self.num_mini_batch)
                for i in range(self.num_workers)
            ]
            for data_generator in data_generators:
                loss = torch.tensor(0, device=self.device)
                value_loss_ = 0
                action_loss_ = 0
                dist_entropy_ = 0
                for sample in data_generator:
                    (
                        obs_batch,
                        actions_batch,
                        value_preds_batch,
                        return_batch,
                        masks_batch,
                        old_action_log_probs_batch,
                        adv_targ,
                    ) = sample

                    values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(
                        obs_batch, actions_batch
                    )

                    ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                    surr1 = ratio * adv_targ
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                    action_loss = -torch.min(surr1, surr2).mean()

                    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
                        -self.clip_param, self.clip_param
                    )
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

                    self.optimizer.zero_grad()
                    loss += value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef

                    value_loss_ += value_loss.item()
                    action_loss_ += action_loss.item()
                    dist_entropy_ += dist_entropy.item()

                self.optimizer.zero_grad()
                (loss / self.num_workers).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss_ / self.num_workers
                action_loss_epoch += action_loss_ / self.num_workers
                dist_entropy_epoch += dist_entropy_ / self.num_workers

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch


class SimplePPO:
    """
    PPO for Simple Arch for Interpretability
    """

    def __init__(
        self,
        actor_critic,
        clip_param,
        ppo_epoch,
        num_mini_batch,
        value_loss_coef,
        entropy_coef,
        lr=None,
        eps=None,
        max_grad_norm=None,
        env_name=None,
    ):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr, eps=eps)

        self.env_name = env_name

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for _ in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                (
                    obs_batch,
                    actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                ) = sample

                values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(
                    obs_batch, actions_batch
                )

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (values - return_batch).pow(2)
                value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

                self.optimizer.zero_grad()
                loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
