
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from dataclasses import dataclass

from level_replay.algo.buffer import Buffer, AugBuffer


@dataclass
class LevelBufferConfig:
    device: str
    seeds: list
    batch_size = 32
    memory_capacity = 5000
    ptr = 0
    size = 0
    PER = False


class PLRBufferV2:
    def __init__(self, args, env):
        self.device = args.device
        seeds = args.seeds
        self.seeds = np.array(seeds, dtype=np.int64)
        self.obs_space = env.observation_space.shape
        self.action_space = env.action_space.n
        self.num_actors = args.num_processes

        self.num_seeds_in_update = 16
        self.batch_size_per_seed = 32

        buffer_config = LevelBufferConfig(self.device, self.seeds)
        buffer_config.batch_size = self.batch_size_per_seed
        self.drq = args.drq

        if args.drq:
            self.buffers = {seed: AugBuffer(buffer_config, env) for seed in self.seeds}
        else:
            self.buffers = {seed: Buffer(buffer_config, env) for seed in self.seeds}

        self.valid_buffers = np.array([0.0] * len(self.seeds), dtype=np.float)

    def add(self, state, action, next_state, reward, done, seed):
        self.buffers[seed.item()].add(state, action, next_state, reward, done, seed)
        if self.buffers[seed.item()].size > self.batch_size_per_seed:
            self.valid_buffers[seed.item()] = 1.0

    def get_level_sampler(self, level_sampler):
        self.level_sampler = level_sampler

    def sample(self):
        if self.drq:
            return self.sample_aug()
        sub_batch = int(self.batch_size_per_seed)
        batch_size = self.num_seeds_in_update * self.batch_size_per_seed
        state = torch.empty((batch_size,) + self.obs_space, dtype=torch.float, device=self.device)
        action = torch.empty((batch_size, 1), dtype=torch.long, device=self.device)
        next_state = torch.empty((batch_size,) + self.obs_space, dtype=torch.float, device=self.device)
        reward = torch.empty((batch_size, 1), dtype=torch.float, device=self.device)
        not_done = torch.empty((batch_size, 1), dtype=torch.float, device=self.device)
        seeds = torch.empty((batch_size, 1), dtype=torch.long, device=self.device)
        levels = self._sample_levels()
        for i, seed in enumerate(levels):
            state_, action_, next_state_, reward_, not_done_, seeds_, _, _ = self.buffers[seed].sample()
            state[i * sub_batch : (i + 1) * sub_batch] = state_
            action[i * sub_batch : (i + 1) * sub_batch] = action_
            next_state[i * sub_batch : (i + 1) * sub_batch] = next_state_
            reward[i * sub_batch : (i + 1) * sub_batch] = reward_
            not_done[i * sub_batch : (i + 1) * sub_batch] = not_done_
            seeds[i * sub_batch : (i + 1) * sub_batch] = seeds_

        return state, action, next_state, reward, not_done, seeds, 0, 1

    def sample_aug(self):
        sub_batch = int(self.batch_size_per_seed)
        batch_size = self.num_seeds_in_update * self.batch_size_per_seed
        state = torch.empty((batch_size,) + self.obs_space, dtype=torch.float, device=self.device)
        action = torch.empty((batch_size, 1), dtype=torch.long, device=self.device)
        next_state = torch.empty((batch_size,) + self.obs_space, dtype=torch.float, device=self.device)
        reward = torch.empty((batch_size, 1), dtype=torch.float, device=self.device)
        not_done = torch.empty((batch_size, 1), dtype=torch.float, device=self.device)
        seeds = torch.empty((batch_size, 1), dtype=torch.long, device=self.device)
        state_aug = torch.empty((batch_size,) + self.obs_space, dtype=torch.float, device=self.device)
        next_state_aug = torch.empty((batch_size,) + self.obs_space, dtype=torch.float, device=self.device)
        levels = self._sample_levels()

        for i, seed in enumerate(levels):
            (
                state_,
                action_,
                next_state_,
                reward_,
                not_done_,
                seeds_,
                _,
                _,
                state_aug_,
                next_state_aug_,
            ) = self.buffers[seed].sample()
            state[i * sub_batch : (i + 1) * sub_batch] = state_
            action[i * sub_batch : (i + 1) * sub_batch] = action_
            next_state[i * sub_batch : (i + 1) * sub_batch] = next_state_
            reward[i * sub_batch : (i + 1) * sub_batch] = reward_
            not_done[i * sub_batch : (i + 1) * sub_batch] = not_done_
            seeds[i * sub_batch : (i + 1) * sub_batch] = seeds_
            state_aug[i * sub_batch : (i + 1) * sub_batch] = state_aug_
            next_state_aug[i * sub_batch : (i + 1) * sub_batch] = next_state_aug_

        return state, action, next_state, reward, not_done, seeds, 0, 1, state_aug, next_state_aug

    def _sample_levels(self):
        # prev_transform = self.level_sampler.score_transform
        # self.level_sampler.score_transform = "power"
        if self.level_sampler.has_sampled_weights:
            sample_weights = self.level_sampler.probs
        else:
            sample_weights = self.level_sampler.sample_weights()
        weights = sample_weights * self.valid_buffers
        if np.isclose(np.sum(weights), 0):
            weights = np.ones_like(weights, dtype=float) * self.valid_buffers
        weights = weights / np.sum(weights)
        levels = np.random.choice(self.seeds, self.num_seeds_in_update, p=weights)
        # self.level_sampler.score_transform = prev_transform

        return levels
