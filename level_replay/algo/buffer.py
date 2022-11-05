# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import math
import random
import kornia
import torch.nn as nn
from collections import deque
from typing import List

from level_replay.algo.binary_heap import BinaryHeap
from level_replay import data_augs


class AbstractBuffer:
    def __init__(self, args, env):
        self.batch_size = args.batch_size
        self.max_size = int(args.memory_capacity)
        self.device = args.device
        self.all_seeds = args.seeds

        self.ptr = 0
        self.size = 0

        self.state = np.zeros((self.max_size, *env.observation_space.shape), dtype=np.uint8)
        self.action = np.zeros((self.max_size, 1), dtype=np.uint8)
        self.next_state = np.array(self.state)
        self.reward = np.zeros((self.max_size, 1))
        self.not_done = np.zeros((self.max_size, 1), dtype=np.uint8)
        self.seeds = np.zeros((self.max_size, 1), dtype=np.uint8)

    def add(self, state, action, next_state, reward, done, seeds):
        pass

    def sample(self):
        pass

    def update_priority(self, ind, priority):
        pass


class Buffer(AbstractBuffer):
    def __init__(self, args, env):
        super(Buffer, self).__init__(args, env)
        self.prioritized = args.PER

        if self.prioritized:
            num_updates = (args.T_max // args.num_processes - args.start_timesteps) // args.train_freq
            self.tree = SumTree(self.max_size)
            self.max_priority = 1.0
            self.beta = args.beta
            self.beta_stepper = (1 - self.beta) / float(num_updates)
            self.alpha = args.alpha

    def add(self, state, action, next_state, reward, done, seeds):
        n_transitions = state.shape[0] if len(state.shape) == 4 else 1
        end = (self.ptr + n_transitions) % self.max_size
        if "cuda" in self.device.type:
            state = (state * 255).cpu().numpy().astype(np.uint8)
            action = action.cpu().numpy().astype(np.uint8)
            next_state = (next_state * 255).cpu().numpy().astype(np.uint8)
            # We leave reward as numpy throughout
            # reward = reward.cpu().numpy()
            seeds = seeds.cpu().numpy().astype(np.uint8)
        else:
            state = (state * 255).numpy().astype(np.uint8)
            action = action.numpy().astype(np.uint8)
            next_state = (next_state * 255).numpy().astype(np.uint8)
            seeds = seeds.numpy().astype(np.uint8)

        not_done = (1 - done).reshape(-1, 1)

        if self.ptr + n_transitions > self.max_size:
            self.state[self.ptr :] = state[: n_transitions - end]
            self.state[:end] = state[n_transitions - end :]

            self.action[self.ptr :] = action[: n_transitions - end]
            self.action[:end] = action[n_transitions - end :]

            self.next_state[self.ptr :] = next_state[: n_transitions - end]
            self.next_state[:end] = next_state[n_transitions - end :]

            self.reward[self.ptr :] = reward[: n_transitions - end]
            self.reward[:end] = reward[n_transitions - end :]

            self.not_done[self.ptr :] = not_done[: n_transitions - end]
            self.not_done[:end] = not_done[n_transitions - end :]
            self.seeds[self.ptr :] = seeds[: n_transitions - end]
            self.seeds[:end] = seeds[n_transitions - end :]
        else:
            self.state[self.ptr : self.ptr + n_transitions] = state
            self.action[self.ptr : self.ptr + n_transitions] = action
            self.next_state[self.ptr : self.ptr + n_transitions] = next_state
            self.reward[self.ptr : self.ptr + n_transitions] = reward
            self.not_done[self.ptr : self.ptr + n_transitions] = not_done
            self.seeds[self.ptr : self.ptr + n_transitions] = seeds

        if self.prioritized:
            for index in [i % self.max_size for i in range(self.ptr, self.ptr + n_transitions)]:
                self.tree.set(index, self.max_priority)

        self.ptr = end
        self.size = min(self.size + n_transitions, self.max_size)

    def sample(self, **kwargs):
        if self.prioritized:
            ind = self.tree.sample(self.batch_size)
            weights = np.array(self.tree.nodes[-1][ind]) ** -self.beta
            weights = torch.FloatTensor(weights / weights.max()).to(self.device).reshape(-1, 1)
            self.beta = min(self.beta + self.beta_stepper, 1)
        else:
            ind = np.random.randint(0, self.size, size=self.batch_size)
            weights = torch.FloatTensor([1]).to(self.device)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device) / 255.0,
            torch.LongTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device) / 255.0,
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.LongTensor(self.seeds[ind]).to(self.device),
            ind,
            weights,
        )

    def update_priority(self, ind, priority):
        priority = np.power(priority, self.alpha)
        self.max_priority = max(priority.max(), self.max_priority)
        self.tree.batch_set(ind, priority)

    def weights_per_seed(self):
        if self.prioritized:
            seed2weight = {}
            max_weight = np.max(self.tree.nodes[-1][: self.size] ** -self.beta)
            for seed in self.all_seeds:
                idx = np.where(self.seeds[: self.size] == seed)[0]
                if len(idx) > 0:
                    avg_level_weight = np.mean(self.tree.nodes[-1][idx] ** -self.beta)
                    seed2weight[seed] = avg_level_weight / max_weight
                else:
                    seed2weight[seed] = 0.0
            m = max(seed2weight.values())
            seed2weight = {seed: seed2weight[seed] / m for seed in self.all_seeds}
            return seed2weight
        else:
            return {seed: 1.0 for seed in self.all_seeds}


class AugBuffer(Buffer):
    def __init__(self, args, env):
        super(AugBuffer, self).__init__(args, env)
        self.aug_trans = nn.Sequential(
            nn.ReplicationPad2d(4),
            kornia.augmentation.RandomCrop(
                (env.observation_space.shape[-1], env.observation_space.shape[-1])
            ),
        )

        def f():
            pass

        self.aug_trans.change_randomization_params_all = f

    def sample(self, aug=True):
        if self.prioritized:
            ind = self.tree.sample(self.batch_size)
            weights = np.array(self.tree.nodes[-1][ind]) ** -self.beta
            weights = torch.FloatTensor(weights / weights.max()).to(self.device).reshape(-1, 1)
            self.beta = min(self.beta + self.beta_stepper, 1)
        else:
            ind = np.random.randint(0, self.size, size=self.batch_size)
            weights = torch.FloatTensor([1]).to(self.device)

        state = self.state[ind]
        next_state = self.next_state[ind]

        if not aug:
            return (
                torch.FloatTensor(self.state[ind]).to(self.device) / 255.0,
                torch.LongTensor(self.action[ind]).to(self.device),
                torch.FloatTensor(self.next_state[ind]).to(self.device) / 255.0,
                torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.not_done[ind]).to(self.device),
                torch.LongTensor(self.seeds[ind]).to(self.device),
                ind,
                weights,
            )

        state_aug = state.copy()
        next_state_aug = next_state.copy()

        state = torch.as_tensor(state, device=self.device).float() / 255.0
        next_state = torch.as_tensor(next_state, device=self.device).float() / 255.0
        state_aug = torch.as_tensor(state_aug, device=self.device).float() / 255.0
        next_state_aug = torch.as_tensor(next_state_aug, device=self.device).float() / 255.0

        state = self.aug_trans(state)
        next_state = self.aug_trans(next_state)

        state_aug = self.aug_trans(state_aug)
        next_state_aug = self.aug_trans(next_state_aug)

        return (
            state,
            torch.LongTensor(self.action[ind]).to(self.device),
            next_state,
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.LongTensor(self.seeds[ind]).to(self.device),
            ind,
            weights,
            state_aug,
            next_state_aug,
        )

    def sample_atc(self):
        if self.prioritized:
            ind = self.tree.sample(self.batch_size)
            weights = np.array(self.tree.nodes[-1][ind]) ** -self.beta
            weights = torch.FloatTensor(weights / weights.max()).to(self.device).reshape(-1, 1)
            self.beta = min(self.beta + self.beta_stepper, 1)
        else:
            ind = np.random.randint(0, self.size, size=self.batch_size)
            weights = torch.FloatTensor([1]).to(self.device)

        state = self.state[ind]
        next_state = self.next_state[ind]
        state_aug = state.copy()
        next_state_aug = next_state.copy()

        state = torch.as_tensor(state, device=self.device).float() / 255.0
        next_state = torch.as_tensor(next_state, device=self.device).float() / 255.0
        state_aug = torch.as_tensor(state_aug, device=self.device).float() / 255.0
        next_state_aug = torch.as_tensor(next_state_aug, device=self.device).float() / 255.0

        state_aug = self.aug_trans(state_aug)
        next_state_aug = self.aug_trans(next_state_aug)

        return (
            state,
            torch.LongTensor(self.action[ind]).to(self.device),
            next_state,
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.LongTensor(self.seeds[ind]).to(self.device),
            ind,
            weights,
            state_aug,
            next_state_aug,
        )


class AutoAugBuffer(Buffer):
    def __init__(self, args, env):
        super(AutoAugBuffer, self).__init__(args, env)
        aug_to_func = {
            "crop": data_augs.Crop,
            "random-conv": data_augs.RandomConv,
            "grayscale": data_augs.Grayscale,
            "flip": data_augs.Flip,
            "rotate": data_augs.Rotate,
            "cutout": data_augs.Cutout,
            "cutout-color": data_augs.CutoutColor,
            "color-jitter": data_augs.ColorJitter,
        }
        self.ucb_coef = 0.1
        self.aug_list = [aug_to_func[t](batch_size=self.batch_size) for t in list(aug_to_func.keys())]
        self.aug_idx = 0
        self.aug_trans = self.aug_list[self.aug_idx]
        self.aug_counts = np.zeros((len(self.aug_list)))
        self.aug_q = np.zeros((len(self.aug_list)))
        self.aug_returns: List[deque] = [deque(maxlen=10)]
        self.total_num = 1

    def sample(self):
        if self.prioritized:
            ind = self.tree.sample(self.batch_size)
            weights = np.array(self.tree.nodes[-1][ind]) ** -self.beta
            weights = torch.FloatTensor(weights / weights.max()).to(self.device).reshape(-1, 1)
            self.beta = min(self.beta + self.beta_stepper, 1)
        else:
            ind = np.random.randint(0, self.size, size=self.batch_size)
            weights = torch.FloatTensor([1]).to(self.device)

        state = self.state[ind]
        next_state = self.next_state[ind]
        state_aug = state.copy()
        next_state_aug = next_state.copy()

        state = torch.as_tensor(state, device=self.device).float() / 255.0
        next_state = torch.as_tensor(next_state, device=self.device).float() / 255.0
        state_aug = torch.as_tensor(state_aug, device=self.device).float() / 255.0
        next_state_aug = torch.as_tensor(next_state_aug, device=self.device).float() / 255.0

        state_aug = self.aug_trans.do_augmentation(state_aug)
        next_state_aug = self.aug_trans.do_augmentation(next_state_aug)

        return (
            state,
            torch.LongTensor(self.action[ind]).to(self.device),
            next_state,
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.LongTensor(self.seeds[ind]).to(self.device),
            ind,
            weights,
            state_aug,
            next_state_aug,
        )

    def select_aug(self):
        ucb = self.aug_q + self.ucb_coef * np.sqrt(self.total_num / self.aug_counts)
        argmax = ucb.argmax()
        self.aug_counts[argmax] += 1
        self.aug_idx = argmax
        self.aug_trans = self.aug_list[self.aug_idx]

    def add_reward(self, reward):
        self.aug_returns[self.aug_idx].append(reward)
        self.aug_q[self.aug_idx] = np.mean(self.aug_returns[self.aug_idx])

    def update_ucb_values(self, rollouts):
        self.total_num += 1
        self.aug_counts[self.aug_idx] += 1
        self.aug_returns[self.aug_idx].append(rollouts.returns.mean().item())
        self.aug_q[self.aug_idx] = np.mean(self.aug_returns[self.aug_idx])


class RankBuffer(AbstractBuffer):
    def __init__(self, args, env):
        super(RankBuffer, self).__init__(args, env)

        self.prioritized = args.PER
        num_updates = (args.T_max // args.num_processes - args.start_timesteps) // args.train_freq

        self.beta = args.beta
        self.beta_stepper = (1 - self.beta) / float(num_updates)
        self.priority_queue = BinaryHeap(self.max_size)
        self.max_priority = 1.0
        self.alpha = args.alpha

        self.build_distribution()

    def build_distribution(self):
        pdf = list(map(lambda x: math.pow(x, -self.max_priority), range(1, self.max_size + 1)))
        pdf_sum = math.fsum(pdf)
        self.power_law_distribution = list(map(lambda x: x / pdf_sum, pdf))

    def add(self, state, action, next_state, reward, done, seeds):
        n_transitions = state.shape[0] if len(state.shape) == 4 else 1
        end = (self.ptr + n_transitions) % self.max_size
        if "cuda" in self.device.type:
            state = (state * 255).cpu().numpy().astype(np.uint8)
            action = action.cpu().numpy().astype(np.uint8)
            next_state = (next_state * 255).cpu().numpy().astype(np.uint8)
            # reward = reward.cpu().numpy()
            seeds = seeds.cpu().numpy().astype(np.uint8)
        else:
            state = (state * 255).numpy().astype(np.uint8)
            action = action.numpy().astype(np.uint8)
            next_state = (next_state * 255).numpy().astype(np.uint8)
            seeds = seeds.numpy().astype(np.uint8)

        not_done = (1 - done).reshape(-1, 1)

        if self.ptr + n_transitions > self.max_size:
            self.state[self.ptr :] = state[: n_transitions - end]
            self.state[:end] = state[n_transitions - end :]

            self.action[self.ptr :] = action[: n_transitions - end]
            self.action[:end] = action[n_transitions - end :]

            self.next_state[self.ptr :] = next_state[: n_transitions - end]
            self.next_state[:end] = next_state[n_transitions - end :]

            self.reward[self.ptr :] = reward[: n_transitions - end]
            self.reward[:end] = reward[n_transitions - end :]

            self.not_done[self.ptr :] = not_done[: n_transitions - end]
            self.not_done[:end] = not_done[n_transitions - end :]
            self.seeds[self.ptr :] = seeds[: n_transitions - end]
            self.seeds[:end] = seeds[n_transitions - end :]
        else:
            self.state[self.ptr : self.ptr + n_transitions] = state
            self.action[self.ptr : self.ptr + n_transitions] = action
            self.next_state[self.ptr : self.ptr + n_transitions] = next_state
            self.reward[self.ptr : self.ptr + n_transitions] = reward
            self.not_done[self.ptr : self.ptr + n_transitions] = not_done
            self.seeds[self.ptr : self.ptr + n_transitions] = seeds

        priority = self.priority_queue.get_max_priority()
        for index in [i % self.max_size for i in range(self.ptr, self.ptr + n_transitions)]:
            self.priority_queue.update(priority, index)

        self.ptr = end
        self.size = min(self.size + n_transitions, self.max_size)

    def sample(self):
        ind, weights = self.select(self.batch_size)
        weights = torch.FloatTensor(weights).to(self.device).reshape(-1, 1)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device) / 255.0,
            torch.LongTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device) / 255.0,
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.LongTensor(self.seeds[ind]).to(self.device),
            ind,
            weights,
        )

    def rebalance(self):
        self.priority_queue.balance_tree()

    def update_priority(self, indices, delta):
        delta = np.power(delta, self.alpha)
        for i in range(len(indices)):
            self.priority_queue.update(math.fabs(delta[i]), indices[i])

    def select(self, batch_size):
        distribution = self.power_law_distribution
        rank_list = []
        for _ in range(batch_size):
            index = random.randint(1, self.priority_queue.size)
            rank_list.append(index)

        alpha_pow = [distribution[v - 1] for v in rank_list]
        w = np.power(np.array(alpha_pow) * self.size, -self.beta)
        w_max = max(w)
        w = np.divide(w, w_max)
        rank_e_id = 0
        self.beta = min(self.beta + self.beta_stepper, 1)
        rank_e_id = self.priority_queue.priority_to_experience(rank_list)
        return rank_e_id, w


class PLRBuffer(AbstractBuffer):
    def __init__(self, args, env):
        super(PLRBuffer, self).__init__(args, env)
        self._seeds = args.seeds
        self.obs_space = env.observation_space.shape
        self.action_space = env.action_space.n
        self.num_actors = args.num_processes
        self.strategy = "value_l1"
        self.replay_schedule = "proportionate"
        self.score_transform = "rank"
        self.temperature = 0.1
        self.eps = 0.05
        self.rho = 1.0
        self.nu = 0.5
        self.alpha = 1.0
        self.staleness_coef = 0.1
        self.staleness_transform = "power"
        self.staleness_temperature = 1.0

        self._init_seed_index(self._seeds)

        self.unseen_seed_weights = np.array([1.0] * len(self._seeds))
        self.seed_scores = np.array([0.0] * len(self._seeds), dtype=np.float)
        self.partial_seed_scores = np.zeros((self.num_actors, len(self._seeds)), dtype=np.float)
        self.partial_seed_steps = np.zeros((self.num_actors, len(self._seeds)), dtype=np.int64)
        self.seed_staleness = np.array([0.0] * len(self._seeds), dtype=np.float)
        self.staleness_counter = np.array([0] * len(self._seeds), dtype=np.uint8)

        self.next_seed_index = 0

    def add(self, state, action, next_state, reward, done, seeds):
        n_transitions = state.shape[0] if len(state.shape) == 4 else 1
        end = (self.ptr + n_transitions) % self.max_size
        if "cuda" in self.device.type:
            state = (state * 255).cpu().numpy().astype(np.uint8)
            action = action.cpu().numpy().astype(np.uint8)
            next_state = (next_state * 255).cpu().numpy().astype(np.uint8)
            seeds = seeds.cpu().numpy().astype(np.uint8)
        else:
            state = (state * 255).numpy().astype(np.uint8)
            action = action.numpy().astype(np.uint8)
            next_state = (next_state * 255).numpy().astype(np.uint8)
            seeds = seeds.numpy().astype(np.uint8)

        not_done = (1 - done).reshape(-1, 1)

        if self.ptr + n_transitions > self.max_size:
            self.state[self.ptr :] = state[: n_transitions - end]
            self.state[:end] = state[n_transitions - end :]

            self.action[self.ptr :] = action[: n_transitions - end]
            self.action[:end] = action[n_transitions - end :]

            self.next_state[self.ptr :] = next_state[: n_transitions - end]
            self.next_state[:end] = next_state[n_transitions - end :]

            self.reward[self.ptr :] = reward[: n_transitions - end]
            self.reward[:end] = reward[n_transitions - end :]

            self.not_done[self.ptr :] = not_done[: n_transitions - end]
            self.not_done[:end] = not_done[n_transitions - end :]
            self.seeds[self.ptr :] = seeds[: n_transitions - end]
            self.seeds[:end] = seeds[n_transitions - end :]
        else:
            self.state[self.ptr : self.ptr + n_transitions] = state
            self.action[self.ptr : self.ptr + n_transitions] = action
            self.next_state[self.ptr : self.ptr + n_transitions] = next_state
            self.reward[self.ptr : self.ptr + n_transitions] = reward
            self.not_done[self.ptr : self.ptr + n_transitions] = not_done
            self.seeds[self.ptr : self.ptr + n_transitions] = seeds

        self.ptr = end
        self.size = min(self.size + n_transitions, self.max_size)

    def sample(self):
        ind = np.random.randint(0, self.size, size=self.batch_size)
        weights = self._get_weights2(ind)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device) / 255.0,
            torch.LongTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device) / 255.0,
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.LongTensor(self.seeds[ind]).to(self.device),
            ind,
            weights / weights.max(),
        )

    def _get_weights(self, ind):
        seeds = self.seeds[ind]
        weights = []
        for seed in seeds:
            weight = 1.0 if self.unseen_seed_weights[seed][0] != 0.0 else self.seed_scores[seed][0]
            weights.append(weight)

        weights = torch.FloatTensor(weights).to(self.device).reshape(-1, 1)

        return weights

    def _get_weights2(self, ind):
        seeds = self.seeds[ind]
        all_weights = self.sample_weights()
        batch_weights = np.zeros(len(seeds))
        for idx, seed in enumerate(seeds):
            self.staleness_counter[seed] += 1
            if self.staleness_counter[seed] == 256:
                self._update_staleness(seed)
                self.staleness_counter[seed] = 0
            weight = all_weights[seed]
            batch_weights[idx] = weight ** -0.4 if weight > 0 else 0

        batch_weights = torch.FloatTensor(batch_weights).to(self.device).reshape(-1, 1)

        return batch_weights

    def sample_weights(self):
        weights = self._score_transform(self.score_transform, self.temperature, self.seed_scores)
        weights = weights * (1 - self.unseen_seed_weights)  # zero out unseen levels

        z = np.sum(weights)
        if z > 0:
            weights /= z

        staleness_weights = 0
        if self.staleness_coef > 0:
            staleness_weights = self._score_transform(
                self.staleness_transform, self.staleness_temperature, self.seed_staleness
            )
            staleness_weights = staleness_weights * (1 - self.unseen_seed_weights)
            z = np.sum(staleness_weights)
            if z > 0:
                staleness_weights /= z

            weights = (1 - self.staleness_coef) * weights + self.staleness_coef * staleness_weights

        return weights

    def seed_range(self):
        return (int(min(self._seeds)), int(max(self._seeds)))

    def _init_seed_index(self, seeds):
        self._seeds = np.array(seeds, dtype=np.int64)
        self.seed2index = {seed: i for i, seed in enumerate(seeds)}

    def update_with_rollouts(self, rollouts):
        score_function = self._average_value_l1

        self._update_with_rollouts(rollouts, score_function)

    def update_seed_score(self, actor_index, seed_idx, score, num_steps):
        score = self._partial_update_seed_score(actor_index, seed_idx, score, num_steps, done=True)

        self.unseen_seed_weights[seed_idx] = 0.0  # No longer unseen

        old_score = self.seed_scores[seed_idx]
        self.seed_scores[seed_idx] = (1 - self.alpha) * old_score + self.alpha * score

    def _partial_update_seed_score(self, actor_index, seed_idx, score, num_steps, done=False):
        partial_score = self.partial_seed_scores[actor_index][seed_idx]
        partial_num_steps = self.partial_seed_steps[actor_index][seed_idx]

        running_num_steps = partial_num_steps + num_steps
        merged_score = partial_score + (score - partial_score) * num_steps / float(running_num_steps)

        if done:
            self.partial_seed_scores[actor_index][seed_idx] = 0.0  # zero partial score, partial num_steps
            self.partial_seed_steps[actor_index][seed_idx] = 0
        else:
            self.partial_seed_scores[actor_index][seed_idx] = merged_score
            self.partial_seed_steps[actor_index][seed_idx] = running_num_steps

        return merged_score

    def _average_value_l1(self, **kwargs):
        returns = kwargs["returns"]
        value_preds = kwargs["value_preds"]

        advantages = returns - value_preds

        return advantages.abs().mean().item()

    def _update_with_rollouts(self, rollouts, score_function):
        level_seeds = rollouts.level_seeds
        done = ~(rollouts.masks > 0)
        total_steps, num_actors = rollouts.rewards.shape[:2]

        for actor_index in range(num_actors):
            done_steps = torch.nonzero(done[:, actor_index], as_tuple=False)[:total_steps, 0]
            start_t = 0

            for t in done_steps:
                if not start_t < total_steps:
                    break

                if t == 0:  # if t is 0, then this done step caused a full update of previous seed last cycle
                    continue

                seed_t = level_seeds[start_t, actor_index].item()
                seed_idx_t = self.seed2index[seed_t]

                score_function_kwargs = {}

                score_function_kwargs["returns"] = rollouts.returns[start_t:t, actor_index]
                score_function_kwargs["rewards"] = rollouts.rewards[start_t:t, actor_index]
                score_function_kwargs["value_preds"] = rollouts.value_preds[start_t:t, actor_index]

                score = score_function(**score_function_kwargs)
                num_steps = len(rollouts.rewards[start_t:t, actor_index])
                self.update_seed_score(actor_index, seed_idx_t, score, num_steps)

                start_t = t.item()

            if start_t < total_steps:
                seed_t = level_seeds[start_t, actor_index].item()
                seed_idx_t = self.seed2index[seed_t]

                score_function_kwargs = {}

                score_function_kwargs["returns"] = rollouts.returns[start_t:, actor_index]
                score_function_kwargs["rewards"] = rollouts.rewards[start_t:, actor_index]
                score_function_kwargs["value_preds"] = rollouts.value_preds[start_t:, actor_index]

                score = score_function(**score_function_kwargs)
                num_steps = len(rollouts.rewards[start_t:, actor_index])
                self._partial_update_seed_score(actor_index, seed_idx_t, score, num_steps)

    def after_update(self):
        # Reset partial updates, since weights have changed, and thus logits are now stale
        for actor_index in range(self.partial_seed_scores.shape[0]):
            for seed_idx in range(self.partial_seed_scores.shape[1]):
                if self.partial_seed_scores[actor_index][seed_idx] != 0:
                    self.update_seed_score(actor_index, seed_idx, 0, 0)
        self.partial_seed_scores.fill(0)
        self.partial_seed_steps.fill(0)

    def _update_staleness(self, selected_idx):
        if self.staleness_coef > 0:
            self.seed_staleness = self.seed_staleness + 1
            self.seed_staleness[selected_idx] = 0

    def _score_transform(self, transform, temperature, scores):
        if transform == "constant":
            weights = np.ones_like(scores)
        if transform == "max":
            weights = np.zeros_like(scores)
            scores = scores[:]
            scores[self.unseen_seed_weights > 0] = -float("inf")  # only argmax over seen levels
            argmax = np.random.choice(np.flatnonzero(np.isclose(scores, scores.max())))
            weights[argmax] = 1.0
        elif transform == "eps_greedy":
            weights = np.zeros_like(scores)
            weights[scores.argmax()] = 1.0 - self.eps
            weights += self.eps / len(self.seeds)
        elif transform == "rank":
            temp = np.flip(scores.argsort())
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(temp)) + 1
            weights = 1 / ranks ** 0.5  # (1.0 / temperature)
        elif transform == "power":
            eps = 0 if self.staleness_coef > 0 else 1e-3
            weights = (np.array(scores) + eps) ** (1.0 / temperature)
        elif transform == "softmax":
            weights = np.exp(np.array(scores) / temperature)

        return weights


class AtariBuffer:
    def __init__(self, args, env):
        self.batch_size = args.batch_size
        self.max_size = int(args.memory_capacity)
        self.device = args.device
        self.all_seeds = args.seeds

        self.ptr = 0
        self.size = 0

        self.state = np.zeros((self.max_size, 4, 84, 84), dtype=np.uint8)
        self.action = np.zeros((self.max_size, 1), dtype=np.uint8)
        self.next_state = np.array(self.state)
        self.reward = np.zeros((self.max_size, 1))
        self.not_done = np.zeros((self.max_size, 1), dtype=np.uint8)
        self.seeds = np.zeros((self.max_size, 1), dtype=np.uint8)
        self.prioritized = args.PER
        num_updates = (args.T_max // args.num_processes - args.start_timesteps) // args.train_freq
        if self.prioritized:
            self.tree = SumTree(self.max_size)
            self.max_priority = 1.0
            self.beta = 0.4
            self.beta_stepper = (1 - self.beta) / float(num_updates)

    def add(self, state, action, next_state, reward, done, seeds):
        end = (self.ptr + 1) % self.max_size
        if "cuda" in self.device.type:
            state = (state * 255).cpu().numpy().astype(np.uint8)
            action = action.cpu().numpy().astype(np.uint8)
            next_state = (next_state * 255).cpu().numpy().astype(np.uint8)
            # reward = reward.cpu().numpy()
            # seeds = seeds.cpu().numpy().astype(np.uint8)
        else:
            state = (state * 255).numpy().astype(np.uint8)
            action = action.numpy().astype(np.uint8)
            next_state = (next_state * 255).numpy().astype(np.uint8)
            # seeds = seeds.numpy().astype(np.uint8)

        not_done = (1 - done).reshape(-1, 1)

        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = not_done
        self.seeds[self.ptr] = seeds

        self.ptr = end
        self.size = min(self.size + 1, self.max_size)

        if self.prioritized:
            self.tree.set(self.ptr, self.max_priority)

    def sample(self):
        ind = (
            self.tree.sample(self.batch_size)
            if self.prioritized
            else np.random.randint(0, self.size, size=self.batch_size)
        )

        if self.prioritized:
            weights = np.array(self.tree.nodes[-1][ind]) ** -self.beta
            weights = torch.FloatTensor(weights / weights.max()).to(self.device).reshape(-1, 1)
            self.beta = min(self.beta + self.beta_stepper, 1)
        else:
            weights = torch.FloatTensor([1]).to(self.device)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device) / 255.0,
            torch.LongTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device) / 255.0,
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.LongTensor(self.seeds[ind]).to(self.device),
            ind,
            weights,
        )

    def update_priority(self, ind, priority):
        self.max_priority = max(priority.max(), self.max_priority)
        self.tree.batch_set(ind, priority)


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space):
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        if action_space.__class__.__name__ == "Discrete":
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == "Discrete":
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        self.level_seeds = torch.zeros(num_steps, num_processes, 1, dtype=torch.int)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.level_seeds = self.level_seeds.to(device)

    def insert(
        self,
        obs,
        actions,
        value_preds,
        rewards,
        masks,
        level_seeds=None,
    ):
        if len(rewards.shape) == 3:
            rewards = rewards.squeeze(2)
        self.obs[self.step + 1].copy_(obs)
        self.actions[self.step].copy_(actions)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)

        if level_seeds is not None:
            self.level_seeds[self.step].copy_(level_seeds)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.masks[0].copy_(self.masks[-1])
        # self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(self, next_value, gamma, gae_lambda):
        self.value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(self.rewards.size(0))):
            delta = (
                self.rewards[step]
                + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                - self.value_preds[step]
            )
            gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
            self.returns[step] = gae + self.value_preds[step]


class SumTree(object):
    def __init__(self, max_size):
        self.nodes = []
        # Tree construction
        # Double the number of nodes at each level
        level_size = 1
        for _ in range(int(np.ceil(np.log2(max_size))) + 1):
            nodes = np.zeros(level_size)
            self.nodes.append(nodes)
            level_size *= 2

    # Batch binary search through sum tree
    # Sample a priority between 0 and the max priority
    # and then search the tree for the corresponding index
    def sample(self, batch_size):
        query_value = np.random.uniform(0, self.nodes[0][0], size=batch_size)
        node_index = np.zeros(batch_size, dtype=int)

        for nodes in self.nodes[1:]:
            node_index *= 2
            left_sum = nodes[node_index]

            is_greater = np.greater(query_value, left_sum)
            # If query_value > left_sum -> go right (+1), else go left (+0)
            node_index += is_greater
            # If we go right, we only need to consider the values in the right tree
            # so we subtract the sum of values in the left tree
            query_value -= left_sum * is_greater

        return node_index

    def set(self, node_index, new_priority):
        priority_diff = new_priority - self.nodes[-1][node_index]

        for nodes in self.nodes[::-1]:
            np.add.at(nodes, node_index, priority_diff)
            node_index //= 2

    def batch_set(self, node_index, new_priority):
        # Confirm we don't increment a node twice
        node_index, unique_index = np.unique(node_index, return_index=True)
        priority_diff = new_priority[unique_index] - self.nodes[-1][node_index]

        for nodes in self.nodes[::-1]:
            np.add.at(nodes, node_index, priority_diff)
            node_index //= 2


def make_buffer(args, env, atari=False):
    if atari:
        return AtariBuffer(args, env)
    if args.rank_based_PER:
        return RankBuffer(args, env)
    if args.autodrq:
        return AutoAugBuffer(args, env)
    if args.drq:
        return AugBuffer(args, env)
    return Buffer(args, env)
