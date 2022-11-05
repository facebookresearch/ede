# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from typing import List

from level_replay.algo.dqn import ImpalaCNN
from level_replay.envs import make_dqn_lr_venv

from torch import nn
from torch.nn import functional as F


class Policy(nn.Module):
    def __init__(self, args, env):
        super(Policy, self).__init__()
        self.device = args.device
        self.command_scale = torch.FloatTensor([args.return_scale, args.horizon_scale]).to(self.device)
        self.state_embedding = nn.Sequential(
            ImpalaCNN(env.observation_space.shape[0]),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.Tanh(),
        )
        self.command_embedding = nn.Sequential(nn.Linear(2, 128), nn.Sigmoid())
        self.output = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, env.action_space.n))

        self.create_optimizer(args.learning_rate)

    def forward(self, state, command):
        state_emb = self.state_embedding(state)
        command_emb = self.command_embedding(command * self.command_scale)
        emb = torch.mul(state_emb, command_emb)
        action_logits = self.output(emb)
        return action_logits

    def act(self, state, command, greedy=False):
        action_logits = self.forward(state, command)
        action_probs = F.softmax(action_logits, dim=-1)
        if greedy:
            action = torch.argmax(action_probs, 1)
        else:
            action = torch.distributions.Categorical(action_probs).sample()
        return action

    def create_optimizer(self, lr):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)


class EpisodeMemory:
    def __init__(self):
        self.state = []
        self.action = []
        self.reward = []

    def add(self, state, action, reward):
        state_ = (state * 255).cpu().numpy().astype(np.uint8)
        action_ = action.cpu().numpy().astype(np.uint8)
        self.state.append(state_)
        self.action.append(action_)
        self.reward.append(reward)

    def write(self, ep_return, ep_length):
        d = {}
        d["state"] = self.state
        d["action"] = self.action
        d["reward"] = self.reward
        d["ep_return"] = ep_return
        d["ep_length"] = ep_length

        return d

    def clear(self):
        self.state = []
        self.action = []
        self.reward = []
        self.ep_return = 0
        self.ep_length = 0


class Memory:
    def __init__(self, args, env):
        self.batch_size = args.batch_size
        self.replay_size = int(args.replay_size)
        self.device = args.device

        self.ptr = 0
        self.size = 0

        self.state_shape = env.observation_space.shape

        self.episodes = []

    def add(self, episode):
        self.episodes.append(episode)
        self.size = min(self.size + 1, self.replay_size)

    def sort(self):
        size = min(self.size, self.replay_size)
        self.episodes = sorted(self.episodes, key=lambda x: x["ep_return"])[-size:]

    def sample_single(self):
        idx = np.random.randint(0, self.size)
        t1 = np.random.randint(0, len(self.episodes[idx]["state"]))
        t2 = self.episodes[idx]["ep_length"]
        length = t2 - t1
        state = self.episodes[idx]["state"][t1]
        desired_reward = sum(self.episodes[idx]["reward"][t1:t2])
        action = self.episodes[idx]["action"][t1]
        return state, [desired_reward, length], action

    def sample(self):
        states = np.zeros((self.batch_size, *self.state_shape))
        commands = np.zeros((self.batch_size, 2))
        actions = np.zeros((self.batch_size))
        for i in range(self.batch_size):
            states[i], commands[i], actions[i] = self.sample_single()

        states = torch.FloatTensor(states).to(self.device) / 255.0
        commands = torch.FloatTensor(commands).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)

        return states, commands, actions


def update(policy, mem, n_updates=100):
    all_losses = []
    for _ in range(n_updates):
        states, commands, actions = mem.sample()
        pred = policy.forward(states, commands)

        loss = F.cross_entropy(pred, actions)
        all_losses.append(loss.item())

        policy.optimizer.zero_grad()
        loss.backward()
        policy.optimizer.step()

    return np.mean(all_losses)


def sample_commands(mem, last_few=50):
    if mem.size == 0:
        return [1, 1]
    else:
        last_few = min(last_few, mem.size)

        command_samples = mem.episodes[-last_few:]
        # [mem.episodes[i] for i in range(-last_few, 0)]
        lengths = [ep_mem["ep_length"] for ep_mem in command_samples]
        returns = [ep_mem["ep_return"] for ep_mem in command_samples]
        mean_return, std_return = np.mean(returns), np.std(returns)
        command_horizon = np.mean(lengths)
        desired_reward = np.random.uniform(mean_return, mean_return + std_return)
        return [desired_reward, command_horizon]


def generate_episodes(policy, mem, commands, env_steps, args):
    returns: List[float] = []
    commands_ = torch.FloatTensor([commands for _ in range(args.num_processes)]).to(args.device)
    ep_memories = [EpisodeMemory() for _ in range(args.num_processes)]

    num_levels = 1
    level_sampler_args = dict(
        num_actors=args.num_processes,
        strategy=args.level_replay_strategy,
    )
    envs, level_sampler = make_dqn_lr_venv(
        num_envs=args.num_processes,
        env_name=args.env_name,
        seeds=args.seeds,
        device=args.device,
        num_levels=num_levels,
        start_level=args.start_level,
        no_ret_normalization=args.no_ret_normalization,
        distribution_mode=args.distribution_mode,
        paint_vel_info=args.paint_vel_info,
        use_sequential_levels=args.use_sequential_levels,
        level_sampler_args=level_sampler_args,
    )

    level_seeds = torch.zeros(args.num_processes)
    if level_sampler:
        state, level_seeds = envs.reset()
    else:
        state = envs.reset()

    level_seeds = level_seeds.unsqueeze(-1)

    while len(returns) < args.n_episodes_per_iter:
        action = policy.act(state, commands_).reshape(-1, 1)
        next_state, reward, done, infos = envs.step(action)

        env_steps += args.num_processes

        for i, info in enumerate(infos):
            commands_[i] -= torch.FloatTensor([reward[i][0], 1]).to(args.device)
            if level_sampler:
                level_seed = info["level_seed"]
                if level_seeds[i][0] != level_seed:
                    level_seeds[i][0] = level_seed
            ep_memories[i].add(state[i], action[i], reward[i])
            if "episode" in info.keys():
                episode_reward = info["episode"]["r"]
                ep = ep_memories[i].write(episode_reward, info["episode"]["l"])
                mem.add(ep)
                ep_memories[i].clear()
                returns.append(episode_reward)
                commands_[i] = torch.FloatTensor(commands).to(args.device)

        state = next_state

    envs.close()
    mem.sort()

    return env_steps


def warm_up(args):
    num_levels = 1
    level_sampler_args = dict(
        num_actors=args.num_processes,
        strategy=args.level_replay_strategy,
    )
    envs, level_sampler = make_dqn_lr_venv(
        num_envs=args.num_processes,
        env_name=args.env_name,
        seeds=args.seeds,
        device=args.device,
        num_levels=num_levels,
        start_level=args.start_level,
        no_ret_normalization=args.no_ret_normalization,
        distribution_mode=args.distribution_mode,
        paint_vel_info=args.paint_vel_info,
        use_sequential_levels=args.use_sequential_levels,
        level_sampler_args=level_sampler_args,
    )

    mem = Memory(args, envs)

    level_seeds = torch.zeros(args.num_processes)
    if level_sampler:
        state, level_seeds = envs.reset()
    else:
        state = envs.reset()

    level_seeds = level_seeds.unsqueeze(-1)

    env_steps = 0

    # Run warm-up episodes with random actions
    warm_ups: List[float] = []
    ep_memories = [EpisodeMemory() for _ in range(args.num_processes)]
    while len(warm_ups) < args.n_warm_up_episodes:
        action = (
            torch.LongTensor([envs.action_space.sample() for _ in range(args.num_processes)])
            .reshape(-1, 1)
            .to(args.device)
        )
        next_state, reward, done, infos = envs.step(action)
        env_steps += args.num_processes

        for i, info in enumerate(infos):
            if level_sampler:
                level_seed = info["level_seed"]
                if level_seeds[i][0] != level_seed:
                    level_seeds[i][0] = level_seed
            ep_memories[i].add(state[i], action[i], reward[i])
            if "episode" in info.keys():
                episode_reward = info["episode"]["r"]
                ep = ep_memories[i].write(episode_reward, info["episode"]["l"])
                mem.add(ep)
                ep_memories[i].clear()
                warm_ups.append(episode_reward)

        state = next_state

    policy = Policy(args, envs).to(args.device)

    envs.close()

    mem.sort()

    return mem, env_steps, policy


def evaluate(
    args,
    policy,
    commands,
    num_episodes=10,
    num_processes=1,
    deterministic=False,
    start_level=0,
    num_levels=0,
    seeds=None,
    level_sampler=None,
    progressbar=None,
    record=False,
    print_score=True,
):
    if level_sampler:
        start_level = level_sampler.seed_range()[0]
        num_levels = 1

    eval_envs, level_sampler = make_dqn_lr_venv(
        num_envs=num_processes,
        env_name=args.env_name,
        seeds=seeds,
        device=args.device,
        num_levels=num_levels,
        start_level=start_level,
        no_ret_normalization=args.no_ret_normalization,
        distribution_mode=args.distribution_mode,
        paint_vel_info=args.paint_vel_info,
        level_sampler=level_sampler,
        record_runs=record,
    )

    if level_sampler:
        state, level_seeds = eval_envs.reset()
    else:
        state = eval_envs.reset()

    commands_ = torch.FloatTensor([commands for _ in range(num_processes)]).to(args.device)

    eval_episode_rewards: List[float] = []

    while len(eval_episode_rewards) < num_episodes:
        action = policy.act(state, commands_, greedy=deterministic).reshape(-1, 1)
        next_state, reward, done, infos = eval_envs.step(action)

        for i, info in enumerate(infos):
            commands_[i] -= torch.FloatTensor([reward[i][0], 1]).to(args.device)
            if level_sampler:
                level_seed = info["level_seed"]
                if level_seeds[i][0] != level_seed:
                    level_seeds[i][0] = level_seed
            if "episode" in info.keys():
                episode_reward = info["episode"]["r"]
                eval_episode_rewards.append(episode_reward)
                commands_[i] = torch.FloatTensor(commands).to(args.device)

    eval_envs.close()

    return eval_episode_rewards
