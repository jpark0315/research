import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import random
import numpy as np 

from bnn import BNN 
from replay import ReplayBuffer


class MBPO:

    def __init__(self, args, agent, env, ensemble = True):

        if not ensemble:
            self.model = BNN(args).to(args.device)
            self.model_optim = torch.optim.Adam(self.model.parameters(), lr = args.model_lr)

        else:
            self.models = [BNN(args).to(args.device) for _ in range(args.ensemble_size)]
            self.model_optims = [torch.optim.Adam(m.parameters(), lr = args.model_lr) for m in self.models]

        self.ensemble = ensemble
        self.args = args

        self.D_env = ReplayBuffer(args.D_env_size, args.seed, args, true_buffer = True)
        self.D_model = ReplayBuffer(args.D_model_size, args.seed, args)

        self.agent = agent
        self.env = env

        self.episode_reward = 0
        self.episode_timesteps = 0
        self.updates = 0

    def collect_random_rollouts(self):

        random_timesteps = 2000
        episode_steps = 0
        print('Collecting Random Rollouts...')
        state = self.env.reset()
        for _ in range(random_timesteps):

            action = self.env.action_space.sample()
            next_state, reward, done, _ = self.env.step(action)

            mask = 1 if episode_steps == self.env._max_episode_steps else float(not done)
            self.D_env.push(state, action, reward, next_state, mask)
            state = next_state
            episode_steps += 1
            if done:
                episode_steps = 0
                state = self.env.reset()


    def simulate(self, logger):
        """
        For num_model_rollouts, sample random state from D_env
        and perform model_rollout_step from that state
        and add data to D_model
        """
        simul_reward = 0
        masks = 0
        print('Collecting simulated rollouts...')
        for _ in (range(self.args.num_model_rollouts)):
            state = self.D_env.sample_state()

            for _ in range(self.args.model_rollout_step):

                if self.ensemble:
                    model = random.choice(self.models)
                else:
                    model = self.model

                action, _ = self.agent.select_action(state)

                output = model(state, action).detach().squeeze(0).cpu().numpy()
                next_state , reward = output[:-1], output[-1]
                mask = float(not self.termination_fn(next_state))
                self.D_model.push(state, action, reward, next_state, mask)

                state = next_state
                simul_reward += reward
                masks += mask

        logger.log('avg_simulated reward', simul_reward/(self.args.num_model_rollouts * self.args.model_rollout_step))
        logger.log('Dones', masks/(self.args.num_model_rollouts * self.args.model_rollout_step))







        