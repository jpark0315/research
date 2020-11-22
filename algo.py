import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import random
import numpy as np 

from bnn import BNN 
from non_bnn import Ensemble
from replay import ReplayBuffer


class MBPO:

    def __init__(self, args, agent, env):


        self.models = Ensemble(args)
        self.args = args
        self.MLE  = nn.MSELoss()
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
        prev_length = np.copy(len(self.D_model))
        simul_reward = 0
        masks = 0
        print('Collecting simulated rollouts...')
        for _ in (range(self.args.num_model_rollouts)):
            state = self.D_env.sample_state()

            for _ in range(self.args.model_rollout_step):

                action, log_prob = self.agent.select_action(state)

                # ensemble_mean, ensemble_logvar = self.models.predict(state, action)    #This is actually not logvar
                # assert len(ensemble_mean) == self.args.ensemble_size
                # cur_model_index = np.random.randint(self.args.ensemble_size)
                # cur_mean, cur_logvar = ensemble_mean[cur_model_index], ensemble_logvar[cur_model_index]

                cur_mean, cur_logvar = self.models.predict(state, action, one_model = True)
                output = cur_mean + np.random.normal(size = cur_mean.shape) * np.sqrt(cur_logvar)
                assert len(output.shape) == 2
                output = output.reshape(-1)

                next_state , reward = output[:-1] + state, output[-1]
                done = self.termination_fn(next_state)
                mask = float(not done)

                self.D_model.push(state, action, reward, next_state, mask)
                state = next_state
                simul_reward += reward
                masks += mask
                if done:
                    break 
                logger.log('Simul Log Prob', log_prob)
    
        logger.log('avg_simulated reward', simul_reward/(self.args.num_model_rollouts * self.args.model_rollout_step))
        logger.log('Dones', masks/(self.args.num_model_rollouts * self.args.model_rollout_step))

    def update_policy(self, logger):

        if len(self.D_model) > self.args.batch_size:
            print('Updating Policy...')
            self.updates += 1
            for i in (range(self.args.policy_update_per_step)):
                if i % 10 != 0:
                
                    (critic_1_loss, critic_2_loss,
                    policy_loss, qf1, target_q) = self.agent.update_parameters(self.D_model,
                                                        self.args.batch_size, self.updates)

                else:
                    (critic_1_loss, critic_2_loss,
                    policy_loss, qf1, target_q) = self.agent.update_parameters(self.D_env,
                                                        self.args.batch_size, self.updates)

                logger.log('c1loss', critic_1_loss)
                logger.log('c2loss', critic_2_loss)
                logger.log('ploss', policy_loss)
                logger.log('critic_q', qf1)
                logger.log('target_q', target_q)

    def train_model(self, logger):
        """
        Called start of every episode when len(D_env) > model_batch_size
        """
        #Sample real data from D_env
      
        for _ in range(self.args.model_train_step):
            (state_batch, action_batch, reward_batch, 
                next_state_batch, _) = self.D_env.sample(self.args.model_batch_size)
            delta_state = next_state_batch - state_batch 

            state_batch = torch.FloatTensor(state_batch).to(self.args.device)
            action_batch = torch.FloatTensor(action_batch).to(self.args.device)
            delta_state = torch.FloatTensor(delta_state).to(self.args.device)
            reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1)

            assert len(state_batch) == len(delta_state) & len(reward_batch) == len(delta_state)
            labels = torch.cat((delta_state, reward_batch), 1).to(self.args.device)

            self.models.fit(state_batch, action_batch, labels, logger)


    def eval_model(self, logger):
        """
        Evaluate Model on freshly arrived real samples
        """
        (state_batch, action_batch, reward_batch, 
                next_state_batch, _) = self.D_env.sample(self.args.model_batch_size, train = False)
        delta_state = next_state_batch - state_batch 

        state_batch = torch.FloatTensor(state_batch).to(self.args.device)
        action_batch = torch.FloatTensor(action_batch).to(self.args.device)
        delta_state = torch.FloatTensor(delta_state).to(self.args.device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1)

        assert len(state_batch) == len(delta_state) & len(delta_state) == len(reward_batch)
        labels = torch.cat((delta_state, reward_batch), 1).to(self.args.device)
        self.models.eval(state_batch, action_batch, labels, logger)


    def one_step(self, state, logger):
        """
        Go through one timestep and append it to d_env buffer
        """
        action, log_prob = self.agent.select_action(state)
        next_state, reward, done, _ = self.env.step(action)
        mask = 1 if self.episode_timesteps == self.env._max_episode_steps else float(not done)
        self.D_env.push(state, action, reward, next_state, mask)

        self.episode_reward += reward
        self.episode_timesteps += 1
        if done:
            next_state = self.env.reset()
            print('_____________________')
            print('EPISODE DONE,TIMESTEP/REWARD:',self.episode_timesteps, self.episode_reward)
            logger.log('episode_reward', self.episode_reward)
            self.episode_reward = 0
            self.episode_timesteps = 0
        logger.log('action', action)
        logger.log('log_prob', log_prob)
        return next_state

    def run(self, logger):

        state = self.env.reset()
        for i in range(self.args.N_epochs):
            print()
            print('EPOCH', i+1, 'TimeStep', self.args.E_steps *i,
                'D_env size', len(self.D_env))
            self.train_model(logger)
            self.eval_model(logger)

            for _ in range(self.args.E_steps):  #Is this while not done?

                state = self.one_step(state, logger)
                self.simulate(logger)
                self.update_policy(logger)

    @staticmethod
    def termination_fn(next_obs):
        next_obs = np.expand_dims(next_obs, 0)
        assert len(next_obs.shape) == 2

        height = next_obs[:, 0]
        angle = next_obs[:, 1]
        not_done =  np.isfinite(next_obs).all(axis=-1) \
                    * np.abs(next_obs[:,1:] < 100).all(axis=-1) \
                    * (height > .7) \
                    * (np.abs(angle) < .2)

        done = ~not_done
        done = done[:,None]
        return done




