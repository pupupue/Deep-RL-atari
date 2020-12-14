#!/usr/bin/env python 
import gym
import torch 
import random
import numpy as np
from torch.autograd import Variable
from utils.decay_schedule import LinearDecaySchedule
from neurals.linear import NNet

env = gym.make("CartPole-v0")
MAX_NUM_EPISODES = 100000
MAX_STEPS_PER_EPISODE = 300

class Shallow_Q_Learner(object):
    def __init__(self, state_shape, action_shape, learning_rate=0.005, gamma=0.98):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.gamma = gamma # Agent's discount factor
        self.learning_rate = learning_rate # Agent's Q-learning rate
        # self.Q is the Action-Value function. This agent represents Q using a
        # Neural Network.
        self.Q = NNet(state_shape, action_shape, hidden_shape=40)
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr=1e-3)
        # self.policy is the policy followed by the agent. This agents follows
        # an epsilon-greedy policy w.r.t it's Q estimate.
        self.policy = self.epsilon_greedy_Q
        self.epsilon_max = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = LinearDecaySchedule(
            initial_value=self.epsilon_max,
            final_value=self.epsilon_min,
            max_steps= 0.5 * MAX_NUM_EPISODES * MAX_STEPS_PER_EPISODE
        )
        self.step_num = 0

    def get_action(self, observation):
        return self.policy(observation)

    def epsilon_greedy_Q(self, observation):
        # Decay Epsilion/exploratin as per schedule
        if random.random() < self.epsilon_decay(self.step_num):
            action = random.choice([i for i in range(self.action_shape)])
        else:
            action = np.argmax(self.Q(observation).data.numpy())
        return action

    def learn(self, s, a, r, s_next):
        td_target = r + self.gamma * torch.max(self.Q(s_next))
        td_error = torch.nn.functional.mse_loss(self.Q(s)[a], td_target)
        # Update Q estimate
        #self.Q(s)[a] = self.Q(s)[a] + self.learning_rate * td_error
        self.Q_optimizer.zero_grad()
        td_error.backward()
        self.Q_optimizer.step()

if __name__ == "__main__":
    observation_shape = env.observation_space.shape
    action_shape = env.action_space.n
    agent = Shallow_Q_Learner(observation_shape, action_shape)
    first_episode = True
    episode_rewards = list()
    for episode in range(MAX_NUM_EPISODES):
        obs = env.reset()
        cum_reward = 0.0 # Cumulative reward
        for step in range(MAX_STEPS_PER_EPISODE):
            # env.render()
            action = agent.get_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.learn(obs, action, reward, next_obs)

            obs = next_obs
            cum_reward += reward

            if done is True:
                if first_episode: # Initialize max_reward at the end of first episode
                    max_reward = cum_reward
                    first_episode = False
                episode_rewards.append(cum_reward)
                if cum_reward > max_reward:
                    max_reward = cum_reward
                # print
                if(episode % 100 == 0):
                    print(f"Episode #{episode} ended in {step+1} steps, reward ={cum_reward}, mean_reward={np.mean(episode_rewards):.4f}, \tbest_reward={max_reward}")
                break
    env.close()