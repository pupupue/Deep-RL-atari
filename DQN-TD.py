#!/usr/bin/env python
import sys
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import pandas as pd
from datetime import datetime
from argparse import ArgumentParser

from neurals.conv import CONV
from neurals.linear import NNet
import environment.atari as Atari
import environment.utils as env_utils
import utils.weights_initializer
from utils.params_manager import ParamsManager
from utils.decay_schedule import LinearDecaySchedule
from utils.experience_memory import Experience, ExperienceMemory
from torch.utils.tensorboard import SummaryWriter
from pympler import muppy, summary
"""
TODO:
    rewrite TD to double Q learning
"""
 
args = ArgumentParser("deep_Q_learner")
args.add_argument(
    "--params-file",
    help="Path to the parameters json file. Default is parameters.json",
    default="parameters.json", 
    metavar="PFILE"
)
args.add_argument(
    "--env", 
    help="ID of the Atari environment available in OpenAI Gym.Default is SeaquestNoFrameskip-v4",
    default="SeaquestNoFrameskip-v4",
    metavar="ENV"
)
args.add_argument("--gpu-id", help="GPU device ID to use. Default=0", default=0, type=int, metavar="GPU_ID")
args.add_argument("--render", help="Render environment to Screen. Off by default", action="store_true", default=False)
args.add_argument("--test", help="Test mode. Used for playing without learning. Off by default", action="store_true", default=False)
args.add_argument("--record", help="Enable recording (video & stats) of the agent's performance", action="store_true", default=False)
args.add_argument(
    "--recording-output-dir", 
    help="Directory to store monitor outputs. Default=./trained_models/results", 
    default="./trained_models/results"
)
args = args.parse_args()

params_manager= ParamsManager(args.params_file)
seed = params_manager.get_agent_params()['seed']
summary_file_path_prefix = params_manager.get_agent_params()['summary_file_path_prefix']
summary_file_path= summary_file_path_prefix + args.env+ "_" + datetime.now().strftime("%y-%m-%d-%H-%M")
writer = SummaryWriter(summary_file_path)
# Export the parameters as json files to the log directory to keep track of the parameters used in each experiment
params_manager.export_env_params(summary_file_path + "/" + "env_params.json")
params_manager.export_agent_params(summary_file_path + "/" + "agent_params.json")
global_step_num = 0
use_cuda = params_manager.get_agent_params()['use_cuda']
# new in PyTorch 0.4
device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() and use_cuda else "cpu")
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available() and use_cuda:
    torch.cuda.manual_seed_all(seed)

class Deep_Q_Learner(object):
    def __init__(self, state_shape, action_shape, params):
        """
        self.Q is the Action-Value function. This agent represents Q using a Neural Network
        If the input is a single dimensional vector, uses a Single-Layer-Perceptron else if the input is 3 dimensional
        image, use a Convolutional-Neural-Network

        :param state_shape: Shape (tuple) of the observation/state
        :param action_shape: Shape (number) of the discrete action space
        :param params: A dictionary containing various Agent configuration parameters and hyper-parameters
        """
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.params = params
        self.gamma = self.params['gamma']  # Agent's discount factor
        self.learning_rate = self.params['lr']  # Agent's Q-learning rate
        self.best_mean_reward = - float("inf") # Agent's personal best mean episode reward
        self.best_reward = - float("inf")
        self.training_steps_completed = 0  # Number of training batch steps completed so far

        if len(self.state_shape) == 1:  # Single dimensional observation/state space
            self.DQN = NNet
        elif len(self.state_shape) == 3:  # 3D/image observation/state
            self.DQN = CONV

        self.Q = self.DQN(state_shape, action_shape, device).to(device)
        self.Q.apply(utils.weights_initializer.xavier)

        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr=self.learning_rate)
        if self.params['use_target_network']:
            self.Q_target = self.DQN(state_shape, action_shape, device).to(device)
        # self.policy is the policy followed by the agent. This agents follows
        # an epsilon-greedy policy w.r.t it's Q estimate.
        self.policy = self.epsilon_greedy_Q
        self.epsilon_max = params["epsilon_max"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = LinearDecaySchedule(
            initial_value=self.epsilon_max,
            final_value=self.epsilon_min,
            max_steps= self.params['epsilon_decay_final_step']
        )
        self.step_num = 0
        self.memory = ExperienceMemory(capacity=int(self.params['experience_memory_capacity']))  # Initialize an Experience memory with 1M capacity

    """
    GET_ACTION: returns current action from our policy

    :param observation:  
    """
    def get_action(self, observation):
        observation = np.array(observation)  # Observations could be lazy frames. So force fetch before moving forward
        observation = observation / 255.0  # Scale/Divide by max limit of obs' dtype. 255 for uint8
        if len(observation.shape) == 3: # Single image (not a batch)
            if observation.shape[2] < observation.shape[0]:  # Probably observation is in W x H x C format
                # NOTE: This is just an additional check. The env wrappers are taking care of this conversion already
                # Reshape to C x H x W format as per PyTorch's convention
                observation = observation.reshape(observation.shape[2], observation.shape[1], observation.shape[0])
            observation = np.expand_dims(observation, 0)  # Create a batch dimension
        return self.policy(observation)

    def epsilon_greedy_Q(self, observation):
        # Decay Epsilon/exploration as per schedule
        writer.add_scalar("DQL/epsilon", self.epsilon_decay(self.step_num), self.step_num)
        self.step_num +=1
        if random.random() < self.epsilon_decay(self.step_num) and not self.params["test"]:
            action = random.choice([i for i in range(self.action_shape)])
        else:
            action = np.argmax(self.Q(observation).data.to(torch.device('cpu')).numpy())
        return action

    def learn(self, s, a, r, s_next, done):
        # TD(0) Q-learning
        if done:  # End of episode
            td_target = reward + 0.0  # Set the value of terminal state to zero
        else:
            td_target = r + self.gamma * torch.max(self.Q(s_next))
        td_error = td_target - self.Q(s)[a]
        # Update Q estimate
        #self.Q(s)[a] = self.Q(s)[a] + self.learning_rate * td_error
        self.Q_optimizer.zero_grad()
        td_error.backward()
        self.Q_optimizer.step()

    def learn_from_batch_experience(self, experiences):
        batch_xp = Experience(*zip(*experiences))
        obs_batch = np.array(batch_xp.obs) / 255.0  # Scale/Divide by max limit of obs's dtype. 255 for uint8
        action_batch = np.array(batch_xp.action)
        reward_batch = np.array(batch_xp.reward)
        # Clip the rewards
        if self.params["clip_rewards"]:
            reward_batch = np.sign(reward_batch)
        next_obs_batch = np.array(batch_xp.next_obs) / 255.0  # Scale/Divide by max limit of obs' dtype. 255 for uint8
        done_batch = np.array(batch_xp.done)
        if self.params['use_target_network']:
            #if self.training_steps_completed % self.params['target_network_update_freq'] == 0:
            if self.step_num % self.params['target_network_update_freq'] == 0:
                # The *update_freq is the Num steps after which target net is updated.
                # A schedule can be used instead to vary the update freq.
                self.Q_target.load_state_dict(self.Q.state_dict())
            td_target = reward_batch + ~done_batch * np.tile(self.gamma, len(next_obs_batch)) * \
                self.Q_target(next_obs_batch).max(1)[0].data.cpu().numpy()
        else:
            td_target = reward_batch + ~done_batch * np.tile(self.gamma, len(next_obs_batch)) * \
                self.Q(next_obs_batch).detach().max(1)[0].data.cpu().numpy()

        td_target = torch.from_numpy(td_target).to(device)
        action_idx = torch.from_numpy(action_batch).to(device)
        td_error = F.mse_loss(self.Q(obs_batch).gather(1, action_idx.view(-1, 1)), td_target.float().unsqueeze(1))

        self.Q_optimizer.zero_grad()
        td_error.mean().backward()
        writer.add_scalar("DQL/td_error", td_error.mean(), self.step_num)
        self.Q_optimizer.step()

    def replay_experience(self, batch_size = None):
        batch_size = batch_size if batch_size is not None else self.params['replay_batch_size']
        experience_batch = self.memory.sample(batch_size)
        self.learn_from_batch_experience(experience_batch)
        self.training_steps_completed += 1  # Increment the number of training batch steps complemented

    def save(self, env_name):
        file_name = self.params['save_dir'] + "DQL_" + env_name + ".ptm"
        agent_state = {
            "Q": self.Q.state_dict(),
            "best_mean_reward": self.best_mean_reward,
            "best_reward": self.best_reward};
        torch.save(agent_state, file_name)
        print("Agent's state saved to ", file_name)

    def load(self, env_name):
        file_name = self.params['load_dir'] + "DQL_" + env_name + ".ptm"
        agent_state = torch.load(file_name, map_location= lambda storage, loc: storage)
        self.Q.load_state_dict(agent_state["Q"])
        self.Q.to(device)
        self.best_mean_reward = agent_state["best_mean_reward"]
        self.best_reward = agent_state["best_reward"]
        print("Loaded Q model state from", file_name,
              " which fetched a best mean reward of:", self.best_mean_reward,
              " and an all time best reward of:", self.best_reward)

if __name__ == "__main__":
    env_conf = params_manager.get_env_params()
    env_conf["env_name"] = args.env
    # In test mode, let the end of the game be the end of episode rather than ending episode at the end of every life.
    # This helps to report out the (mean and max) episode rewards per game (rather than per life!)
    if args.test:
        env_conf["episodic_life"] = False
    # Specify the reward calculation type used for printing stats at the end of every episode.
    # If "episode_life" is true, the printed stats (reward, mean reward, max reward) are per life. If "episodic_life"
    # is false, the printed stats/scores are per game in Atari environments
    rew_type = "LIFE" if env_conf["episodic_life"] else "GAME"

    # If a custom useful_region configuration for this environment ID is available, use it if not use the Default
    custom_region_available = False
    for key, value in env_conf['useful_region'].items():
        if key in args.env:
            env_conf['useful_region'] = value
            custom_region_available = True
            break
    if custom_region_available is not True:
        env_conf['useful_region'] = env_conf['useful_region']['Default']

    print("Using env_conf:", env_conf)
    atari_env = False
    for game in Atari.get_games_list():
        if game.replace("_", "") in args.env.lower():
            atari_env = True
    if atari_env:
        env = Atari.make_env(args.env, env_conf) 
    else:
        print("Given environment name is not an Atari Env. Creating a Gym env")
        # Resize the obs to w x h (84 x 84 by default) and then reshape it to be in the C x H x W format
        env = env_utils.ResizeReshapeFrames(gym.make(args.env))
    if args.record:  # If monitor is enabled, record stats and video of agent's performance
        env = gym.wrappers.Monitor(env, args.recording_output_dir, force=True)

    observation_shape = env.observation_space.shape
    action_shape = env.action_space.n
    print(action_shape)
    agent_params = params_manager.get_agent_params()
    agent_params["test"] = args.test
    agent = Deep_Q_Learner(observation_shape, action_shape, agent_params)

    episode_rewards = list()
    prev_checkpoint_mean_ep_rew = agent.best_mean_reward
    num_improved_episodes_before_checkpoint = 0  # To keep track of the num of ep with higher perf to save model
    print("Using agent_params:", agent_params)
    if agent_params['load_trained_model']:
        try:
            agent.load(env_conf["env_name"])
            prev_checkpoint_mean_ep_rew = agent.best_mean_reward
        except FileNotFoundError:
            print("WARNING: No trained model found for this environment. Training from scratch.")

    episode = 0
    while global_step_num <= agent_params['max_training_steps']:
        obs = env.reset()
        cum_reward = 0.0  # Cumulative reward
        done = False
        step = 0

        while not done:
            if env_conf['render'] or args.render:
                env.render()
            action = agent.get_action(obs)
            next_obs, reward, done, info = env.step(action)

            """
            STORING MEMORY:
                this stores all memories, but i think non trivial memories
                should be stored only 

                repeat rewarding / punishing memories 
                VS
                add less trivial memories (chosen tactic)
            """
            if reward != 0:
                agent.memory.store(Experience(obs, action, reward, next_obs, done))
            else:
                if random.random() > 0.8:
                    agent.memory.store(Experience(obs, action, reward, next_obs, done))
            """
            STATISTICS:
            """
            obs = next_obs
            cum_reward += reward
            step += 1
            global_step_num += 1

            if done is True:
                episode += 1
                episode_rewards.append(cum_reward)
                if cum_reward > agent.best_reward:
                    agent.best_reward = cum_reward
                if np.mean(episode_rewards) > prev_checkpoint_mean_ep_rew:
                    num_improved_episodes_before_checkpoint += 1
                if num_improved_episodes_before_checkpoint >= agent_params["save_freq_when_perf_improves"]:
                    prev_checkpoint_mean_ep_rew = np.mean(episode_rewards)
                    agent.best_mean_reward = np.mean(episode_rewards)
                    agent.save(env_conf['env_name'])
                    num_improved_episodes_before_checkpoint = 0
                """
                Viewing Policy:
                """
                if episode == 1:
                    print(agent.policy)
                # Print Statement
                print(f"[{(global_step_num / agent_params['max_training_steps'])*100:.2f}%]", end="\t")
                print(f"Episode# {episode} ended in {step+1} steps.", end="\t")
                print(f"mem.size[{agent.memory.get_size()}]", end="\t")
                print(f"Per {rew_type} stats: [reward = {cum_reward},  ", end="\t")
                print(f"mean={np.mean(episode_rewards):.3f}: best={agent.best_reward}]")
                # # # # # # # # #
                writer.add_scalar("main/ep_reward", cum_reward, global_step_num)
                writer.add_scalar("main/mean_ep_reward", np.mean(episode_rewards), global_step_num)
                writer.add_scalar("main/max_ep_rew", agent.best_reward, global_step_num)
                # Learn from batches of experience once a certain amount of xp is available unless in test only mode
                # print(f"mem size = {agent.memory.get_size()}")
                if agent.memory.get_size() >= 2 * agent_params['replay_start_size'] and not args.test:
                    agent.replay_experience()

                break
    env.close()
    writer.close()
