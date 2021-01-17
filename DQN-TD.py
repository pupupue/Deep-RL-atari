#!/usr/bin/env python
import sys
import gym
import json
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
from neurals.dueling_neurals import DDQN
import environment.atari as Atari
import environment.utils as env_utils
import utils.weights_initializer
from utils.params_manager import ParamsManager
from utils.decay_schedule import LinearDecaySchedule
from utils.experience_memory import Experience, ExperienceMemory
from utils.prioritized_memory import PrioritizedMemory
from torch.utils.tensorboard import SummaryWriter
import tracemalloc

tracemalloc.start()

args = ArgumentParser("deep_Q_learner")
args.add_argument(
    "--params-file",
    help="Path to the parameters json file. Default is parameters.json",
    default="parameters.json", 
    metavar="PFILE"
)
args.add_argument(
    "--env", 
    help="ID of the Atari environment available in OpenAI Gym.Default is SpaceInvaders-v4",
    default="SpaceInvaders-v4",
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
use_cuda = params_manager.get_agent_params()['use_cuda']
device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() and use_cuda else "cpu")
torch.manual_seed(seed)
np.random.seed(seed)
# CUDA THINGS
if torch.cuda.is_available() and use_cuda:
    torch.cuda.manual_seed_all(seed)


class Deep_Q_Learner(object):
    def __init__(self, state_shape, action_shape, params):
        """   HYPERPARAMATERS   """
        """
            :param state_shape: Shape (tuple) of the observation/state
            :param action_shape: Shape (number) of the discrete action space
            :param params: A dictionary containing various Agent configuration parameters and hyper-parameters
        """
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.action_dict = {}
        self.params = params
        self.gamma = self.params['gamma']  # Agent's discount factor
        self.learning_rate = self.params['lr']  # Agent's Q-learning rate
        self.prioritized_memory = self.params['prioritized_memory'] # use [prio mem] BOOL
        self.best_mean_reward = -float("inf") # Agent's personal best mean episode reward
        self.best_reward = -float("inf")
        self.max_training_steps = agent_params['max_training_steps']
        self.rew_type = "LIFE" if env_conf["episodic_life"] else "GAME"
        """   For live & offline statistics   """
        self.training_steps_completed = 0  # Number of training batch steps completed so far
        self.step_num = 0
        self.ep_num = 0
        #reward tracking
        self.reward_length = 500
        self.episode_rewards = []
        self.prev_checkpoint_mean_ep_rew = 0
        self.num_improved_episodes_before_checkpoint = 0

        if len(self.state_shape) == 1:  # Single dimensional observation/state space
            self.DQN = NNet
        elif len(self.state_shape) == 3:  # 3D/image observation/state
            self.DQN = CONV

        """   Q net   """
        self.Q = self.DQN(state_shape, action_shape, device).to(device)
        self.Q.apply(utils.weights_initializer.xavier)
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr=self.learning_rate)

        """   Tnet   """
        if self.params['use_target_network']:
            self.Q_target = self.DQN(state_shape, action_shape, device).to(device)

        """   Policy   """
        """
        self.policy is the policy followed by the agent. This agents follows
        an epsilon-greedy policy w.r.t it's Q estimate.
        """
        self.policy = self.epsilon_greedy_Q
        self.epsilon_max = params["epsilon_max"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = LinearDecaySchedule(
            initial_value=self.epsilon_max,
            final_value=self.epsilon_min,
            max_steps= self.params['epsilon_decay_final_step']
        )

        """   Memory   """
        if self.prioritized_memory:
            """ Prioritized Memory replay """
            self.memory = PrioritizedMemory(capacity=int(self.params['experience_memory_capacity']))  # Initialize an Experience memory with 1M capacity
        else:
            """ Normal Memory replay """
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
                # NOTE: This is just an additional check. 
                # The env wrappers are taking care of this conversion already
                # Reshape to C x H x W format as per PyTorch's convention
                observation = observation.reshape(observation.shape[2], observation.shape[1], observation.shape[0])
            observation = np.expand_dims(observation, 0)  # Create a batch dimension
        return self.policy(observation)

    """
    EPSILON_GREEDY_Q: exploit/explore

    :param observation:  
    saves scalars
    chooses to exploit or explore randomly
    """
    def epsilon_greedy_Q(self, observation):
        self.step_num += 1
        # Decay Epsilon/exploration as per schedule
        if self.ep_num % 10 == 0:
            writer.add_scalar("DQL/epsilon", self.epsilon_decay(self.step_num), self.step_num)
        if random.random() < self.epsilon_decay(self.step_num) and not self.params["test"]:
            action = random.choice([i for i in range(self.action_shape)])
        else:
            action = np.argmax(self.Q(observation).data.to(torch.device(device)).numpy())
        return action

    """
    LEARN_FROM_BATCH_EXPERIENCE

    :param experiences: tuple, contains batches of experiences 
    :param b_idx:
    :param is_weights:
    """
    def learn_from_batch_experience(self, experiences, b_idx=None, is_weights=None):
        xp_batch = Experience(*zip(*experiences))
        obs_batch = np.array(xp_batch.obs) / 255.0  # Scale/Divide by max limit of obs's dtype. 255 for uint8
        next_obs_batch = np.array(xp_batch.next_obs) / 255.0  # Scale/Divide by max limit of obs' dtype. 255 for uint8
        action_batch = np.array(xp_batch.action)
        reward_batch = np.array(xp_batch.reward)
        done_batch = np.array(xp_batch.done)
        gamma_arr = np.tile(self.gamma, len(next_obs_batch))
        # Clip the rewards
        if self.params["clip_rewards"]:
            reward_batch = np.sign(reward_batch)
        
        if self.params['use_target_network']:
            #if self.training_steps_completed % self.params['target_network_update_freq'] == 0:
            if self.step_num % self.params['target_network_update_freq'] == 100:
                self.Q_target.load_state_dict(self.Q.state_dict())

            T_target = reward_batch + ~done_batch * gamma_arr * self.Q_target(next_obs_batch).max(1)[0].data.cpu().numpy()
        else:
            T_target = reward_batch + ~done_batch * gamma_arr * self.Q(next_obs_batch).detach().max(1)[0].data.cpu().numpy()

        T_target = torch.from_numpy(T_target).to(device)
        action_idx = torch.from_numpy(action_batch).to(device)

        self.Q_optimizer.zero_grad()
        """ IF prio_mem Update priority """
        if self.prioritized_memory:
            errors = torch.abs(self.Q(obs_batch).gather(1, action_idx.view(-1, 1).type(torch.LongTensor)) - T_target.float().unsqueeze(1))
            self.memory.batch_update(b_idx, errors)
            loss = (F.mse_loss(self.Q(obs_batch).gather(1, action_idx.view(-1, 1).type(torch.LongTensor)), T_target.float().unsqueeze(1)) * torch.FloatTensor(is_weights)).mean()
            loss.backward()
        else:
            loss = F.mse_loss(self.Q(obs_batch).gather(1, action_idx.view(-1, 1).type(torch.LongTensor)), T_target.float().unsqueeze(1))
            loss.mean().backward()
        if self.ep_num % 10 == 0:
            writer.add_scalar("DQL/td_error", loss.mean(), self.step_num)
        self.Q_optimizer.step()

    """
    REPLAY FROM EXPERIENCE

    :param batch_size: default None/from params
    """
    def replay_experience(self, batch_size = None):
        batch_size = batch_size if batch_size is not None else self.params['replay_batch_size']
        if self.prioritized_memory:
            experience_batch, b_idx, is_weights = self.memory.sample(batch_size)
            self.learn_from_batch_experience(experience_batch, b_idx, is_weights)
        else:
            """   Regular Random memory   """
            experience_batch = self.memory.sample(batch_size)
            self.learn_from_batch_experience(experience_batch)
        self.training_steps_completed += 1 # Increment the number of training batch steps complemented

    """
    SAVE

    :param env_name: to create name
    """
    def save(self, env_name):    
        file_name = self.params['save_dir'] + "DQL_" + env_name + ".ptm"
        """ HERE SAVE EPSILON AND STEP COUNT AND REASSIGN IN LOAD """
        agent_state = {
            "Q": self.Q.state_dict(),
            "ep_num": self.ep_num,
            "step_num": self.step_num,
            "best_mean_reward": self.best_mean_reward,
            "best_reward": self.best_reward
        }
        torch.save(agent_state, file_name)
        print("Agent's state saved to ", file_name)

    """
    LOAD

    :param env_name: to load from name
    """
    def load(self, env_name):
        file_name = self.params['load_dir'] + "DQL_" + env_name + ".ptm"
        agent_state = torch.load(file_name, map_location= lambda storage, loc: storage)
        self.Q.load_state_dict(agent_state["Q"])
        self.Q.to(device)
        if 'ep_num' in agent_state.keys():
            self.ep_num = agent_state["ep_num"]
            self.step_num = agent_state["step_num"]
        self.best_mean_reward = agent_state["best_mean_reward"]
        self.best_reward = agent_state["best_reward"]
        print(f"--- Loading frome checkpoint ---")
        print(f"Loaded Q model state from ", file_name,
              " with step count of: ", self.step_num,
              " which fetched a best mean reward of: ", self.best_mean_reward,
              " and an all time best reward of: ", self.best_reward)


    """
    TRAIN
    all what happens inside train loop

    :param done: Bool
    :param obs: Observation matrix
    :param cum_reward: cumalitive reward in episode
    :param step: episode step count
    :param render: Bool
    """
    def train(self, done, obs, cum_reward, step, render):
        while not done:
            if render:
                env.render()
            action = self.get_action(obs)
            next_obs, reward, done, info = env.step(action)

            """
            STORING MEMORY:
                this stores all memories, but i think non trivial memories
                should be stored only 

                repeat rewarding / punishing memories 
                VS
                add less trivial memories (chosen tactic)
            """
            if reward != 0 or done is True:
                self.memory.store(Experience(obs, action, reward, next_obs, done))
            else:
                if random.random() > 0.7:
                    self.memory.store(Experience(obs, action, reward, next_obs, done))
            """ THIS IS CORRECT FORM BUT I TRAINED EARLY WITH 0.7 thing so put this back when done with same hypers
            
            if self.prioritized_memory:
                self.memory.store(Experience(obs, action, reward, next_obs, done))
            else:
                if reward != 0 or done is True:
                    self.memory.store(Experience(obs, action, reward, next_obs, done))
                else:
                    if random.random() > 0.7:
                        self.memory.store(Experience(obs, action, reward, next_obs, done))
            """
            obs = next_obs
            cum_reward += reward
            step += 1

            if done is True:
                self.ep_num +=1
                self.calc_stats(cum_reward)
                # Print Statement
                if self.ep_num % 1 == 0:
                    self.print_stats(cum_reward, step)
                    self.add_scalar(cum_reward)
                # Learn from batches of experience once a certain amount of xp is available
                # print(f"mem size = {agent.memory.get_size()}")
                if self.memory.get_size() >= 2 * self.params['replay_start_size']:
                    """
                    We want to batchsize proportional to game length
                    to ensure new rewrites dont errase potentially good game knowledge
                    and add extra batchsize to replay adjusted important memories

                    loss.mean().backward() is called more often making it more accurate
                    THIS WILL WORK WITH 2^n only and migth break otherwise
                    """
                    step = step // self.params["replay_batch_size"]
                    batch_size_n = step * self.params["replay_batch_size"] + self.params["replay_batch_size"]
                    for x in range(batch_size_n // 64):
                        """ 256 / (256 / 64) = 256 / 4 = 64 """
                        self.replay_experience(self.params["replay_batch_size"] // (self.params["replay_batch_size"] // 64))
                break

    """
    TEST

    :param done: Bool
    :param obs: Observation matrix
    :param cum_reward: cumalitive reward in episode
    :param step: episode step count
    :param render: Bool
    """
    def test(self, done, obs, cum_reward, step, render):
        game_log = {"steps": {}}
        while not done:
            if render:
                env.render()
            action = self.get_action(obs)
            next_obs, reward, done, info = env.step(action)
            game_log["steps"][str(step)] = {
                "selected-action-id": int(action),
                "selected-action": self.action_dict[int(action)],
                "reward": float(reward)
            }
            # add Q value for selected action
            # add Q values for all actions  
            Q_val = self.get_Qvalues(obs)
            game_log["steps"][str(step)]["Q_val"] = Q_val

            obs = next_obs
            cum_reward += reward
            step += 1

            if done is True:
                self.ep_num +=1
                self.calc_stats(cum_reward)

                game_log["episode"] = self.ep_num
                game_log["step_count"] = step
                game_log["total_reward"] = int(cum_reward)
                """   Write To Json file   """
                with open('./trained_models/results/' + 'openaigym' + str(self.ep_num) + '.json', 'w', encoding='utf-8') as f:
                    json.dump(game_log, f, ensure_ascii=False, indent=4)
                
                if self.ep_num % 1 == 0:
                    self.print_stats(cum_reward, step)
                break
    
    """
    GET QVALUES

    :param obs: observation
    """
    def get_Qvalues(self, obs):
        obs = np.array(obs) / 255.0
        obs = np.expand_dims(obs, axis=0)
        Q_val = self.Q(obs, test=True).data.to(torch.device(device)).numpy()
        return Q_val.tolist()
    
    """
    CALC STATS
    calculates all stats and calls save if model improved n times

    :param cum_reward: cumalitive reward in episode
    """
    def calc_stats(self, cum_reward):
        if len(self.episode_rewards) <= self.reward_length:
            self.episode_rewards.append(cum_reward)
        else:
            self.episode_rewards[self.ep_num % self.reward_length] = cum_reward
        
        if cum_reward > self.best_reward:
            self.best_reward = cum_reward
        if np.mean(self.episode_rewards) > self.prev_checkpoint_mean_ep_rew:
            self.num_improved_episodes_before_checkpoint += 1
        if self.num_improved_episodes_before_checkpoint >= self.params["save_freq_when_perf_improves"]:
            self.prev_checkpoint_mean_ep_rew = np.mean(self.episode_rewards)
            self.best_mean_reward = np.mean(self.episode_rewards)
            self.save(env_conf['env_name'])
            self.num_improved_episodes_before_checkpoint = 0

    """
    PRINT STATS

    :param step: episode step count
    :param cum_reward: cumalitive reward in episode
    """
    def print_stats(self, cum_reward, step):
        print(f"[{(self.step_num / self.max_training_steps)*100:.2f}%]", end="\t")
        print(f"Episode# {self.ep_num}\t ended in {step+1}\t steps.", end="\t")
        print(f"[training step #{self.training_steps_completed}]", end="\t")
        print(f"Per {self.rew_type} stats: [reward = {cum_reward},   ", end="\t")
        print(f"mean={np.mean(self.episode_rewards):.3f}: best={self.best_reward}]")
    
    """
    ADD SCALAR

    :param cum_reward: cumalitive reward in episode
    """
    def add_scalar(self, cum_reward):
        writer.add_scalar("main/ep_reward", cum_reward, self.step_num)
        writer.add_scalar("main/mean_ep_reward", np.mean(self.episode_rewards), self.step_num)
        writer.add_scalar("main/max_ep_rew", self.best_reward, self.step_num)

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

    # If a custom useful_region configuration for this environment ID is available, use it if not use the Default
    custom_region_available = False
    for key, value in env_conf['useful_region'].items():
        if key in args.env:
            env_conf['useful_region'] = value
            custom_region_available = True
            break
    if custom_region_available is not True:
        env_conf['useful_region'] = env_conf['useful_region']['Default']

    print("Using env_conf:", json.dumps(env_conf, indent = 1))
    """   Create atari env   """
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

    """   Returns every episodes video   """
    def return_every_ep(episode_id):
        return True

    if args.record:  # If monitor is enabled, record stats and video of agent's performance
        env = gym.wrappers.Monitor(env, args.recording_output_dir, force=True, video_callable=return_every_ep)

    observation_shape = env.observation_space.shape
    action_shape = env.action_space.n
    
    action_dict = {}
    action_names = env.get_action_meanings()
    for id, _action in enumerate(action_names):
        action_dict[id] = _action
    agent_params = params_manager.get_agent_params()
    agent_params["test"] = args.test
    agent_params["recording-output-dir"] = args.recording_output_dir
    agent = Deep_Q_Learner(observation_shape, action_shape, agent_params)
    agent.action_dict = action_dict

    print("Using agent_params:", json.dumps(agent_params, indent = 1))
    """   LOADING   """
    if agent.params['load_trained_model']:
        try:
            agent.load(env_conf["env_name"])
            prev_checkpoint_mean_ep_rew = agent.best_mean_reward
        except FileNotFoundError:
            prev_checkpoint_mean_ep_rew = agent.best_mean_reward
            print("WARNING: No trained model found for this environment. Training from scratch.")
    """   IF PARAMS SET TO TEST THEN RESET    """
    if agent.params["test"]:
        agent.ep_num = 0
        agent.step_num = 0

    while agent.step_num <= agent.max_training_steps:
        obs = env.reset()
        cum_reward = 0.0  # Cumulative reward
        done = False
        step = 0
        render = args.render or env_conf['render']
        """   TRAIN or TEST   """
        if not agent.params["test"]:
            agent.train(done, obs, cum_reward, step, render)
        else:
            agent.test(done, obs, cum_reward, step, render)
            if agent.ep_num == 10:
                break
    env.close()
    writer.close()
