#!/usr/bin/env python
import sys
import random
import numpy as np
from collections import namedtuple

Experience = namedtuple("Experience", ['obs', 'action', 'reward', 'next_obs', 'done'])


class ExperienceMemory():
    """
    A cyclic/ring buffer based Experience Memory implementation
    """
    def __init__(self, capacity=int(1e6)):
        """
        :param capacity: Total capacity (Max number of Experiences) default: 10^6
        :return:
        """
        self.capacity = capacity
        self.mem_idx = 0  # Index of the current experience
        self.memory = []

    def store(self, experience):
        """
        :param experience: The Experience object to be stored into the memory
        :return:
        """
        self.pack_memory(experience)
        if self.mem_idx < self.capacity:
            # Extend the memory and create space
            self.memory.append(None)
        self.memory[self.mem_idx % self.capacity] = experience
        self.mem_idx += 1
    
    """
    :param experience: The Experience object to be optimized
    both obs can be uint8'ed
    done is bool
    reward is somehow more effic
    action is -3 bytes
    """
    def pack_memory(self, experience):
        obs, action, reward, next_obs, done = experience
        action = np.uint8(action)
        # ITS GONNA BREAK IF ANY __FORCE IS NOT CALLED 
        len(next_obs)
        obs._out = obs._out.astype(np.uint8)
        next_obs._out = next_obs._out.astype(np.uint8)
        experience = Experience(obs, action, reward, next_obs, done)
        return experience

    def unpack_memories(self, experience_list):
        return_list = []
        for index, experience in enumerate(experience_list):
            obs, action, reward, next_obs, done = experience
            _out, _next_obs = obs._out.astype(np.float32), next_obs._out.astype(np.float32)
            obs._out, next_obs._out = _out, _next_obs
            return_list.append(Experience(obs, action, reward, next_obs, done))
        return return_list

    def sample(self, batch_size):
        """
        :param batch_size:  Sample batch_size
        :return: A list of batch_size number of Experiences sampled at random from mem
        """
        assert batch_size <= len(self.memory), "Sample batch_size is more than available exp in mem"
        sample = random.sample(self.memory, batch_size)
        sample = self.unpack_memories(sample)
        return sample

    def get_size(self):
        """
        :return: Number of Experiences stored in the memory
        """
        return len(self.memory)
