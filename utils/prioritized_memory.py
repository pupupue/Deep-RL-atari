import random
import numpy as np
from collections import namedtuple
from utils.sumtree import SumTree

Experience = namedtuple("Experience", ['obs', 'action', 'reward', 'next_obs', 'done'])

"""   Prioritized Memory   """
class PrioritizedMemory():
    def __init__(self, alpha=0.4, beta=0.6, e=0.01, capacity=1e6):
        """
        :param capacity: Total capacity (Max number of Experiences) default: 10^6
        :return:
        """
        self.tree = SumTree(capacity)
        self.clip = 1. # clipped abs error
        self.alpha = alpha
        self.beta = beta
        self.beta_inc = 0.000025
        self.e = e
    
    # save sample (error,<s,a,r,s'>) to the replay memory
    def store(self, experience):
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        """   check for if priority = 0   """
        if max_priority == 0:
            max_priority = self.clip # error
        experience = self.pack_memory(experience) # <s,a,r,s'>
        self.tree.add(max_priority, experience)
        
    """
    SAMPLE:
    :param batch_size: int32 size of batch to get from memory
    uniformly samples from range in sumtree

    :return b_idx: batch of indexes referencing memory location
    :return batch: batch of Experiences()
    """
    def sample(self, batch_size):
        """
        :param batch_size:  Sample batch_size
        :return: A list of batch_size number of Experiences sampled with prio
        """
        assert batch_size <= self.tree.capacity, "Sample batch_size is more than available exp in mem"

        batch = []
        b_idx = []
        segment = self.tree.total() / batch_size
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_inc])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get_leaf(s)
            priorities.append(p)
            batch.append(data)
            b_idx.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, b_idx, is_weight
    
    """ UPDATING PRIORITY BY BATCH INDEXES """
    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors.detach().numpy(), self.clip)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
    
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
        len(next_obs) # ITS GONNA BREAK IF ANY __FORCE IS NOT CALLED 
        obs._out = obs._out.astype(np.uint8)
        next_obs._out = next_obs._out.astype(np.uint8)
        experience = Experience(obs, action, reward, next_obs, done)
        return experience

    def unpack_memory(self, experience):
        obs, action, reward, next_obs, done = experience
        _out, _next_obs = obs._out.astype(np.float32), next_obs._out.astype(np.float32)
        obs._out, next_obs._out = _out, _next_obs
        experience = Experience(obs, action, reward, next_obs, done)
        return experience
        
    def get_size(self):
        """
        :return: Number of Experiences stored in the memory
        """
        return self.tree.n_entries