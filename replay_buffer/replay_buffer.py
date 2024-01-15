import copy
import numpy as np
from typing import Dict, Optional


class ReplayBuffer(object):
    '''
    Generic replay buffer.
    '''
    def __init__(
        self,
        max_number_of_transitions: int,
        frame_stack: int=1,
        mem_option: str='static',
        name: str=''
        ) -> None:
        # input args
        self.name = name
        self.capacity = int(max_number_of_transitions)
        self.frame_stack = frame_stack
        self._mem_option = mem_option

        # buffer
        self.num_transitions = 0
        self.index = 0
        self.data = None
        self.item_keys = None
        self.episode_stats = {}
    
    def __getitem__(self, index):
        return {key: d[index] for key, d in zip(self.item_keys, self.data)}

    def __setitem__(self, key, value):
        raise TypeError('Insertion forbidden!')
    
    def __len__(self):
        return self.num_transitions

    def store(
        self,
        transition: Dict
        ) -> None:
        '''
        Store transition.

        params:
            :param transition: Transition to store
        returns:
        '''
        # allocating memory (run only once)
        if self.data is None:
            self._preallocate(transition)
        
        # if episode id is provided
        if 'episode' in self.item_keys:
            if self.num_transitions == self.capacity: # consider overwrite
                overwritten_episode_id = self.data[self.item_keys.index('episode')][self.index].tolist()[0]
                s_overwritten, e_overwritten = self.episode_stats[overwritten_episode_id]
                if s_overwritten == e_overwritten:
                    self.episode_stats.pop(overwritten_episode_id)
                else:
                    self.episode_stats[overwritten_episode_id] = ((self.index + 1) % self.capacity, e_overwritten)
            episode_id = int(transition['episode'])
            if episode_id in self.episode_stats:
                s, _ = self.episode_stats[episode_id]
                self.episode_stats[episode_id] = (s, self.index)
            else:
                self.episode_stats[episode_id] = (self.index, self.index)

        # add/overwrite transition
        for key, d in zip(self.item_keys, self.data):
            d[self.index] = copy.deepcopy(transition[key])
        
        # update num_transitions and index
        self.num_transitions = min(self.num_transitions + 1, self.capacity)
        self.index = (self.index + 1) % self.capacity

    def sample(
        self,
        batch_size: int,
        n_step: int=1,
        frame_stack: Optional[int]=None
        ) -> Dict:
        '''
        Randomly sample batch(es) of transition(s) from buffer.

        params:
            :param batch_size: Size of batch
            :param n_step: Sample transitions n-steps apart
        returns:
            :return *: Batch(es) of transition(s) of size batch_size
        '''
        assert self.num_transitions > 0
        if self.frame_stack == 1:
            assert frame_stack is None or frame_stack == 1, 'Cannot change frame_stack if it was set to 1 at initialization!'
        num_stack = self.frame_stack if frame_stack is None else frame_stack

        def _get_indices(batch_size, n_step):
            if n_step == 1:
                return np.random.choice(self.num_transitions, size=batch_size), 1
            else:
                assert 'episode' in self.item_keys
                episode_len = {k: (v[1] - v[0])%self.capacity + 1 for k, v in self.episode_stats.items()}
                episode_len_nstep = {k: v for k, v in episode_len.items() if v >= n_step}
                if len(episode_len_nstep) > 0:
                    episode_ids = np.array(list(episode_len_nstep.keys()))
                    sampled_ids = np.random.choice(episode_ids, size=batch_size, p=None)
                    d = self.episode_stats
                    lowers = np.array([d[idx][0] for idx in sampled_ids])
                    uppers = np.array([d[idx][1] if d[idx][1] >= d[idx][0] else d[idx][1] + self.capacity for idx in sampled_ids]) - (n_step-1)
                    return np.random.randint(lowers, uppers+1)%self.capacity, n_step
                else:
                    raise RuntimeError('No episodes with length >= n_step found!')
        
        sampled_indices, n_step_ = _get_indices(batch_size, n_step)

        ret = {key+'s': d[sampled_indices] for key, d in zip(self.item_keys, self.data)}
        ret['sampled_indices'] = sampled_indices
        for i in range(n_step_-1):
            ret.update({key+'s_%d'%(i+1): d[(sampled_indices+i+1)%self.capacity] for key, d in zip(self.item_keys, self.data)})
        if num_stack > 1:

            def get_stacked_indices(sampled_indices, batch_size):
                sampled_episodes = self.data[self.item_keys.index('episode')][sampled_indices].squeeze(axis=-1)
                stacked_indices = (sampled_indices - np.arange(num_stack, dtype=np.int32)[:, None]) % self.capacity
                stacked_episodes = self.data[self.item_keys.index('episode')][stacked_indices].squeeze(axis=-1)
                valid = np.logical_and(stacked_indices < self.num_transitions, 
                                       stacked_episodes == sampled_episodes)
                min_indices = stacked_indices[np.sum(valid, axis=0) - 1, np.arange(batch_size)]
                return np.where(valid, stacked_indices, min_indices).T
            
            stacked_indices = get_stacked_indices(sampled_indices, batch_size)
            stacked_shape = (stacked_indices.shape[0],-1)+ret['observations'].shape[2:]
            ret['observations'] = self.data[self.item_keys.index('observation')][stacked_indices].reshape(stacked_shape) # (batch_size, stacked obs_dim)
            ret['next_observations'] = self.data[self.item_keys.index('next_observation')][stacked_indices].reshape(stacked_shape) # (batch_size, stacked obs_dim)
            for i in range(n_step_-1):
                stacked_indices = get_stacked_indices((sampled_indices+i+1)%self.capacity, batch_size)
                stacked_shape = (stacked_indices.shape[0],-1)+ret['observations_%d'%(i+1)].shape[2:]
                ret['observations_%d'%(i+1)] = self.data[self.item_keys.index('observation')][stacked_indices].reshape(stacked_shape)# (batch_size, stacked obs_dim)
                ret['next_observations_%d'%(i+1)] = self.data[self.item_keys.index('next_observation')][stacked_indices].reshape(stacked_shape) # (batch_size, stacked obs_dim)
        return ret

    def clear(self) -> None:
        '''
        Reset buffer (does NOT free memory).
        '''
        self.num_transitions = 0
        self.index = 0
        self.episode_stats = {}

    def _preallocate(
        self, 
        transition
        ) -> None:
        '''
        Preallocate memory for buffer.
        '''
        self.item_keys = list(transition.keys())
        if self.frame_stack > 1:
            assert 'episode' in self.item_keys, 'Required key for transition "episode" not found'
        transition_np = [np.atleast_1d(np.asarray(transition[key])) for key in self.item_keys]
        # check memory usage
        mem_usage = sum([x.nbytes for x in transition_np]) * self.capacity
        if mem_usage > 10737418240:
            print('Memory usage may exceed 10GiB (name=%s)'%(self.name))
        # preallocate buffer
        if self._mem_option == 'dynamic':
            print('Required free memory for replay buffer (name=%s): %.2f MiB'%(self.name, mem_usage/1024/1024))
            self.data = [np.zeros(dtype=x.dtype, shape=(self.capacity,) + x.shape) for x in transition_np]
        elif self._mem_option == 'static':
            print('Preallocating memory for replay buffer (name=%s): %.2f MiB'%(self.name, mem_usage/1024/1024))
            self.data = [np.ones(dtype=x.dtype, shape=(self.capacity,) + x.shape) for x in transition_np]
        else:
            raise ValueError('Unknown memory option: %s' % self._mem_option)