import copy
import math
import numpy as np
from typing import Callable, Dict, Optional

from .replay_buffer import ReplayBuffer
from .her_buffer import HindsightReplayBuffer


class LazyReplayBufferOld(ReplayBuffer):
    '''
    Lazy replay buffer that require fixed episode lengths.

    params:
        :param max_number_of_transitions: Maximum number of transitions to keep
        :param episode_len: Episode length
        :param frame_stack: Number of frames to stack
        :param mem_option: Memory option (static, dynamic)
        :param name: Name for identification (optional)
    returns:
    '''
    def __init__(
        self,
        max_number_of_transitions: int,
        episode_len: int,
        frame_stack: int=1,
        mem_option: str='static',
        name: str=''
        ) -> None:
        super(LazyReplayBufferOld, self).__init__(max_number_of_transitions, frame_stack, mem_option, name)
        self.episode_len = episode_len
        assert max_number_of_transitions%episode_len == 0, 'buffer capacity must be a multiple of episode length!'
    
    def __getitem__(self, index):
        return TypeError('Indexing forbidden!')
    
    def sample(
        self,
        batch_size: int,
        frame_stack: Optional[int]=None
        ) -> Dict:
        '''
        Randomly sample batch(es) of transition(s) from buffer.

        params:
            :param batch_size: Size of batch
            :param frame_stack: Number of frames to stack
        returns:
            :return *: Batch(es) of transition(s) of size batch_size
        '''
        assert self.num_transitions > 0
        if self.frame_stack == 1:
            assert frame_stack is None or frame_stack == 1, 'Cannot change frame_stack if it was set to 1 at initialization!'
        num_stack = self.frame_stack if frame_stack is None else frame_stack

        if self.num_transitions < self.capacity:
            sampled_indices = np.random.choice(self.num_transitions, size=batch_size)
        else:
            sampled_indices = (self.index + 1 + np.random.choice(self.capacity - 1, size=batch_size)) % self.capacity
        
        ret = {key+'s': d[sampled_indices] for key, d in zip(self.item_keys, self.data)}
        ret['sampled_indices'] = sampled_indices
        if num_stack > 1:
            sampled_episodes = self.data[self.item_keys.index('episode')][sampled_indices].squeeze(axis=-1)
            stacked_indices = (sampled_indices - np.arange(num_stack, dtype=np.int32)[:, None]) % self.capacity
            stacked_episodes = self.data[self.item_keys.index('episode')][stacked_indices].squeeze(axis=-1)
            
            valid1 = stacked_indices < self.num_transitions if self.num_transitions < self.capacity else stacked_indices != self.index # valid index range
            valid2 = stacked_episodes == sampled_episodes # must be from the same episode
            valid = np.logical_and(valid1, valid2)
            min_indices = stacked_indices[np.sum(valid, axis=0) - 1, np.arange(batch_size)]

            stacked_indices = np.where(valid, stacked_indices, min_indices).T
            stacked_shape = (stacked_indices.shape[0],-1)+ret['observations'].shape[2:]
            ret['observations'] = self.data[self.item_keys.index('observation')][stacked_indices].reshape(stacked_shape) # (batch_size, stacked obs_dim)
            ret['next_observations'] = self.data[self.item_keys.index('next_observation')][stacked_indices].reshape(stacked_shape) # (batch_size, stacked obs_dim)
        
        return ret

    def _preallocate(
        self, 
        transition
        ) -> None:
        '''
        Preallocate memory for buffer.
        '''
        self.item_keys = list(transition.keys())
        required_keys = [
            'observation',
            'next_observation',
            'episode',
        ]
        assert all([key in self.item_keys for key in required_keys]), f'Required keys for transition not found: {required_keys}'
        self.item_keys.remove('observation')
        self.item_keys.remove('next_observation')
        self.item_keys.append('observation')
        self.item_keys.append('next_observation')

        keys = list(transition.keys())
        keys.remove('observation')
        keys.remove('next_observation')
        transition_np = [np.atleast_1d(np.asarray(transition[key])) for key in keys]
        mem_usage = sum([x.nbytes for x in transition_np]) * self.capacity
        lazy_obs_np = np.atleast_1d(np.asarray(transition['observation']))
        mem_usage += lazy_obs_np.nbytes * self.capacity * (1 + 1/self.episode_len)
        # check memory usage
        if mem_usage > 10737418240:
            print('Memory usage may exceed 10GiB (name=%s)'%(self.name))
        
        class IndexMappedArray(object):
            def __init__(self, ndarray, map_func):
                self._data = ndarray
                self._map = map_func
            def __getitem__(self, idx):
                return self._data[self._map(idx)]
            def __setitem__(self, idx, value):
                self._data[self._map(idx)] = value

        # preallocate buffer
        obs_idx_mapping = lambda idx: idx%self.episode_len + idx//self.episode_len*(self.episode_len + 1)
        next_obs_idx_mapping = lambda idx: idx%self.episode_len + idx//self.episode_len*(self.episode_len + 1) + 1
        lazy_capacity = math.ceil(self.capacity * (1 + 1/self.episode_len))
        if self._mem_option == 'dynamic':
            print('Required free memory for replay buffer (name=%s): %.2f MiB'%(self.name, mem_usage/1024/1024))
            self.lazy_data = np.zeros(dtype=lazy_obs_np.dtype, shape=(lazy_capacity,) + lazy_obs_np.shape)
            self.data = [np.zeros(dtype=x.dtype, shape=(self.capacity,) + x.shape) for x in transition_np]
            self.data.append(IndexMappedArray(self.lazy_data, obs_idx_mapping))
            self.data.append(IndexMappedArray(self.lazy_data, next_obs_idx_mapping))
        elif self._mem_option == 'static':
            print('Preallocating memory for replay buffer (name=%s): %.2f MiB'%(self.name, mem_usage/1024/1024))
            self.lazy_data = np.ones(dtype=lazy_obs_np.dtype, shape=(lazy_capacity,) + lazy_obs_np.shape)
            self.data = [np.ones(dtype=x.dtype, shape=(self.capacity,) + x.shape) for x in transition_np]
            self.data.append(IndexMappedArray(self.lazy_data, obs_idx_mapping))
            self.data.append(IndexMappedArray(self.lazy_data, next_obs_idx_mapping))
        else:
            raise ValueError('Unknown memory option: %s' % self._mem_option)


class LazyReplayBuffer(ReplayBuffer):
    '''
    Lazy replay buffer that supports variable episode lengths.
    This is achieved by maintaining the obs data as the lazy data and only storing the latest next obs in the aux data.

    params:
        :param max_number_of_transitions: Maximum number of transitions to keep
        :param episode_len: Rough hint for episode length
        :param frame_stack: Number of frames to stack
        :param mem_option: Memory option (static, dynamic)
        :param name: Name for identification (optional)
    returns:
    '''
    def __init__(
        self,
        max_number_of_transitions: int,
        episode_len: int=-1, # rough hint for episode length, always supports variable episode lengths
        frame_stack: int=1,
        mem_option: str='static',
        name: str=''
        ) -> None:
        super(LazyReplayBuffer, self).__init__(max_number_of_transitions, frame_stack, mem_option, name)
        if episode_len > 0:
            print(f'Episode length hinted as {episode_len}, taking it into account for memory allocation! (name={self.name}')
        else:
            print(f'Episode length unspecified! (name={self.name}')
        self.episode_len = episode_len
    
    def __getitem__(self, index):
        return TypeError('Indexing forbidden!')
    
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
        if self.data is None:
            self._preallocate(transition)

        # update episode stats and aux idx
        if self.num_transitions == self.capacity: # consider overwrite
            overwritten_episode_id = self.data[self.item_keys.index('episode')][self.index].tolist()[0]
            s_overwritten, e_overwritten = self.episode_stats[overwritten_episode_id]
            if s_overwritten == e_overwritten == self.index:
                self.episode_stats.pop(overwritten_episode_id)
                self.aux_inuse[self.aux_id[self.index]] = False
            else:
                self.episode_stats[overwritten_episode_id] = ((self.index + 1) % self.capacity, e_overwritten)
        episode_id = int(transition['episode'])
        existing_episode = episode_id in self.episode_stats
        if existing_episode:
            last_obs_pointer = self.aux_id[(self.index-1)%self.capacity]
            self.aux_id[(self.index-1)%self.capacity] = -1
            self.aux_id[self.index] = last_obs_pointer
            s, _ = self.episode_stats[episode_id]
            self.episode_stats[episode_id] = (s, self.index)
        else:
            last_obs_pointer = np.argmin(self.aux_inuse)
            assert not self.aux_inuse[last_obs_pointer], 'No room left for aux data!'
            self.aux_inuse[last_obs_pointer] = True
            self.aux_id[self.index] = last_obs_pointer
            self.episode_stats[episode_id] = (self.index, self.index)

        # add/overwrite transition
        for key, d in zip(self.item_keys, self.data[:-2]): # writable data
            d[self.index] = copy.deepcopy(transition[key])
        self.lazy_data[self.index] = copy.deepcopy(transition[self.item_keys[-2]]) # obs
        last_obs_pointer = self.aux_id[self.index]
        assert last_obs_pointer >= 0, 'Invalid pointer!'
        self.aux_data[last_obs_pointer] = copy.deepcopy(transition[self.item_keys[-1]]) # last obs of the episode

        # update num_transitions and index
        self.num_transitions = min(self.num_transitions + 1, self.capacity)
        self.index = (self.index + 1) % self.capacity

    def clear(self) -> None:
        '''
        Reset buffer (does NOT free memory).
        '''
        super(LazyReplayBuffer, self).clear()
        self.aux_id[:] = -1
        self.aux_inuse[:] = False

    def _preallocate(
        self, 
        transition
        ) -> None:
        '''
        Preallocate memory for buffer.
        '''
        self.item_keys = list(transition.keys())
        required_keys = [
            'observation',
            'next_observation',
            'episode',
        ]
        assert all([key in self.item_keys for key in required_keys]), f'Required keys for transition not found: {required_keys}'
        # let 'observation' and 'next_observation' be the last two items
        self.item_keys.remove('observation')
        self.item_keys.remove('next_observation')
        self.item_keys.append('observation')
        self.item_keys.append('next_observation')

        # check memory usage
        keys = list(transition.keys())
        keys.remove('observation')
        keys.remove('next_observation')
        transition_np = [np.atleast_1d(np.asarray(transition[key])) for key in keys]
        base_mem_usage = sum([x.nbytes for x in transition_np] + [8]) * self.capacity
        lazy_obs_np = np.atleast_1d(np.asarray(transition['observation']))
        aux_obs_ratio = 1/self.episode_len if self.episode_len > 0 else 0
        min_mem_usage = base_mem_usage + lazy_obs_np.nbytes * self.capacity * (1 + aux_obs_ratio)
        max_mem_usage = base_mem_usage + lazy_obs_np.nbytes * self.capacity * 2
        if max_mem_usage > 10737418240:
            min_mem_usage_in_gib = min_mem_usage/1024**3
            max_mem_usage_in_gib = max_mem_usage/1024**3
            print(f'Memory usage will exceed {min_mem_usage_in_gib:.2f} GiB in the best case, \
                  may exceed {max_mem_usage_in_gib:.2f} GiB in the worst case (name={self.name})')
        
        class IndexMappedArray(object): # read-only array
            def __init__(self, lazy_data, aux_data, map_func):
                self._data1 = lazy_data
                self._data2 = aux_data
                self._map = map_func
            def __getitem__(self, idx):
                arr_id, raw_idx = self._map(idx)
                return np.where(arr_id.reshape(arr_id.shape + (1,)*(self._data1.ndim - 1)), \
                                self._data2[raw_idx], self._data1[raw_idx])
            def __setitem__(self, idx, value):
                raise TypeError('Insertion forbidden!') # current/latest episode id required to determine raw_idx

        def obs_idx_mapping(idx): # read-only
            return np.zeros_like(idx), np.asarray(idx) # observation always comes from lazy data
        def next_obs_idx_mapping(idx): # read-only
            idx = np.asarray(idx)
            is_not_episode_end = self.aux_id[idx] == -1
            return np.where(is_not_episode_end, 0, 1), np.where(is_not_episode_end, (idx+1)%self.capacity, self.aux_id[idx])

        # preallocate buffer
        if self._mem_option == 'dynamic':
            print('Required minimum free memory for replay buffer (name=%s): %.2f MiB'%(self.name, min_mem_usage/1024/1024))
            self.lazy_data = np.zeros(dtype=lazy_obs_np.dtype, shape=(self.capacity,) + lazy_obs_np.shape) # store obs
            self.aux_data = np.zeros(dtype=lazy_obs_np.dtype, shape=(self.capacity,) + lazy_obs_np.shape) # store last obs of the episode
            self.aux_id = -np.ones(dtype=np.int64, shape=(self.capacity,))
            self.aux_inuse = np.zeros(dtype=bool, shape=(self.capacity,))
            self.data = [np.zeros(dtype=x.dtype, shape=(self.capacity,) + x.shape) for x in transition_np]
            self.data.append(IndexMappedArray(self.lazy_data, self.aux_data, obs_idx_mapping))
            self.data.append(IndexMappedArray(self.lazy_data, self.aux_data, next_obs_idx_mapping))
        elif self._mem_option == 'static': # not completely static, but preallocate as much as possible with episode length hint
            print('Preallocating minimum memory for replay buffer (name=%s): %.2f MiB'%(self.name, min_mem_usage/1024/1024))
            self.lazy_data = np.ones(dtype=lazy_obs_np.dtype, shape=(self.capacity,) + lazy_obs_np.shape)
            self.aux_data = np.zeros(dtype=lazy_obs_np.dtype, shape=(self.capacity,) + lazy_obs_np.shape) # variable memory with np.zeros
            self.aux_data[:math.ceil(self.capacity*aux_obs_ratio), ...] = 1 # preallocate with episode length hint (may use more memory)
            self.aux_id = -np.ones(dtype=np.int64, shape=(self.capacity,))
            self.aux_inuse = np.zeros(dtype=bool, shape=(self.capacity,))
            self.data = [np.ones(dtype=x.dtype, shape=(self.capacity,) + x.shape) for x in transition_np]
            self.data.append(IndexMappedArray(self.lazy_data, self.aux_data, obs_idx_mapping))
            self.data.append(IndexMappedArray(self.lazy_data, self.aux_data, next_obs_idx_mapping))
        else:
            raise ValueError('Unknown memory option: %s' % self._mem_option)
        

class LazyHindsightReplayBuffer(HindsightReplayBuffer):
    '''
    Lazy hindsight replay buffer that supports variable episode lengths.
    This is achieved by maintaining the obs data as the lazy data and only storing the latest next obs in the aux data.

    params:
        :param max_number_of_transitions: Maximum number of transitions to keep
        :param compute_reward: Goal-conditioned reward function
        :param compute_done: Goal-conditioned done function
        :param n_sampled_goal: Number of virtual transitions to create per real transition
        :param goal_selection_strategy: Goal selection strategy (final, future, episode)
        :param episode_len: Rough hint for episode length
        :param mem_option: Memory option (static, dynamic)
        :param name: Name for identification (optional)
    returns:
    '''
    def __init__(
        self,
        max_number_of_transitions: int,
        compute_reward,
        compute_done: Optional[Callable]=None,
        n_sampled_goal: int=4,
        goal_selection_strategy: str='future',
        episode_len: int=-1, # rough hint for episode length, always supports variable episode lengths
        mem_option: str='static',
        name: str=''
        ) -> None:
        super(LazyHindsightReplayBuffer, self).__init__(max_number_of_transitions, compute_reward,
                                                                compute_done=compute_done,
                                                                n_sampled_goal=n_sampled_goal,
                                                                goal_selection_strategy=goal_selection_strategy,
                                                                online_sampling=True,
                                                                mem_option=mem_option,
                                                                name=name)
        if episode_len > 0:
            print(f'Episode length hinted as {episode_len}, taking it into account for memory allocation! (name={self.name}')
        else:
            print(f'Episode length unspecified! (name={self.name}')
        self.episode_len = episode_len

    def __getitem__(self, index):
        return TypeError('Indexing forbidden!')
    
    def store(
        self,
        transition: Dict,
        ) -> None:
        '''
        Store transition.

        params:
            :param transition: Transition to store
        returns:
        '''
        if self.data is None:
            self._preallocate(transition)

        # update episode_stats and aux_id, aux_inuse
        if self.num_transitions == self.capacity: # consider overwrite
            overwritten_episode_id = self.data[self.item_keys.index('episode')][self.index].tolist()[0]
            s_overwritten, e_overwritten = self.episode_stats[overwritten_episode_id]
            if s_overwritten == e_overwritten == self.index:
                self.episode_stats.pop(overwritten_episode_id)
                self.aux_inuse[self.aux_id[self.index]] = False
            else:
                self.episode_stats[overwritten_episode_id] = ((self.index + 1) % self.capacity, e_overwritten)
        episode_id = int(transition['episode'])
        existing_episode = episode_id in self.episode_stats
        if existing_episode:
            last_obs_pointer = self.aux_id[(self.index-1)%self.capacity]
            self.aux_id[(self.index-1)%self.capacity] = -1
            self.aux_id[self.index] = last_obs_pointer
            s, _ = self.episode_stats[episode_id]
            self.episode_stats[episode_id] = (s, self.index)
        else:
            last_obs_pointer = np.argmin(self.aux_inuse)
            assert not self.aux_inuse[last_obs_pointer], 'No room left for aux data!'
            self.aux_inuse[last_obs_pointer] = True
            self.aux_id[self.index] = last_obs_pointer
            self.episode_stats[episode_id] = (self.index, self.index)

        # add/overwrite transition
        for key, d in zip(self.item_keys, self.data[:-2]): # writable data
            d[self.index] = copy.deepcopy(transition[key])
        self.lazy_data[self.index] = copy.deepcopy(transition[self.item_keys[-2]]) # obs
        last_obs_pointer = self.aux_id[self.index]
        assert last_obs_pointer >= 0, 'Invalid pointer!'
        self.aux_data[last_obs_pointer] = copy.deepcopy(transition[self.item_keys[-1]]) # last obs of the episode
        self.info_buffer[self.index] = copy.deepcopy(transition.pop('info'))

        # update num_transitions and index
        self.num_transitions = min(self.num_transitions + 1, self.capacity)
        self.index = (self.index + 1) % self.capacity

    def clear(self) -> None:
        '''
        Reset buffer (does NOT free memory).
        '''
        super(LazyHindsightReplayBuffer, self).clear()
        self.aux_id[:] = -1
        self.aux_inuse[:] = False
    
    def _preallocate(
        self,
        transition
        ) -> None:
        '''
        Preallocate memory for buffer.
        '''
        self.item_keys = list(transition.keys())
        required_keys = [
            'observation',
            'achieved_goal',
            'desired_goal',
            'action',
            'reward',
            'next_observation',
            'next_achieved_goal',
            'next_desired_goal',
            'done',
            'episode',
            'info'
        ]
        assert all([key in self.item_keys for key in required_keys]), f'Required keys for transition not found: {required_keys}'
        self.item_keys.remove('info') # info handled by a separate buffer
        # let 'observation' and 'next_observation' be the last two items
        self.item_keys.remove('observation')
        self.item_keys.remove('next_observation')
        self.item_keys.append('observation')
        self.item_keys.append('next_observation')

        # check memory usage
        keys = list(transition.keys())
        keys.remove('observation')
        keys.remove('next_observation')
        keys.remove('info')
        transition_np = [np.atleast_1d(np.asarray(transition[key])) for key in keys]
        base_mem_usage = sum([x.nbytes for x in transition_np] + [8]) * self.capacity
        lazy_obs_np = np.atleast_1d(np.asarray(transition['observation']))
        aux_obs_ratio = 1/self.episode_len if self.episode_len > 0 else 0
        min_mem_usage = base_mem_usage + lazy_obs_np.nbytes * self.capacity * (1 + aux_obs_ratio)
        max_mem_usage = base_mem_usage + lazy_obs_np.nbytes * self.capacity * 2
        if max_mem_usage > 10737418240:
            min_mem_usage_in_gib = min_mem_usage/1024**3
            max_mem_usage_in_gib = max_mem_usage/1024**3
            print(f'Memory usage will exceed {min_mem_usage_in_gib:.2f} GiB in the best case, \
                  may exceed {max_mem_usage_in_gib:.2f} GiB in the worst case (name={self.name})')
            
        class IndexMappedArray(object): # read-only array
            def __init__(self, lazy_data, aux_data, map_func):
                self._data1 = lazy_data
                self._data2 = aux_data
                self._map = map_func
            def __getitem__(self, idx):
                arr_id, raw_idx = self._map(idx)
                return np.where(arr_id.reshape(arr_id.shape + (1,)*(self._data1.ndim - 1)), \
                                self._data2[raw_idx], self._data1[raw_idx])
            def __setitem__(self, idx, value):
                raise TypeError('Insertion forbidden!') # current/latest episode id required to determine raw_idx

        def obs_idx_mapping(idx): # read-only
            return np.zeros_like(idx), np.asarray(idx) # observation always comes from lazy data
        def next_obs_idx_mapping(idx): # read-only
            idx = np.asarray(idx)
            is_not_episode_end = self.aux_id[idx] == -1
            return np.where(is_not_episode_end, 0, 1), np.where(is_not_episode_end, (idx+1)%self.capacity, self.aux_id[idx])

        # preallocate buffer
        if self._mem_option == 'dynamic':
            print('Required minimum free memory for replay buffer (name=%s): %.2f MiB'%(self.name, min_mem_usage/1024/1024))
            self.lazy_data = np.zeros(dtype=lazy_obs_np.dtype, shape=(self.capacity,) + lazy_obs_np.shape) # store obs
            self.aux_data = np.zeros(dtype=lazy_obs_np.dtype, shape=(self.capacity,) + lazy_obs_np.shape) # store last obs of the episode
            self.aux_id = -np.ones(dtype=np.int64, shape=(self.capacity,))
            self.aux_inuse = np.zeros(dtype=bool, shape=(self.capacity,))
            self.data = [np.zeros(dtype=x.dtype, shape=(self.capacity,) + x.shape) for x in transition_np]
            self.data.append(IndexMappedArray(self.lazy_data, self.aux_data, obs_idx_mapping))
            self.data.append(IndexMappedArray(self.lazy_data, self.aux_data, next_obs_idx_mapping))
        elif self._mem_option == 'static': # not completely static, but preallocate as much as possible with episode length hint
            print('Preallocating minimum memory for replay buffer (name=%s): %.2f MiB'%(self.name, min_mem_usage/1024/1024))
            self.lazy_data = np.ones(dtype=lazy_obs_np.dtype, shape=(self.capacity,) + lazy_obs_np.shape)
            self.aux_data = np.zeros(dtype=lazy_obs_np.dtype, shape=(self.capacity,) + lazy_obs_np.shape) # variable memory with np.zeros
            self.aux_data[:math.ceil(self.capacity*aux_obs_ratio), ...] = 1 # preallocate with episode length hint (may use more memory)
            self.aux_id = -np.ones(dtype=np.int64, shape=(self.capacity,))
            self.aux_inuse = np.zeros(dtype=bool, shape=(self.capacity,))
            self.data = [np.ones(dtype=x.dtype, shape=(self.capacity,) + x.shape) for x in transition_np]
            self.data.append(IndexMappedArray(self.lazy_data, self.aux_data, obs_idx_mapping))
            self.data.append(IndexMappedArray(self.lazy_data, self.aux_data, next_obs_idx_mapping))
        else:
            raise ValueError('Unknown memory option: %s' % self._mem_option)
        # separate list for info data
        self.info_buffer = [None]*self.capacity
    
    def _store_her_offline(self) -> None:
        raise TypeError('Offline sampling forbidden!')