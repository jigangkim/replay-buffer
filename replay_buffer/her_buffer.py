import copy
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple

from .replay_buffer import ReplayBuffer


class HindsightReplayBuffer(object):
    def __init__(
        self,
        max_number_of_transitions: int,
        compute_reward,
        compute_done: Optional[Callable]=None,
        n_sampled_goal: int=4,
        goal_selection_strategy: str='future',
        online_sampling: bool=True,
        mem_option: str='static',
        name: str=''
        ) -> None:
        # input args
        self.name = name
        self.capacity = int(max_number_of_transitions)
        self._mem_option = mem_option
        # input args (hindsight)
        self._compute_reward = compute_reward
        self._compute_done = compute_done
        self.n_sampled_goal = int(n_sampled_goal)
        self.her_ratio = 1 - 1/(self.n_sampled_goal + 1)
        self.goal_selection_strategy = goal_selection_strategy
        self._online_sampling = online_sampling
        if not self._online_sampling:
            self._mem_option = 'offline'
            self._replay_buffer = ReplayBuffer(
                max_number_of_transitions*(self.n_sampled_goal + 1),
                mem_option=mem_option,
                name=name+'._replay_buffer'
            )

        # buffer
        self.num_transitions = 0
        self.index = 0
        self.data = None
        self.item_keys = None
        self.episode_stats = {}

    def __getitem__(self, index):
        if self._online_sampling:
            return {key: d[index] for key, d in zip(self.item_keys, self.data)}
        else:
            return self._replay_buffer[index]

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

        # update episode_stats
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
            if not self._online_sampling and len(self.episode_stats) == 1: # previous episode has ended for offline sampling
                self._store_her_offline()
                self.num_transitions = 0
                self.index = 0
                self.episode_stats = {}
            self.episode_stats[episode_id] = (self.index, self.index)

        # add/overwrite transition
        for key, d in zip(self.item_keys, self.data):
            d[self.index] = copy.deepcopy(transition[key])
        self.info_buffer[self.index] = copy.deepcopy(transition.pop('info'))

        # update num_transitions and index
        self.num_transitions = min(self.num_transitions + 1, self.capacity)
        self.index = (self.index + 1) % self.capacity

        if not self._online_sampling:
            self._replay_buffer.store(transition)
            assert len(self.episode_stats) == 1, 'No more than one episode must be stored for offline mode'
 
    def sample(
        self,
        batch_size: int,
        n_step: int=1,
        frame_stack: int=1
        ) -> Dict[str,np.ndarray]:
        '''
        Randomly sample batch(es) of transition(s) from buffer.

        params:
            :param batch_size: Size of batch
            :param n_step: Sample transitions n-steps apart
        returns:
            :return *: Batch of transition(s) of size batch_size
        '''
        assert self.num_transitions > 0

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

        if self._online_sampling:
            her_batch_size = sum(np.random.uniform(size=batch_size) < self.her_ratio)
            real_batch_size = batch_size - her_batch_size
            if her_batch_size > 0:
                real_indices, n_step_ = _get_indices(real_batch_size, n_step)
                real_data = [d[real_indices] for d in self.data]
                sampled_indices, new_goal_indices = self._sample_her_indices(her_batch_size, online_sampling=True, n_step=n_step)
                her_data = self._generate_her_data(sampled_indices, new_goal_indices)
                combined_data = [np.concatenate([d1, d2], axis=0) for d1, d2 in zip(her_data, real_data)]
                ret = {key+'s': d for key, d in zip(self.item_keys, combined_data)}
                for i in range(n_step_ - 1):
                    real_data = [d[(real_indices+i+1)%self.capacity] for d in self.data]
                    her_data = self._generate_her_data((sampled_indices+i+1)%self.capacity, new_goal_indices)
                    combined_data = [np.concatenate([d1, d2], axis=0) for d1, d2 in zip(her_data, real_data)]
                    ret.update({key+'s_%d'%(i+1): d for key, d in zip(self.item_keys, combined_data)})
            else: # standard replay buffer (no hindsight)
                real_indices, n_step_ = _get_indices(real_batch_size, n_step)
                ret = {key+'s': d[real_indices] for key, d in zip(self.item_keys, self.data)}
                for i in range(n_step_ - 1):
                    ret.update({key+'s_%d'%(i+1): d[(real_indices+i+1)%self.capacity] for key, d in zip(self.item_keys, self.data)})

            if frame_stack > 1:
                
                def get_stacked_indices(sampled_indices, batch_size):
                    sampled_episodes = self.data[self.item_keys.index('episode')][sampled_indices].squeeze(axis=-1)
                    stacked_indices = (sampled_indices - np.arange(frame_stack, dtype=np.int32)[:, None]) % self.capacity
                    stacked_episodes = self.data[self.item_keys.index('episode')][stacked_indices].squeeze(axis=-1)
                    valid = np.logical_and(stacked_indices < self.num_transitions, 
                                        stacked_episodes == sampled_episodes)
                    min_indices = stacked_indices[np.sum(valid, axis=0) - 1, np.arange(batch_size)]
                    return np.where(valid, stacked_indices, min_indices).T
                
                ret_indices = np.concatenate([sampled_indices, real_indices]) if her_batch_size > 0 else real_indices
                stacked_indices = get_stacked_indices(ret_indices, batch_size)
                stacked_shape = (stacked_indices.shape[0],-1)+ret['observations'].shape[2:]
                ret['observations'] = self.data[self.item_keys.index('observation')][stacked_indices].reshape(stacked_shape) # (batch_size, stacked obs_dim)
                ret['next_observations'] = self.data[self.item_keys.index('next_observation')][stacked_indices].reshape(stacked_shape) # (batch_size, stacked obs_dim)
                for i in range(n_step_-1):
                    stacked_indices = get_stacked_indices((ret_indices+i+1)%self.capacity, batch_size)
                    stacked_shape = (stacked_indices.shape[0],-1)+ret['observations_%d'%(i+1)].shape[2:]
                    ret['observations_%d'%(i+1)] = self.data[self.item_keys.index('observation')][stacked_indices].reshape(stacked_shape)# (batch_size, stacked obs_dim)
                    ret['next_observations_%d'%(i+1)] = self.data[self.item_keys.index('next_observation')][stacked_indices].reshape(stacked_shape) # (batch_size, stacked obs_dim)
            return ret
        else:
            assert n_step == 1, 'n_step > 1 not suported for offline sampling'
            assert frame_stack == 1, 'frame_stack not supported for offline sampling'
            return self._replay_buffer.sample(batch_size)

    def clear(self) -> None:
        '''
        Reset buffer (does NOT free memory).
        '''
        self.num_transitions = 0
        self.index = 0
        self.episode_stats = {}
        if not self._online_sampling:
            self._replay_buffer.clear()

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
        transition_np = [np.atleast_1d(np.asarray(transition[key])) for key in self.item_keys]
        # check memory usage
        mem_usage = sum([x.nbytes for x in transition_np]) * self.capacity
        if mem_usage > 10737418240 and self._mem_option != 'offline':
            print('Memory usage may exceed 10GiB (name=%s)'%(self.name))
        # preallocate buffer
        if self._mem_option == 'offline':
            print('No more than one episode is stored for offline mode (name=%s).'%(self.name))
            self.data = [np.zeros(dtype=x.dtype, shape=(self.capacity,) + x.shape) for x in transition_np]
        elif self._mem_option == 'dynamic':
            print('Required free memory for replay buffer (name=%s): %.2f MiB'%(self.name, mem_usage/1024/1024))
            self.data = [np.zeros(dtype=x.dtype, shape=(self.capacity,) + x.shape) for x in transition_np]
        elif self._mem_option == 'static':
            print('Preallocating memory for replay buffer (name=%s): %.2f MiB'%(self.name, mem_usage/1024/1024))
            self.data = [np.ones(dtype=x.dtype, shape=(self.capacity,) + x.shape) for x in transition_np]
        else:
            raise ValueError('Unknown memory option: %s' % self._mem_option)
        # separate list for info data
        self.info_buffer = [None]*self.capacity

    def _sample_her_indices(
        self,
        batch_size: int,
        online_sampling: bool,
        n_sampled_goal: Optional[int]=None,
        n_step: int=1
        ) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Sample batch(es) of transition indices from buffer.

        params:
            :param batch_size: Size of batch
            :param online_sampling: Whether to re-label at sample time or at store time
            :param n_sampled_goal: Number of virtual transitions to create per real transition
            :param n_step: Sample transition indices n-steps apart
        returns:
            :return transition_indices: Sampled transition indices
            :return new_goal_indices: New goal indices for sampled transitions
        '''
        assert self.num_transitions > 0
        if online_sampling:
            assert batch_size is not None, 'Batch size must be specified for online sampling'
            if n_step > 1:
                episode_len = {k: (v[1] - v[0])%self.capacity + 1 for k, v in self.episode_stats.items()}
                episode_len_nstep = {k: v for k, v in episode_len.items() if v >= n_step}
                episode_ids = np.random.choice(np.array(list(episode_len_nstep.keys())), batch_size)
            else:
                episode_ids = np.random.choice(list(self.episode_stats.keys()), batch_size)
            ep_stats = np.array([self.episode_stats[idx] for idx in episode_ids])
            ep_begins, ep_ends = ep_stats[:,0], ep_stats[:,1]
            lowers = ep_begins
            uppers = (ep_ends - (n_step-1))%self.capacity
            ep_lengths = (uppers - lowers)%self.capacity + 1
            her_indices = np.arange(int(batch_size))
            episode_timesteps = np.random.randint(ep_lengths)
        else:
            assert n_sampled_goal is not None, 'Number of sampled goals must be specified for offline sampling'
            assert len(self.episode_stats) == 1, 'No more than one episode must be stored for offline mode'
            assert n_step == 1, 'n_step > 1 not suported for offline sampling'
            episode_id = list(self.episode_stats.keys())[0]
            ep_begin, ep_end = list(self.episode_stats.values())[0]
            episode_length = (ep_end - ep_begin)%self.capacity + 1
            episode_ids = np.array([episode_id]*episode_length*n_sampled_goal)
            ep_begins, ep_ends = np.ones_like(episode_ids)*ep_begin, np.ones_like(episode_ids)*ep_end
            ep_lengths = np.ones_like(episode_ids)*episode_length
            her_indices = np.arange(len(ep_lengths))
            episode_timesteps = np.tile(np.arange(episode_length), n_sampled_goal)
        
        transition_indices = (ep_begins + episode_timesteps)%self.capacity
        new_goal_indices = np.copy(transition_indices)
        if self.goal_selection_strategy == 'final':
            new_goal_indices[her_indices] = ep_ends[her_indices]
        elif self.goal_selection_strategy == 'future':
            ep_future_lengths = (ep_ends[her_indices] - transition_indices[her_indices])%self.capacity + 1
            new_goal_indices[her_indices] = (transition_indices[her_indices] + np.random.randint(ep_future_lengths))%self.capacity
        elif self.goal_selection_strategy == 'episode':
            new_goal_indices[her_indices] = (ep_begins[her_indices] + np.random.randint(ep_lengths[her_indices]))%self.capacity
        elif self.goal_selection_strategy == 'current': # set current states as the relabeled goal
            new_goal_indices[her_indices] = transition_indices[her_indices]
        else:
            raise ValueError(f'Unknown goal selection strategy: {self.goal_selection_strategy}')

        return transition_indices, new_goal_indices

    def _generate_her_data(
        self,
        transition_indices: np.ndarray,
        new_goal_indices: np.ndarray
        ) -> List[np.ndarray]:
        '''
        Generate virtual transition data with new desired goals.

        params:
            :param transition_indices: Indices of transitions to generate virtual transitions for
            :param new_goal_indices: Indices of new desired goals
        returns:
            :return her_data: Virtual transition data
        '''
        her_data = [np.copy(d[transition_indices]) for d in self.data]
        
        achieved_goal_idx = self.item_keys.index('achieved_goal')
        desired_goal_idx = self.item_keys.index('desired_goal')
        reward_idx = self.item_keys.index('reward')
        next_achieved_goal_idx = self.item_keys.index('next_achieved_goal')
        next_desired_goal_idx = self.item_keys.index('next_desired_goal')
        done_idx = self.item_keys.index('done')

        # relabel goal (new desired goal)
        her_data[desired_goal_idx] = self.data[next_achieved_goal_idx][new_goal_indices]
        her_data[next_desired_goal_idx] = her_data[desired_goal_idx]
        # update reward
        her_data[reward_idx] = self._compute_reward(
            her_data[next_achieved_goal_idx],
            her_data[desired_goal_idx], # new desired goal
            [self.info_buffer[idx] for idx in transition_indices]
        ).reshape(her_data[reward_idx].shape)
        # update dones
        if self.relabel_done:
            her_data[done_idx] = self._compute_done(
                her_data[next_achieved_goal_idx],
                her_data[desired_goal_idx], # new desired goal
                [self.info_buffer[idx] for idx in transition_indices]
            ).reshape(her_data[done_idx].shape)
        else:
            her_data[done_idx] = np.where(np.expand_dims(transition_indices==new_goal_indices, axis=-1), her_data[done_idx], False) # done defaults to False for relabeled transitions

        return her_data

    def _store_her_offline(self) -> None:
        '''
        Re-label at store time for offline sampling.
        '''
        # sample goals to create virtual transitions
        sampled_indices, new_goal_indices = self._sample_her_indices(
            batch_size=None,
            online_sampling=False,
            n_sampled_goal=self.n_sampled_goal
        )

        if sampled_indices.size == 0 or new_goal_indices.size == 0: # standard replay buffer if n_sampled_goal is 0 (no hindsight)
            pass
        else:
            her_data = self._generate_her_data(sampled_indices, new_goal_indices)
            # store virtual transitions in replay buffer
            for i in range(len(sampled_indices)):
                transition = {key: d[i] for key, d in zip(self.item_keys, her_data)}
                self._replay_buffer.store(transition)

    def set_n_sampled_goal(self, n_sampled_goal: int=4) -> None:
        self.n_sampled_goal = int(n_sampled_goal)
        self.her_ratio = 1 - 1/(self.n_sampled_goal + 1)