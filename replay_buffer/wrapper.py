import numpy as np
from typing import Callable, Dict, List, Optional


class UnionReplayBuffer:
    def __init__(
        self,
        buffers: List,
        buffer_ratios: Optional[List[Callable]]=None,
        ) -> None: 
        self._rbs = buffers
        self._rb_ratios = buffer_ratios
        
    @property
    def buffer_ratio(self):
        if self._rb_ratios is None:
            num_transitions = np.array([len(rb) for rb in self._rbs])
        else:
            num_transitions = np.array([rb_ratio(rb) for rb, rb_ratio in zip(self._rbs, self._rb_ratios)])
        assert sum(num_transitions) > 0, 'All replay buffers are empty!'
        return num_transitions/np.sum(num_transitions)

    def sample(
        self,
        batch_size: int,
        n_step: int=1,
        frame_stack: int=1
        ) -> Dict[str,np.ndarray]:
        batch_sizes = np.random.multinomial(batch_size, self.buffer_ratio)
        batches = [rb.sample(n, n_step=n_step, frame_stack=frame_stack) for rb, n in zip(self._rbs, batch_sizes)]

        assert np.all([batches[0].keys() == batch.keys() for batch in batches]), 'replay buffer batch keys do not match'
        ret = dict()
        for key in batches[0].keys():
            assert [batches[0][key].dtype == batch[key].dtype for batch in batches], 'replay buffer key dtypes do not match'
            ret[key] = np.concatenate([batch[key] for batch in batches], axis=0)

        return ret