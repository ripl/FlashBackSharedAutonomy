import warnings,sys,os

from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import numpy as np
import torch as th
from gymnasium import spaces
from ..utils.env_utils import get_obs_shape, get_action_dim, get_device, ReplayBufferSamples,GoalReplayBufferSamples
from pathlib import Path
from datetime import datetime
from torch.utils.data import IterableDataset, DataLoader,Dataset
import os 
import pickle
import torch 

def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    else:
        tensor = np.array(tensor)
    return tensor 

class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    observation_space: spaces.Space
    obs_shape: Tuple[int, ...]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space,n_env=n_envs)  # type: ignore[assignment]
        #! TODO for n_envs, maybe we should use single_env action_space  ? 
        self.action_dim = get_action_dim(action_space,n_env=n_envs)# <--- ugly bug fixing

        self.pos = 0
        self.full = False
        self.device = get_device(device)
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int, env= None):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    @abstractmethod
    def _get_samples(
        self, batch_inds: np.ndarray, env =  None
    ):
        """
        :param batch_inds:
        :param env:
        :return:
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return th.tensor(array, device=self.device)
        return th.as_tensor(array, device=self.device)

    def to_torch_flat(self, array: np.ndarray, copy: bool = True, start_dim = 0, end_dim = 1) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return torch.flatten(th.tensor(array, device=self.device), start_dim=start_dim, end_dim = end_dim)
        return torch.flatten(th.as_tensor(array, device=self.device), start_dim=start_dim, end_dim = end_dim)

    @staticmethod
    def _normalize_obs(
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        env = None,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if env is not None:
            return env.normalize_obs(obs)
        return obs

    @staticmethod
    def _normalize_reward(reward: np.ndarray, env = None) -> np.ndarray:
        if env is not None:
            return env.normalize_reward(reward).astype(np.float32)
        return reward
    
class ReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    timeouts: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        goal_dim: int = 0,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)


        # there is a bug if both optimize_memory_usage and handle_timeout_termination are true
        # see https://github.com/DLR-RM/stable-baselines3/issues/934
        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)
        self.goal_dim = goal_dim    
        self.goals = np.zeros((self.buffer_size, self.n_envs, self.goal_dim), dtype=observation_space.dtype)

        if not optimize_memory_usage:
            # When optimizing memory, `observations` contains also the next observation
            self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: Dict[str, Any],
        truncation:np.ndarray = None
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        ## Handle if not np.ndarray
        

        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))
        if self.goal_dim > 0 and 'goal' in infos:
            goal = infos['goal']
            goal = goal.reshape((self.n_envs, self.goal_dim))
            self.goals[self.pos] = to_numpy(goal)

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        

        # Copy to avoid modification by reference
        self.observations[self.pos] = to_numpy(obs)

        # THIS IS THE BUG
        # if self.optimize_memory_usage:
        #     self.observations[(self.pos + 1) % self.buffer_size] = to_numpy(obs)
        # else:
        #     self.next_observations[self.pos] = to_numpy(obs)
        
        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = to_numpy(next_obs)
        else:
            self.next_observations[self.pos] = to_numpy(next_obs) 

        self.actions[self.pos] = to_numpy(action)
        self.rewards[self.pos] = to_numpy(reward)
        self.dones[self.pos] = to_numpy(done)

        if self.handle_timeout_termination:
            # self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])
            self.timeouts[self.pos] = to_numpy(truncation) if truncation is not None else np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env = None):
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        assert batch_size > self.n_envs, (batch_size, self.n_envs)
        n_batch_size = batch_size // self.n_envs
        if not self.optimize_memory_usage:
            return super().sample(batch_size=n_batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=n_batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=n_batch_size)
        samples = self._get_samples(batch_inds, env=env) ## should be in shape (bz, n_envs, dim)
        return samples

    def _get_samples(self, batch_inds: np.ndarray, env = None):
        # Sample randomly the env idx
        
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, :, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, :, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, :, :], env),
            self.actions[batch_inds, :, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, :] * (1 - self.timeouts[batch_inds, :])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, :].reshape(-1, 1), env),
        )
        if self.goal_dim > 0:
            data += (self.goals[batch_inds, :, :],)
            return GoalReplayBufferSamples(*tuple(map(self.to_torch_flat, data)))
        return ReplayBufferSamples(*tuple(map(self.to_torch_flat, data)))

    @staticmethod
    def _maybe_cast_dtype(dtype: np.typing.DTypeLike) -> np.typing.DTypeLike:
        """
        Cast `np.float64` action datatype to `np.float32`,
        keep the others dtype unchanged.
        See GH#1572 for more information.

        :param dtype: The original action space dtype
        :return: ``np.float32`` if the dtype was float64,
            the original dtype otherwise.
        """
        if dtype == np.float64:
            return np.float32
        return dtype
    
class RolloutBuffer:
            
    def __init__(self,num_envs,num_transitions_per_env, obs_shape, act_shape, goal_dim, device):
        self.device = device 
        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        self.goals = torch.zeros(num_transitions_per_env, num_envs, goal_dim, device=self.device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *act_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, device=self.device).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, device=self.device)

        self.returns = torch.zeros(num_transitions_per_env, num_envs, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, device=self.device)
        self.final_values = torch.zeros(num_transitions_per_env, num_envs, device=self.device)
       
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs 
        self.step = 0 
    def add_transitions(self,obs,action,goal,reward,done,logprob,value):
        self.observations[self.step] = obs 
        self.actions[self.step] = action 
        self.goals[self.step] = goal 
        self.rewards[self.step] = reward 
        self.dones[self.step] = done 
        self.actions_log_prob[self.step] = logprob 
        self.values[self.step] = value.flatten(-1)

        self.step += 1 
    
    def add_final_values(self,step, env_idx, values):
        self.final_values[step,env_idx] = values[env_idx]
    
    def compute_returns(self, last_values, last_dones,  gamma, lam):
        advantage = []
        lastgaelam = 0

        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
                next_is_not_terminal = 1.0 - last_dones.float()
            else:
                next_values = self.values[step + 1] ### if terminate at step, the v[step+1] is not next value
                next_is_not_terminal = 1.0 - self.dones[step + 1].float()

            real_next_values = next_is_not_terminal * next_values + (1 - next_is_not_terminal) * self.final_values[step]
            delta = self.rewards[step] +  gamma * real_next_values - self.values[step]
            advantage.append( delta + next_is_not_terminal * gamma * lam * lastgaelam)
            lastgaelam = advantage[-1]
        # Compute and normalize the advantages
        self.advantages = torch.stack(list(reversed(advantage)))
        self.returns = self.advantages + self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def mini_batch_genetor(self,num_mini_batch, num_epochs):
        self.step = 0 
        batch_size = self.num_envs * self.num_transitions_per_env 
        mini_batch_size = batch_size // num_mini_batch 
        indices = torch.randperm(num_mini_batch * mini_batch_size, requires_grad = False, device = self.device )

        try:
            observations = self.observations.flatten(0, 1)
            goals = self.goals.flatten(0,1)
            actions = self.actions.flatten(0, 1)
            values = self.values.flatten(0, 1)
            returns = self.returns.flatten(0, 1)
            old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
            advantages = self.advantages.flatten(0, 1)
            batches = []
            for i in range(num_mini_batch):
                start = i * mini_batch_size
                end = (i+1) * mini_batch_size 
                batch_idx = indices[start:end] 
                obs_batch = observations[batch_idx].detach()
                goal_batch = goals[batch_idx].detach()
                act_batch = actions[batch_idx].detach()
                val_batch = values[batch_idx].detach()
                ret_batch = returns[batch_idx].detach()
                logp_batch = old_actions_log_prob[batch_idx].detach()
                adv_batch = advantages[batch_idx].detach()
                batches.append((obs_batch,act_batch,goal_batch,val_batch,ret_batch,logp_batch,logp_batch,adv_batch))

            for epoch in range(num_epochs):
                for batch in batches:
                    yield batch
                    
        finally:
            torch.cuda.empty_cache()
            
        

class DataCollection(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    timeouts: np.ndarray

    def __init__(
        self,
        directory: str,
        chunk_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        n_envs: int = 1,
        exp_name: str = 'exp',
        goal_dim = None,
        non_goal_dim = None 
    ):
        super().__init__(chunk_size, observation_space, action_space, 'auto', n_envs=n_envs)
        os.makedirs(directory, exist_ok=True)
        self.directory = Path(directory)
        # Adjust buffer size
        self.buffer_size = chunk_size
        if non_goal_dim is not None:
            ## for some env 
            self.obs_shape = (non_goal_dim,)

        self.observations = np.zeros((self.buffer_size, np.prod(self.obs_shape)), dtype=observation_space.dtype)
        self.next_observations = np.zeros((self.buffer_size, np.prod(self.obs_shape)), dtype=observation_space.dtype)
        self.actions = np.zeros(
            (self.buffer_size, self.action_dim)
        )
        self.goal_dim = goal_dim
        self.goals = np.zeros((self.buffer_size, 1), dtype=np.float32) if goal_dim is None else np.zeros((self.buffer_size, goal_dim), dtype=np.float32)
        self.total_size = 0
        self._file_cache = {}
        self.exp_name = exp_name

        # If there are files in the directory, lets read all files first
        for fname in self.directory.iterdir():
            if fname.suffix == ".pkl" and fname.is_file():
                print(f"loading {fname} from {self.directory}...")
                buff = th.load(fname)
                self._file_cache[fname] = buff
                self.total_size += buff.shape[0]
        

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        next_obs:np.ndarray = None,
        goal: np.ndarray = None
    ) -> None:
        if goal is None:
            goal = np.zeros_like((1,),dtype=np.float32)
        if next_obs is None:
            next_obs = np.zeros_like(obs) 
        self.goals[self.pos] = np.array(goal) 
        self.observations[self.pos] = np.array(obs)
        self.next_observations[self.pos] = np.array(next_obs)
        self.actions[self.pos] = np.array(action)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.dump_buffer()
            self.pos = 0

    def sample(self):
        import random

        assert len(self._file_cache) > 0
        fname = random.choice(list(self._file_cache.keys()))
        buff = self._file_cache[fname]

        # Sample a random idx
        idx = random.randint(0, buff.shape[0] - 1)
        return buff[idx]
    

    def dump_buffer(self, run_id=None):
        # Create a folder based on the current date and experiment name if not already created
        if not hasattr(self, 'dump_dir'):
            current_time = datetime.now().strftime("%Y%m%d-%H%M")
            folder_name = f"{self.exp_name}_{current_time}"
            self.dump_dir = Path(self.directory) / folder_name
            self.dump_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a readable file name using the run_id if provided, or just use a counter
        if run_id is None:
            run_id = len(list(self.dump_dir.glob('*.pkl'))) + 1
        
        file_name = self.dump_dir / f"buffer_run_{run_id:03d}.pkl"
        
        # Concatenate the buffer data and dump it
        _current_buffer = np.concatenate([self.observations, self.actions,self.next_observations, self.goals], axis=1)
        
        print(f"Dumping buffer to {file_name}")
        th.save(_current_buffer, file_name)
    
    def _get_samples(self, batch_inds: np.ndarray, env = None):
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))
    

class ExpertTransitionDataset(IterableDataset):
    def __init__(self,
                 directory,
                 state_dim,
                 action_dim,
                 ) -> None:
        super().__init__()
        print(f"Loading dataset from {directory}")
        def print_directory_structure(search_directory):
            for root, dirs, files in os.walk(search_directory):
                print(f"Directory: {root}")
                for file in files:
                    print(f"  File: {file}")
        print_directory_structure(directory)
        
        self.directory = Path(directory)
        self.state_dim = state_dim 
        self.act_dim = action_dim 

        self.total_size = 0
        self._file_cache = {}

        # If there are files in the directory, lets read all files first
        for fname in self.directory.iterdir():
            print(f"loading {fname} from {self.directory}...")
            buff = th.load(fname)
            self._file_cache[fname] = buff
            self.total_size += buff.shape[0]
        self.total_file = np.concatenate(list(self._file_cache.values()), axis=0)
        print("Total size of the dataset: ", self.total_size)
    
    def __iter__(self):
        for fname in self._file_cache:
            buff = self._file_cache[fname]
            for sample in buff:
                state,act = sample[:self.state_dim], sample[self.state_dim:self.state_dim+self.act_dim]
                yield state,act

    def __len__(self):
        return self.total_size
    
    def get_simple_sample(self):
        import random
        assert len(self._file_cache) > 0
        fname = random.choice(list(self._file_cache.keys()))
        buff = self._file_cache[fname]
        # Sample a random idx
        idx = random.randint(0, buff.shape[0] - 1)
        sample = buff[idx]
        state,act = sample[:self.state_dim], sample[self.state_dim:self.state_dim+self.act_dim]
        return state,act

class ExpertTransitionDataset_v2(Dataset):
    def __init__(self,
                 directory,
                 state_dim,
                 action_dim,
                 use_transition = True,
                 add_noise = False
                 ) -> None:
        super().__init__()
        print(f"Loading dataset from {directory}")
        def print_directory_structure(search_directory):
            for root, dirs, files in os.walk(search_directory):
                print(f"Directory: {root}")
                for file in files:
                    print(f"  File: {file}")
        print_directory_structure(directory)
        
        self.directory = Path(directory)
        self.state_dim = state_dim 
        self.act_dim = action_dim 
        self.use_transition = use_transition

        self.total_size = 0
        self._file_cache = {}
        self.add_noise = add_noise

        # If there are files in the directory, lets read all files first
        for fname in self.directory.iterdir():
            print(f"loading {fname} from {self.directory}...")
            buff = th.load(fname)
            print(type(buff))
            if not isinstance(buff, th.Tensor):
                buff = th.tensor(buff)
            self._file_cache[fname] = buff
            self.total_size += buff.shape[0]
        self.total_file = th.cat(list(self._file_cache.values()), axis=0)
        print("Total size of the dataset: ", self.total_size)

        self.obs = self.total_file[:,0:self.state_dim]
        self.act = self.total_file[:,self.state_dim:self.state_dim+self.act_dim]
        self.obs2 = self.total_file[:,self.state_dim+self.act_dim:2*self.state_dim+self.act_dim]

        ## keep some statistics for noise 
        self.obs_std = torch.std(self.obs, dim=0)
        self.act_std = torch.std(self.act, dim=0) 
        self.obs2_std = torch.std(self.obs2, dim=0) 

        
        self.obs_noise_vec = self.obs_std * 0.1
        self.act_noise_vec = self.act_std * 0.1
        self.obs2_noise_vec = self.obs2_std * 0.1 
        obs_min_noise = torch.ones(self.state_dim) * 1e-4
        act_min_noise = torch.ones(self.act_dim) * 1e-4

        self.obs_noise_vec = torch.max(self.obs_noise_vec, obs_min_noise)
        self.act_noise_vec = torch.max(self.act_noise_vec, act_min_noise)
        self.obs2_noise_vec = torch.max(self.obs2_noise_vec, obs_min_noise) 
        
    
    def load_data(self,directory):
        for fname in self.directory.iterdir():
            print(f"loading {fname} from {self.directory}...")
            buff = th.load(fname)
            if not isinstance(buff, th.Tensor):
                buff = th.tensor(buff)
            self._file_cache[fname] = buff
            self.total_size += buff.shape[0]
        
        self.total_file = th.cat(list(self._file_cache.values()), axis=0)
        print("Total size of the dataset: ", self.total_size)
        self.obs = self.total_file[:,0:self.state_dim]
        self.act = self.total_file[:,self.state_dim:self.state_dim+self.act_dim]
        self.obs2 = self.total_file[:,self.state_dim+self.act_dim:2*self.state_dim+self.act_dim]
    
    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        # Return data and label at the specified index
        if self.use_transition:
            obs = self.obs[idx]
            act = self.act[idx]
            obs2 = self.obs2[idx]
            if self.add_noise:
                obs = obs + torch.randn_like(obs) * self.obs_noise_vec
                # act = act + torch.randn_like(act) * self.act_noise_vec
                obs2 = obs2 + torch.randn_like(obs2) * self.obs2_noise_vec
            return obs,act,obs2
        else:
            obs = self.obs[idx]
            act = self.act[idx]
            if self.add_noise:
                obs = obs + torch.randn_like(obs) * self.obs_noise_vec
                # act = act + torch.randn_like(act) * self.act_noise_vec
            return obs,act 


class UR5_DataPreprocess:
    # class-level attribute
    state_keys = ['joint_states', 'gripper_pos', 'gripper_quat']
    action_keys_with_grasp_pos = ['cmd_trans_vel', 'cmd_rot_vel', 'cmd_grasp_pos']
    action_keys_without_grasp_pos = ['cmd_trans_vel', 'cmd_rot_vel']
    data_dims = {
        'joint_states': 6,
        'gripper_pos': 3,
        'gripper_quat': 4,
        'cmd_trans_vel': 3,
        'cmd_rot_vel': 3,
        'cmd_grasp_pos': 1
    }
    def __init__(self):
        pass 
    @classmethod
    def _list2dict(cls,list_dict):
        ## prepare the dictionary
        tmp_dict = dict()
        for key in cls.state_keys:
            tmp_dict[key] = []
        for key in cls.action_keys_with_grasp_pos:
            tmp_dict[key] = []
        ## traverse the list of dictionary
        for i,d in enumerate(list_dict):
            red_flag = False ## don't know why there could be some missing data 
            # TODO: There is a bug, since we remove 1 tuple, than this can not be used to form transition
            # FIXME: Need to fix this bug 
            for key in tmp_dict.keys():
                v = d[key]
                if type(v) == int or type(v) == float:
                    d[key] = np.array([v]).reshape(1,)
                else:
                    d[key] = np.array(v)

                if d[key].shape[0] != cls.data_dims[key]:
                    red_flag = True
            if not red_flag:
                for key in tmp_dict.keys():
                    tmp_dict[key].append(d[key]) 
            else:
                ## Simple interpolation for d['joint_states]
                if i > 0 and i < len(list_dict)-1:
                    red_flag = False
                    d['joint_states'] = (np.array(list_dict[i-1]['joint_states']) + np.array(list_dict[i+1]['joint_states']))/2.0
                    for key in tmp_dict.keys():
                        tmp_dict[key].append(d[key])
                else:
                    print(f"Red Flag: {i}")
                  
                
        for k,v in tmp_dict.items():
            if len(v) == 0:
                return False, None
            
        for key in tmp_dict.keys():
            tmp_dict[key] = np.stack(tmp_dict[key], axis=0)
        
        ## Check the dimension and shape
        n_sample = tmp_dict['joint_states'].shape[0]
        for key in tmp_dict.keys():
            assert tmp_dict[key].shape[1] == cls.data_dims[key]
            assert tmp_dict[key].shape[0] == n_sample
        
        return True, tmp_dict

    """Important functions for data extraction
        1. define the dimension of the ur5 data (state_dim, action_dim)
    """
    @classmethod
    def _ur5_rawdata2transition(cls,data_dict, act_include_grasp_pos = False, remove_start = 0, remove_end = 1):
        ## state = (joint_states, gripper_pos, gripper_quat)
        ## action = (cmd_trans_vel, cmd_rot_vel), if act_include_grasp_pos = True, then add cmd_grasp_pos 

        state = np.concatenate([data_dict[k] for k in cls.state_keys], axis=-1)
        if not act_include_grasp_pos:
            action = np.concatenate([data_dict[k] for k in cls.action_keys_without_grasp_pos], axis=-1)
        else:
            action = np.concatenate([data_dict[k] for k in cls.action_keys_with_grasp_pos], axis=-1)
        ## Prepare transition
        cur_state = state[:-1]
        cur_action = action[:-1]
        next_state = state[1:]   
        transition_dict = {'obs': cur_state, 'act': cur_action, 'obs2': next_state}
        for key in transition_dict.keys():
            transition_dict[key] = transition_dict[key][remove_start:-remove_end]

        return transition_dict
    
    @classmethod
    def _dump_dict(cls,src_dict, dst_dict):
        for key in src_dict.keys():
            dst_dict[key].append(src_dict[key]) ### list of np.array with different size
        return dst_dict
    
    @classmethod
    def ur5_data_extract(cls,list_file_path,save_dir, new_name = 'ur5_data', remove_start = 0, remove_end = 1):
        """
        Input: [path1, path2, ...]
        in each path, there are multiple .pt files, use pickle to load them and concatenate them
        """
        
        data = {
            'obs': [],
            'act': [],
            'obs2': []
        }
        for path in list_file_path:
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith(".pt"):
                        file_path = os.path.join(root, file)
                        with open(file_path, 'rb') as f:
                            tmp_list = pickle.load(f) ## list of dictionary
                            if len(tmp_list) == 0:    ## empty file
                                continue
                            success,tmp_dict = cls._list2dict(tmp_list) 
                            if success:
                                transition_dict = cls._ur5_rawdata2transition(tmp_dict, act_include_grasp_pos = False, remove_start = remove_start, remove_end = remove_end)
                                data = cls._dump_dict(transition_dict, data)
                            else:
                                print(f"Failed to load data from {file_path}")

        for k,v in data.items():
            data[k] = np.concatenate(v, axis=0)
            print("Load Data: ", k, data[k].shape)
        os.makedirs(save_dir, exist_ok=True)
        file_name = os.path.join(save_dir, f"{new_name}.pkl")
        # Concatenate the buffer data and dump it
        _current_buffer = np.concatenate([data['obs'], data['act'],data['obs2']], axis=-1)
        print(f"Dumping buffer to {file_name}")
        torch.save(_current_buffer, file_name)