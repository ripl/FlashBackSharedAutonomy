
import gymnasium as gym
import numpy as np

class DecomposeObservationWrapper(gym.Wrapper):
    def __init__(self, env):
        """
        Initialize the wrapper.
        
        Args:
            env: The base environment.
            goal_indices: A list of indices indicating the positions of the goal-related parts
                          in the observation vector.
        """
        super().__init__(env)

        all_dim = 0
        start_dim = 0
        end_dim = 0
        d = env.obs_space_dict
        for k,v in d.items():
            if k != 'goal_lidar':
                all_dim += v.shape[0]
            else:
                start_dim = all_dim
                end_dim = start_dim + v.shape[0]
                goal_indices = list(range(start_dim, end_dim))
                break

        self.goal_indices = np.array(goal_indices)
        self.goal_dim = len(goal_indices)
        self.non_goal_dim = env.observation_space.shape[0] - self.goal_dim

    def reset(self, **kwargs):
        """
        Reset the environment and return the initial observation and info.
        """
        next_obs, info = self.env.reset(**kwargs)
        info = self._decompose_observation(next_obs, info)
        return next_obs, info

    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: The action to take.
            
        Returns:
            observation: The next observation.
            reward: The reward received.
            done: Whether the episode is over.
            info: Additional information with decomposed observation parts.
        """
        next_obs, reward, cost, terminated, truncated, info = self.env.step(action)
        info = self._decompose_observation(next_obs, info)
        return next_obs, reward, cost, terminated, truncated, info

    def _decompose_observation(self, observation, info):
        """
        Decompose the observation into goal and non-goal parts.
        
        Args:
            observation: The original observation from the environment.
            info: The info dictionary from the environment.
            
        Returns:
            observation: The original observation (unchanged).
            info: Updated info dictionary with decomposed observation parts.
        """
        goal_part = observation[:,self.goal_indices]
        non_goal_part = np.delete(observation, self.goal_indices,axis = -1)
        info['goal'] = goal_part
        info['non_goal_obs'] = non_goal_part
        return info
