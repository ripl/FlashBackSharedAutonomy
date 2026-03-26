from .custom_Lunarlander import LunarLander_v4
import gymnasium as gym 
# from .PegInsertion import SimplePegInsertion
# from .ChargerPlug import SimpleChargerPlug
gym.register(
    id='LunarLander-v4',
    entry_point='diffusha.envs:LunarLander_v4',
    max_episode_steps=1000,
)
