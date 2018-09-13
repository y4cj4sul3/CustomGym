from custom_gym.utils.recoder import Recoder
from gym.envs.registration import register

# classic control
register(
    id='MountainCarEx-v0',
    entry_point='custom_gym.classic_control:MountainCarEnv',
    max_episode_steps=200,
    reward_threshold=-110.0,
)

register(
    id='MassPoint-v0',
    entry_point='custom_gym.classic_control:MassPointEnv',
    max_episode_steps=200,

)

register(
    id='FiveTarget-v0',
    entry_point='custom_gym.classic_control:FiveTargetEnv',
    max_episode_steps=200,
)

register(
    id='FiveTargetColor-v0',
    entry_point='custom_gym.classic_control:FiveTargetColorEnv',
    max_episode_steps=200,
)

# unity
register(
    id='Kobuki-v0',
    entry_point='custom_gym.unity:KobukiEnv',
    
)
