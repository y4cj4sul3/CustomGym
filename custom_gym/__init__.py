from gym.envs.registration import register

register(
    id='MountainCarEx-v0',
    entry_point='custom_gym.classic_control:MountainCarEnv',
    max_episode_steps=200,
    reward_threshold=-110.0,
)

register(
    id='Kobuki-v0',
    entry_point='custom_gym.unity:KobukiEnv',
    
)
