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
    id='MassPointTraj-v0',
    entry_point='custom_gym.classic_control:MassPointTrajEnv',
    max_episode_steps=200,
)

register(
    id='MassPointGEP-v0',
    entry_point='custom_gym.classic_control:MassPointGEPEnv',
    max_episode_steps=200,
)

register(
    id='FiveTarget-v0',
    entry_point='custom_gym.classic_control:FiveTargetEnv',
    max_episode_steps=200,
)

register(
    id='FiveTargetEnv-v1',
    entry_point='custom_gym.classic_control:FiveTargetEnv_v1',
    max_episode_steps=200,
)

register(
    id='FiveTarget-v2',
    entry_point='custom_gym.classic_control:FiveTargetEnv_v2',
    max_episode_steps=200,
)

register(
    id='FiveTargetColor-v0',
    entry_point='custom_gym.classic_control:FiveTargetColorEnv',
    max_episode_steps=200,
)

register(
    id='FiveTargetColor-v1',
    entry_point='custom_gym.classic_control:FiveTargetColorV1Env',
    max_episode_steps=200,
)

register(
    id='FiveTargetRandColor-v0',
    entry_point='custom_gym.classic_control:FiveTargetRandColorEnv',
    max_episode_steps=200,
)

register(
    id='FiveTargetRandColor-v2',
    entry_point='custom_gym.classic_control:FiveTargetRandColorEnv_v2',
    max_episode_steps=200,
)

register(
    id = 'OverCooked-v0',
    entry_point='custom_gym.classic_control:OverCookedEnv',
    max_episode_steps=200,
)

# unity
register(
    id='Kobuki-v0',
    entry_point='custom_gym.unity:KobukiEnv',
    
)

# mujoco

register(
    id='ReacherEx-v0',
    entry_point='custom_gym.mujoco:ReacherEnv',
    max_episode_steps=50,
    reward_threshold=-3.75,
)

register(
    id='ReacherGEP-v0',
    entry_point='custom_gym.mujoco:ReacherGEPEnv',
    max_episode_steps=20,
    reward_threshold=-3.75,
)

register(
    id='ReacherGEPTest-v0',
    entry_point='custom_gym.mujoco:ReacherGEPTestEnv',
    max_episode_steps=20,
    reward_threshold=-3.75,
)

register(
    id='ReacherGEPTestTry-v0',
    entry_point='custom_gym.mujoco:ReacherGEPTestTryEnv',
    max_episode_steps=20,
    reward_threshold=-3.75,
)

register(
    id='ReacherGEPTraj-v0',
    entry_point='custom_gym.mujoco:ReacherGEPTrajEnv',
    max_episode_steps=20,
)

register(
    id='ReacherGEPTrajTest-v0',
    entry_point='custom_gym.mujoco:ReacherGEPTrajTestEnv',
    max_episode_steps=20,
)

register(
    id='ReacherFiveTarget-v0',
    entry_point='custom_gym.mujoco:ReacherFiveTargetEnv',
    max_episode_steps = 50,
    reward_threshold=-3.75,
)

register(
    id='ReacherFiveTarget-v1',
    entry_point='custom_gym.mujoco:ReacherFiveTargetEnv_v1',
    max_episode_steps = 50,
    reward_threshold=-3.75,
)

register(
    id='ReacherOverCooked-v0',
    entry_point='custom_gym.mujoco:ReacherOverCookedEnv',
    max_episode_steps = 50,
    reward_threshold=-3.75,
)

register(
    id='ReacherOverCooked-v1',
    entry_point='custom_gym.mujoco:ReacherOverCookedEnv_v1',
    max_episode_steps = 50,
    reward_threshold=-3.75,
)
