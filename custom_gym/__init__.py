from custom_gym.utils.recoder import Recoder
from custom_gym.utils import RecorderWrapper
from gym.envs.registration import register

# =================== robotics  ===================
register(
    id='FetchReach-v2',
    entry_point='custom_gym.robotics:FetchReachEnv',
    kwargs={'reward_type': 'sparse'},
    max_episode_steps=50,
)

register(
    id='FetchPush-v2',
    entry_point='custom_gym.robotics:FetchPushEnv',
    kwargs={'reward_type': 'sparse'},
    max_episode_steps=50,
)

register(
    id='FetchSlide-v2',
    entry_point='custom_gym.robotics:FetchSlideEnv',
    kwargs={'reward_type': 'sparse'},
    max_episode_steps=50,
)

register(
    id='FetchPickAndPlace-v2',
    entry_point='custom_gym.robotics:FetchPickAndPlaceEnv',
    kwargs={'reward_type': 'sparse'},
    max_episode_steps=50,
)

register(
    id='FetchReach-v3',
    entry_point='custom_gym.robotics:FetchReachEnv',
    kwargs={'reward_type': 'sparse', 'instr_space': 1},
    max_episode_steps=50,
)

register(
    id='FetchPush-v3',
    entry_point='custom_gym.robotics:FetchPushEnv',
    kwargs={'reward_type': 'sparse', 'instr_space': 1},
    max_episode_steps=50,
)

register(
    id='FetchSlide-v3',
    entry_point='custom_gym.robotics:FetchSlideEnv',
    kwargs={'reward_type': 'sparse', 'instr_space': 1},
    max_episode_steps=50,
)

register(
    id='FetchPickAndPlace-v3',
    entry_point='custom_gym.robotics:FetchPickAndPlaceEnv',
    kwargs={'reward_type': 'sparse', 'instr_space': 1},
    max_episode_steps=50,
)

register(
    id='FetchReach-v4',
    entry_point='custom_gym.robotics:FetchReachEnv',
    kwargs={'reward_type': 'sparse', 'act_space': 1},
    max_episode_steps=50,
)

register(
    id='FetchPush-v4',
    entry_point='custom_gym.robotics:FetchPushEnv',
    kwargs={'reward_type': 'sparse', 'act_space': 1},
    max_episode_steps=50,
)

register(
    id='FetchSlide-v4',
    entry_point='custom_gym.robotics:FetchSlideEnv',
    kwargs={'reward_type': 'sparse', 'act_space': 1},
    max_episode_steps=50,
)

register(
    id='FetchPickAndPlace-v4',
    entry_point='custom_gym.robotics:FetchPickAndPlaceEnv',
    kwargs={'reward_type': 'sparse', 'act_space': 1},
    max_episode_steps=50,
)

register(
    id='FetchReach-v5',
    entry_point='custom_gym.robotics:FetchReachEnv',
    kwargs={'reward_type': 'sparse', 'act_space': 1, 'goal_range': 'continuous',
        'obs_content': {
            'achieved_goal': True,
            'desired_goal': True,
            'instruction': False,
        }
    },
    max_episode_steps=50,
)

register(
    id='FetchPush-v5',
    entry_point='custom_gym.robotics:FetchPushEnv',
    kwargs={'reward_type': 'sparse', 'act_space': 1, 'goal_range': 'continuous',
        'obs_content': {
            'achieved_goal': True,
            'desired_goal': True,
            'instruction': False,
        }
    },
    max_episode_steps=50,
)

register(
    id='FetchSlide-v5',
    entry_point='custom_gym.robotics:FetchSlideEnv',
    kwargs={'reward_type': 'sparse', 'act_space': 1, 'goal_range': 'continuous',
        'obs_content': {
            'achieved_goal': True,
            'desired_goal': True,
            'instruction': False,
        }
    },
    max_episode_steps=50,
)

register(
    id='FetchPickAndPlace-v5',
    entry_point='custom_gym.robotics:FetchPickAndPlaceEnv',
    kwargs={'reward_type': 'sparse', 'act_space': 1, 'goal_range': 'continuous',
        'obs_content': {
            'achieved_goal': True,
            'desired_goal': True,
            'instruction': False,
        }
    },
    max_episode_steps=50,
)

# =================== classic control ===================
register(
    id='MassPointGoal-v0',
    entry_point='custom_gym.classic_control:MassPointGoalEnv',
    max_episode_steps=200,
)

register(
    id='MassPointGoalInstr-v0',
    entry_point='custom_gym.classic_control:MassPointGoalInstrEnv',
    max_episode_steps=200,
)

register(
    id='MassPointGoalAction-v0',
    entry_point='custom_gym.classic_control:MassPointGoalActionEnv',
    max_episode_steps=200,
)

# Trajectory
register(
    id = 'MassPointTraj-v0',
    entry_point='custom_gym.classic_control:MassPointTrajEnv',
    max_episode_steps=200,
)

register(
    id = 'MassPointTrajInstr-v0',
    entry_point='custom_gym.classic_control:MassPointTrajInstrEnv',
    max_episode_steps=200,
)

register(
    id = 'MassPointTrajAction-v0',
    entry_point='custom_gym.classic_control:MassPointTrajActionEnv',
    max_episode_steps=200,
)

# Goal-v1
register(
    id = 'MassPointGoal-v1',
    entry_point='custom_gym.classic_control:MassPointGoalEnv_v1',
    max_episode_steps=200,
)

register(
    id = 'MassPointGoalInstr-v1',
    entry_point='custom_gym.classic_control:MassPointGoalInstrEnv_v1',
    max_episode_steps=200,
)

register(
    id = 'MassPointGoalAction-v1',
    entry_point='custom_gym.classic_control:MassPointGoalActionEnv_v1',
    max_episode_steps=200,
)

# Trajectory-v1
register(
    id = 'MassPointTraj-v1',
    entry_point='custom_gym.classic_control:MassPointTrajEnv_v1',
    max_episode_steps=200,
)

register(
    id = 'MassPointTrajInstr-v1',
    entry_point='custom_gym.classic_control:MassPointTrajInstrEnv_v1',
    max_episode_steps=200,
)

register(
    id = 'MassPointTrajAction-v1',
    entry_point='custom_gym.classic_control:MassPointTrajActionEnv_v1',
    max_episode_steps=200,
)

# =================== Kobuki ===================
register(
    id = 'KobukiGEPGoal-v0',
    entry_point='custom_gym.kobukiGEP:KobukiGEPGoal',
    max_episode_steps=200,
)

register(
    id = 'KobukiGEPTraj-v0',
    entry_point='custom_gym.kobukiGEP:KobukiGEPTraj',
    max_episode_steps=200,
)

register(
    id = 'FiveTargetEnv-v2',
    entry_point='custom_gym.kobukiGEP:FiveTargetEnv_v2',
    max_episode_steps=200,
)

# =================== mujoco ===================
# Goal
register(
    id='ReacherGoal-v0',
    entry_point='custom_gym.mujoco:ReacherGoal',
    max_episode_steps = 50,
    reward_threshold=-3.75,
)

register(
    id='ReacherGoalInstr-v0',
    entry_point='custom_gym.mujoco:ReacherGoalInstr',
    max_episode_steps = 50,
    reward_threshold=-3.75,
)

register(
    id='ReacherGoalAction-v0',
    entry_point='custom_gym.mujoco:ReacherGoalAction',
    max_episode_steps = 50,
    reward_threshold=-3.75,
)

# Trajectory
register(
    id='ReacherTraj-v0',
    entry_point='custom_gym.mujoco:ReacherTraj',
    max_episode_steps = 50,
)

register(
    id='ReacherTrajInstr-v0',
    entry_point='custom_gym.mujoco:ReacherTrajInstr',
    max_episode_steps = 50,
)

register(
    id='ReacherTrajAction-v0',
    entry_point='custom_gym.mujoco:ReacherTrajAction',
    max_episode_steps = 50,
)

# =================== unity =================== 
register(
    id='Kobuki-v0',
    entry_point='custom_gym.unity:KobukiEnv',
    
)
