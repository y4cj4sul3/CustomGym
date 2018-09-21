from custom_gym.mujoco.mujoco_env import MujocoEnv
# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
#from gym.envs.mujoco.ant import AntEnv
#from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
#from gym.envs.mujoco.hopper import HopperEnv
#from gym.envs.mujoco.walker2d import Walker2dEnv
#from gym.envs.mujoco.humanoid import HumanoidEnv
#from gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
#from gym.envs.mujoco.inverted_double_pendulum import InvertedDoublePendulumEnv
from custom_gym.mujoco.reacher import ReacherEnv
from custom_gym.mujoco.reacher_gep import ReacherGEPEnv
from custom_gym.mujoco.reacher_gep_traj import ReacherGEPTrajEnv
from custom_gym.mujoco.reacher_five_target import ReacherFiveTargetEnv
from custom_gym.mujoco.reacher_five_target_v1 import ReacherFiveTargetEnv_v1
#from gym.envs.mujoco.swimmer import SwimmerEnv
#from gym.envs.mujoco.humanoidstandup import HumanoidStandupEnv
#from gym.envs.mujoco.pusher import PusherEnv
#from gym.envs.mujoco.thrower import ThrowerEnv
#from gym.envs.mujoco.striker import StrikerEnv
