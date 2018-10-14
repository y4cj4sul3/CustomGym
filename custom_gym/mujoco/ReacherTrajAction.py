import numpy as np

from custom_gym.mujoco import ReacherTraj
from custom_gym.mujoco.my_utils import *

class ReacherTrajAction(ReacherTraj):
    
    def step(self, action):
        action = np.clip(action, self.low_action, self.high_action)
        action = action * 0.9
        return super().step(action)
