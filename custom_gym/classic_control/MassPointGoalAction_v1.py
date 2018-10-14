import numpy as np
from custom_gym.classic_control import MassPointGoalEnv_v1

class MassPointGoalActionEnv_v1(MassPointGoalEnv_v1):

    def step(self, action):
        # Change Action
        action = np.clip(action, self.low_action, self.high_action)
        action = action * 0.5

        return super().step(action)

