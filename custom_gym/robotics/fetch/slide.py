import numpy as np

from gym import utils
from custom_gym.robotics import fetch_env


class FetchSlideEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.05,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.7, 1.1, 0.4, 1., 0., 0., 0.],
        }
        obs_content = {
            'achieved_goal': False,
            'desired_goal': False,
            'instruction': True,
        }
        fetch_env.FetchEnv.__init__(
            self, 'fetch/slide.xml', has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=-0.02, target_in_the_air=False, target_offset=np.array([0.4, 0.0, 0.0]),
            obj_range=0.1, target_range=0.3, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, obs_content=obs_content)
        utils.EzPickle.__init__(self)

    def _sample_goal(self, target=None):
        # random target
        if target is None:
            target = self.np_random.randint(8)

        # one-hot instruction
        self.instruction = np.zeros(8)
        self.instruction[target] = 1

        # desired goal
        goals = self.target_range * np.array([
            [-0.8, -0.8, 0],
            [-0.8, 0.8, 0],
            [-0.4, 0, 0],
            [0, -0.8, 0],
            [0, 0.8, 0],
            [0.4, 0, 0],
            [0.8, -0.8, 0],
            [0.8, 0.8, 0],
        ])
        desired_goal = goals[target]

        # goal
        goal = super()._sample_goal(desired_goal)
        return goal
