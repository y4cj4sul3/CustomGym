import numpy as np
from gym import utils
from custom_gym.robotics import fetch_env


class FetchPickAndPlaceEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', instr_space=0, act_space=0, goal_range='discrete',
        obs_content={
            'achieved_goal': False,
            'desired_goal': False,
            'instruction': True,
        }
    ):
        
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        
        if goal_range == 'discrete':
            target_range = 0.1
        else:
            target_range = 0.15
        
        fetch_env.FetchEnv.__init__(
            self, 'fetch/pick_and_place.xml', has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=target_range, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, obs_content=obs_content,
            instr_space=instr_space, act_space=act_space, goal_range=goal_range)
        utils.EzPickle.__init__(self)

    def _sample_goal(self, target=None):
        # random target
        if target is None:
            target = self.np_random.randint(8)

        # instruction
        self._set_instruction(target)

        if self.goal_range == 'discrete':
            # desired goal
            desired_goal = np.array([self.target_range if (target >> n) & 1 == 1 else -self.target_range for n in range(3)])
            # goal
            goal = super()._sample_goal(desired_goal)
        else:
            # random goal in continuous space
            goal = super()._sample_goal()

        return goal
