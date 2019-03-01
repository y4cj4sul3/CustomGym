import numpy as np
from gym import utils
from custom_gym.robotics import fetch_env


class FetchReachEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', instr_space=0, act_space=0, goal_range='discrete',
        obs_content={
            'achieved_goal': False,
            'desired_goal': False,
            'instruction': True,
        }
    ):

        initial_qpos = {
            'robot0:slide0': 0.4049,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }
        
        if goal_range == 'discrete':
            target_range = 0.1
            distance_threshold = 0.05
        else:
            target_range = 0.15
            distance_threshold = 0.03
            
        fetch_env.FetchEnv.__init__(
            self, 'fetch/reach.xml', has_object=False, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=target_range, distance_threshold=distance_threshold,
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

