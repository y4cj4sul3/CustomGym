import numpy as np

import gym
from gym.spaces import Discrete, Box

from unityagents import UnityEnvironment

class UnityEnvV3():
    def __init__(self, app_name, idx=0, no_graphics=False, recording=True):
        # Unity scene
        self._env = UnityEnvironment(file_name=app_name, worker_id=idx, no_graphics=no_graphics)

        self.name = app_name
        self.idx = idx

        # Check brain configuration
        assert len(self._env.brains) == 1
        self.brain_name = self._env.external_brain_names[0]
        brain = self._env.brains[self.brain_name]
        
        # Check for number of agents in scene
        initial_info = self._env.reset()[self.brain_name]
        self.use_visual = (brain.number_visual_observations == 1) and recording
        self.recording = brain.number_visual_observations == 1 and recording

        # Set action space
        # ---------------------------------
        if brain.vector_action_space_type == "discrete":
            self._action_space = Discrete(1)
        else:
            high = np.array([np.inf] * (brain.vector_action_space_size))
            self._action_space = Box(-high, high)

        # Set observation space
        # ---------------------------------
        if self.use_visual and no_graphics and recording:
            high = np.array([np.inf] * brain.camera_resolutions[0]["height"] * 
                brain.camera_resolutions[0]["width"] * 3)
            self._observation_space = Box(-high, high)
        else:
            high = np.array([np.inf] * (brain.vector_observation_space_size))
            self._observation_space = Box(-high, high)

        # video buffer
        self._time = 0
        self.frames = []
    
    def reset(self):
        self._time = 0
        self.frames = []
        
        info = self._env.reset()[self.brain_name] 
        state = info.vector_observations[0][:]
        
        self._pos = info.vector_observations[0][:2]
        self._collect_frames(info.visual_observations[0][0])
        return state

    def step(self, action):
        info = self._env.step([action])[self.brain_name]
        
        state = info.vector_observations[0][:]
        reward = info.rewards[0]
        done = info.local_done[0]

        exinfo = {
            'episode': {
                'r': reward,
                'l': self._time,
            }
        }

        self._pos = info.vector_observations[0][:2]
        self._time += 1
        self._collect_frames(info.visual_observations[0][0])
        return state, reward, done, exinfo

    def close(self):
        self._env.close()

    def _collect_frames(self, frame):
        if self.recording:
            self.frames.append(np.uint8(frame*255))

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def position(self):
        return self._pos
