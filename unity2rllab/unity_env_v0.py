from unityagents import UnityEnvironment

import gym
from rllab.envs.base import Env, Step
from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
from rllab.spaces.discrete import Discrete

import numpy as np

class UnityEnvV0(Env, Serializable):
    def __init__(self, app_name, time_state=False, idx=0, is_render=False, no_graphics=False, recording=True):
        Serializable.quick_init(self, locals())
    
        # Unity scene
        self._env = UnityEnvironment(file_name=app_name, worker_id=idx, no_graphics=no_graphics)
        self.id = 0

        self.name = app_name
        self.idx = idx
        self.is_render = is_render    

        self.time_state = time_state
        self.time_step = 0

        # Check brain configuration
        assert len(self._env.brains) == 1
        self.brain_name = self._env.external_brain_names[0]
        brain = self._env.brains[self.brain_name]
        
        # Check for number of agents in scene
        initial_info = self._env.reset()[self.brain_name]
        self.use_visual = (brain.number_visual_observations == 1) and False
        self.recording = brain.number_visual_observations == 1 and recording

        # Set observation and action spaces
        if brain.vector_action_space_type == "discrete":
            self._action_space = Discrete(1)
        else:
            high = np.array([np.inf] * (brain.vector_action_space_size))
            self._action_space = Box(-high, high)
        # ----------------------------------
        if self.use_visual and False and no_graphic:
            high = np.array([np.inf] * brain.camera_resolutions[0]["height"] * 
                brain.camera_resolutions[0]["width"] * 3)
            self._observation_space = Box(-high, high)
        else:
            if self.time_state:
                high = np.array([np.inf] * (brain.vector_observation_space_size+1))
            else:
                high = np.array([np.inf] * (brain.vector_observation_space_size))
            self._observation_space = Box(-high, high)

        # video buffer
        self.frames = []
    
    def reset(self):
        self.frames = []
        info = self._env.reset()[self.brain_name] 
        if self.is_render: self.observation = info.visual_observations[0]
        state = info.vector_observations[0][:]
        self._pos = info.vector_observations[0][:2]
        if self.time_state: 
            state = np.hstack((state, [self.time_step]))
            self.time_step += 1
        self._collect_frames(info.visual_observations[0][0])
        return state.flatten()

    def step(self, action):
        info = self._env.step([action])[self.brain_name]
        if self.is_render: self.observation = info.visual_observations[0]
        state = info.vector_observations[0][:]
        self._pos = info.vector_observations[0][:2]
        reward = info.rewards[0]
        done = info.local_done[0]
        if self.time_state: 
            state = np.hstack((state, [self.time_step]))
            self.time_step += 1
            if done: self.time_step = 0
        self._collect_frames(info.visual_observations[0][0])
        return Step(observation=state.flatten(), reward=reward, done=done)

    def terminate(self):
        self._env.close()

    def render(self, mode=None):
        if self.is_render: 
            x = self.observation[0] * 255
            return np.array(x).astype('uint8') 
        else:    
            return np.zeros((480, 360, 3))

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
