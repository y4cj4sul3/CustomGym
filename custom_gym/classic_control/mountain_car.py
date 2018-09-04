"""
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
#from custom_gym.utils import Recoder

class MountainCarEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.min_position = -1.
        self.max_position = 1.
        self.max_speed = 0.07
        
        self.right_goal = 0.9
        self.left_goal = -0.9

        self.low = np.array([self.min_position, -self.max_speed, 0])
        self.high = np.array([self.max_position, self.max_speed, 1])

        self.viewer = None

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low, self.high)

        #self.recoder = Recoder('dataset/MountainCarEx_new/')

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        
        # states before simulate
        position, velocity, _ = self.state

        # record traj
        #self.recoder.step(self.state[0:2], action)

        # simualte
        velocity += (action-1)*0.001 + math.cos(3*position - np.pi /2)*(-0.0025)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if (position==self.min_position and velocity<0): velocity = 0
        if (position==self.max_position and velocity>0): velocity = 0
        
        # states after simulate
        if self.task_id == 0:
            # right goal
            done = bool(position >= self.right_goal)
        elif self.task_id == 1:
            # left goal
            done = bool(position <= self.left_goal)
        
        # save traj
        #if done:
        #    self.recoder.save([position, velocity])

        reward = -1.0

        self.state = (position, velocity, self.task_id)
        return np.array(self.state), reward, done, {}

    def reset(self, task_id=None):
        # task
        if task_id is None:
            task_id = np.random.randint(2)
        self.task_id = task_id

        # goal
        if self.task_id == 0:
            self.goal_position = self.right_goal
        elif self.task_id == 1:
            self.goal_position = self.left_goal

        # reset traj
        #self.recoder.reset_traj()
        #self.recoder.traj['task'] = self.task_id

        # state
        self.state = np.array([self.np_random.uniform(low=-0.1, high=0.1), 0, self.task_id])
        #print(np.shape([self.task_id, self.state]))
        return np.array(self.state)

    def _height(self, xs):
        return np.sin(3 * xs - np.pi / 2)*.45+.55

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width/world_width
        carwidth=40
        carheight=20

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs-self.min_position)*scale, ys*scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l,r,t,b = -carwidth/2, carwidth/2, carheight, 0
            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight/2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth/4,clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight/2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth/4,clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = 0 #(self.goal_position-self.min_position)*scale
            flagy1 = 0 #self._height(self.goal_position)*scale
            flagy2 = flagy1 + 50
            self.flagtrans = rendering.Transform()
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            flagpole.add_attr(self.flagtrans)
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
            flag.add_attr(self.flagtrans)
            flag.set_color(.8,.8,0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation((pos-self.min_position)*scale, self._height(pos)*scale)
        self.cartrans.set_rotation(math.cos(3 * pos - np.pi / 2))
        self.flagtrans.set_translation((self.goal_position-self.min_position)*scale, self._height(self.goal_position)*scale)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()
