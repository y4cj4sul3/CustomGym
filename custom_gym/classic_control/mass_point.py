import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class MassPointEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        # Parameters      
        self.min_pos = -1
        self.max_pos = 1
        self.speed_scale = 0.01
        self.rotate_scale = 0.3
        
        # Define Task Space
        self.high_task = np.array([1, 1])
        self.low_task = np.array([-1, -1])
        
        self.task_space = spaces.Box(self.low_task, self.high_task, dtype=np.float32)

        # Define Action Space
        # [forward_speed, rotate]
        self.high_action = np.array([1, 1])
        self.low_action = np.array([0, -1])
        
        self.action_space = spaces.Box(self.low_action, self.high_action, dtype=np.float32)

        # Define Observation Space
        # [xpos, ypos, xface, yface] + task
        self.high_obs = np.concatenate(([1, 1, 1, 1], self.high_task))
        self.low_obs = np.concatenate(([-1, -1, -1, -1], self.low_task))

        self.observation_space = spaces.Box(self.low_obs, self.high_obs, dtype=np.float32)

        # Reset Env
        self.viewer = None

        self.seed()
        self.reset()
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # Check action
        action = np.clip(action, self.low_action, self.high_action)
        assert self.action_space.contains(action), "%r (%s) invalid action" % (action, type(action))

        # States before simulate
        xpos, ypos, xface, yface = self.obs[0:4]
        f_speed, rotate = action
        theta = np.arctan2(yface, xface)

        # Simulate
        # update facing
        theta = theta + self.rotate_scale*rotate
        xface = np.cos(theta)
        yface = np.sin(theta)
        # update position
        xpos = xpos + xface*self.speed_scale*f_speed
        ypos = ypos + yface*self.speed_scale*f_speed

        # States after simulate
        self.obs = [xpos, ypos, xface, yface]
        self.obs = np.concatenate((self.obs, self.task))
        self.obs = np.clip(self.obs, self.low_obs, self.high_obs)

        # TODO Define reward function
        reward = 0
        dist = np.sqrt(np.sum(np.power(self.obs[0:2]-self.task, 2)))
        if dist < 0.1:
            reward += 2-dist*10

        # TODO Define done
        done = False

        return self.obs, reward, done, {}

    def reset(self, task=None):
        
        # Task
        if task is None:
            task = np.random.random_sample(np.shape(self.low_task))
            task = task*(self.high_task-self.low_task)+self.low_task
        assert self.task_space.contains(task), "%r (%s) invalid task" % (task, type(task))
        self.task = np.array(task)

        # State
        theta = 2*np.pi*np.random.rand()
        self.obs = np.concatenate(([0, 0, np.cos(theta), np.sin(theta)], self.task))

        return self.obs

    def render(self, mode='human'):
        # Parameters
        screen_size = 600
        world_size = self.max_pos - self.min_pos
        scale = screen_size/world_size

        point_size = 5
        region_size = 0.1*scale

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_size, screen_size)

            self.point_trans = rendering.Transform()
            self.region_trans = rendering.Transform()
            
            # draw traget
            region = rendering.make_circle(region_size)
            region.set_color(0.9, 0.9, 0)
            region.add_attr(self.region_trans)
            self.viewer.add_geom(region)

            # draw point
            point = rendering.make_circle(point_size)
            point.set_color(.2, .2, 1)
            point.add_attr(self.point_trans)
            self.viewer.add_geom(point)

            # draw point head
            point_head = rendering.FilledPolygon([(0, -point_size), (2*point_size, 0), (0, point_size)])
            point_head.set_color(1, 0, 0)
            point_head.add_attr(self.point_trans)
            self.viewer.add_geom(point_head)

        # Transform
        xpos, ypos, xface, yface = self.obs[0:4]
        theta = np.arctan2(yface, xface)
        self.point_trans.set_translation((xpos+1)*scale, (ypos+1)*scale)
        self.point_trans.set_rotation(theta)
        xpos, ypos = self.task
        self.region_trans.set_translation((xpos+1)*scale, (ypos+1)*scale)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()





        






