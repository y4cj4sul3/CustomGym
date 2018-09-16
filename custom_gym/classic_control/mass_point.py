import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from getch import getch, pause

class MassPointEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, idx=0):
        # Parameters      
        self.min_pos = -1
        self.max_pos = 1
        self.speed_scale = 0.3
        self.rotate_scale = 0.5
        self.done = False
        self.task = None
        
        # Define Task Space
        self.high_task = np.array([1, 1])
        self.low_task = np.array([-1, -1])
        
        self.task_space = spaces.Box(self.low_task, self.high_task, dtype=np.float32)

        # Define Action Space
        # [forward_speed, rotate]
        self.high_action = np.array([1, 1])
        self.low_action = np.array([0, -1])
        
        self.action_space = spaces.Box(self.low_action, self.high_action, dtype=np.float32)
        
        # Define State Space
        # [xpos, ypos, xface, yface]
        self.high_state = np.array([1, 1, 1, 1])
        self.low_state = -self.high_state
        
        self.state_space = spaces.Box(self.low_state, self.high_state, dtype=np.float32)

        # Define Observation Space
        # [xpos, ypos, xface, yface] + task
        self.high_obs = self.high_task
        self.low_obs = self.low_task

        self.observation_space = spaces.Box(self.low_obs, self.high_obs, dtype=np.float32)

        # Reset Env
        self.viewer = None

        # Timestep
        self.max_timesteps = 200
        self.timesteps = 0

        self.seed()
        self.reset()
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        '''Step reward'''
        reward = 0

        '''Check action'''
        # take action
        #print(str(action))
        action = np.clip(action, self.low_action, self.high_action)
        assert self.action_space.contains(action), "%r (%s) invalid action" % (action, type(action))
        
        # States before simulate
        xpos, ypos, xface, yface = self.state
        f_speed, rotate = action
        theta = np.arctan2(yface, xface)

        '''Reward'''
        # time penalty(distance)
        #vec = np.array([xpos-self.target_coord[self.task][0], ypos-self.target_coord[self.task][1]])
        #dist = np.linalg.norm(vec)
        #reward += -dist

        # check if hit the target point
        hit = 0
        if abs(xpos - self.task[0]) <= self.speed_scale/2 and abs(ypos - self.task[1]) <= self.speed_scale/2:
            hit = 1
            self.done = True
            reward += 1
            print('Success')
        # hit the wall
        if xpos == 1 or xpos == -1 or ypos == 1 or ypos == -1:
            hit = -1
            self.done = True
            reward += -1
            print('Hit the Wall')

        '''Simulate'''
        # step facing
        theta = theta + self.rotate_scale*rotate
        xface = np.cos(theta)
        yface = np.sin(theta)
        # step position
        xpos = xpos + xface*self.speed_scale*f_speed
        ypos = ypos + yface*self.speed_scale*f_speed

        '''Update'''
        # States after simulate
        self.state = [xpos, ypos, xface, yface]
        self.state = np.clip(self.state, self.low_state, self.high_state)
        # time
        self.timesteps += 1

        # check max time step : times up
        if not self.done and self.timesteps >= self.max_timesteps:
            self.done = True
            reward += -0.5
            print('Times Up')

        return self.get_obs(), reward, self.done, {'hit':hit, 't':self.timesteps}

    def reset(self, task=None):
        # Task
        if task is None:
            #self.task = np.random.random_sample(np.shape(self.low_task))
            #self.task = self.task*(self.high_task-self.low_task)+self.low_task
            self.task = np.random.uniform(0, 1, (2,))
            #print('self_task:' + str(self.task))
        else:
            self.task = np.array(task)
            #print('engineer_task:' + str(task))
        assert self.task_space.contains(self.task), "%r (%s) invalid task" % (self.task, type(self.task))

        # State
        #self.state = np.array([0, -0.5, np.sin(np.deg2rad(45)), np.cos(np.deg2rad(45))])
        self.state = np.array([0, -0.5, 0, 1])

        # Timestep
        self.timesteps = 0

        # Parameter
        self.done = False

        return self.get_obs()

    def get_obs(self):
        obs = self.state[0:2]
        assert self.observation_space.contains(obs), "%r (%s) invalid obs" % (obs, type(obs))
        return obs

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
        xpos, ypos, xface, yface = self.state[0:4]
        theta = np.arctan2(yface, xface)
        self.point_trans.set_translation((xpos+1)*scale, (ypos+1)*scale)
        self.point_trans.set_rotation(theta)
        xpos, ypos = self.task
        self.region_trans.set_translation((xpos+1)*scale, (ypos+1)*scale)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()