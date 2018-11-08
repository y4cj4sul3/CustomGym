import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class KobukiGEPGoal(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    rotate_scale = 0.3
    num_targets = 5

    scale_up = 0.01
    room_x = 380 * scale_up
    room_y = 260 * scale_up
    

    x_limit = room_x / 2
    y_limit = room_y / 2

    point_size = 35 / 2 * scale_up
    target_size = 35 / 2 * scale_up

    speed_scale = 10.0
    
    #start_postiion = np.array([-x_limit+1, -y_limit+1, 0, 1])
    start_postiion = np.array([0, -1.125, 0, 1])

    target_radius = 150 * scale_up
    target_transform = np.array([0, -50]) * scale_up

    viewer_scale_up = 100

    def __init__(self):
        # Settings
        self.random_task = True # this should be True when training

        #
        self._define_space()

        # Reset Env
        self.viewer = None

        # Target : target geom (for rendering)
        self._set_targets()
        
        # Timestep
        self.max_timesteps = 200
        self.timesteps = 0

        # Episode
        self.episode = 0

        self.seed()
        self.reset()

    def _define_space(self):
        # Define Instruction Space (5 dim) : one-hot
        self.high_instr = np.ones(self.num_targets)
        self.low_instr = np.zeros(self.num_targets)
        self.instr_space = spaces.Box(self.low_instr, self.high_instr, dtype=np.float32)

        # Define Action Space (1 dim) : [rotate]
        self.high_action = np.array([1])
        self.low_action = np.array([-1])
        self.action_space = spaces.Box(self.low_action, self.high_action, dtype=np.float32)

        # Define State Space (4 dim) : [xpos, ypos, xface, yface]
        self.high_state = np.array([self.x_limit, self.y_limit, 1, 1])
        self.low_state = -self.high_state
        self.state_space = spaces.Box(self.low_state, self.high_state, dtype=np.float32)

        # Define Observation Space (9 dim) : state + instruction
        self.high_obs = np.concatenate((self.high_state, self.high_instr))
        self.low_obs = np.concatenate((self.low_state, self.low_instr))
        self.observation_space = spaces.Box(self.low_obs, self.high_obs, dtype=np.float32)

    def _set_targets(self):
        self.targets = []
        self.target_coord = range(0, 225, 45)
        self.target_coord = [np.deg2rad(x) for x in self.target_coord]
        self.target_coord = [(self.target_radius * np.cos(x), self.target_radius * np.sin(x)) for x in self.target_coord]
        self.target_coord = np.array(self.target_coord) + self.target_transform

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # Check action
        action = np.clip(action, self.low_action, self.high_action)
        assert self.action_space.contains(action), "%r (%s) invalid action" % (action, type(action))

        # States before simulate
        # Simulate: update facing
        # Simulate: update position
        # States after simulate
        xpos, ypos, xface, yface = self.state
        rotate = action[0]
        
        theta = np.arctan2(yface, xface)
        theta = theta + self.rotate_scale*rotate
        xface = np.cos(theta)
        yface = np.sin(theta)
        
        xpos = xpos + xface*self.speed_scale* self.scale_up
        ypos = ypos + yface*self.speed_scale* self.scale_up

        self.state = [xpos, ypos, xface, yface]
        self.state = np.clip(self.state, self.low_state, self.high_state)

        # Define reward function: done
        # time penalty(distance)
        reward = 0
        dist = np.linalg.norm(np.array([xpos, ypos])-self.target_coord[self.task])
        reward = reward + -dist * 0.1

        # hit the target
        done = False
        done_status = ''

        self.timesteps += 1
        
        for i in range(self.num_targets):            
            dist_i = np.linalg.norm(np.array([xpos, ypos])-self.target_coord[i])

            if dist_i < self.target_size + self.point_size:
                done = True
                if i == self.task:
                    done_status = 'Finish Task'
                    reward += 1
                else:
                    done_status = 'Wrong Target'
                    reward += -0.2
                break            
    
        # hit the wall
        if not done:
            if xpos == self.x_limit or xpos == -self.x_limit or ypos == -self.y_limit or ypos == self.y_limit:
                done = True
                reward += -1
                done_status = 'Hit the Wall'

        # times up
        if not done and self.timesteps >= self.max_timesteps:
            done = True
            reward += -0.5
            done_status = 'Times Up'        
        
        # episode count
        if done:
            self.episode = (self.episode + 1) % self.num_targets
        return self.get_obs(), reward, done, {'done_status': done_status, 'dist': dist}

    def reset(self, task=None, timestep=None):
        # timestep
        if timestep != None:
            self.timestep = timestep
        
        # Task
        if task is None:
            if self.random_task: # random task
                task = np.random.randint(self.num_targets)
            else: # increasing task
                task = self.episode
        self.task = np.array(task)

        # Instruction
        self.instr = np.zeros(self.num_targets)
        self.instr[self.task] = 1
        assert self.instr_space.contains(self.instr), "%r (%s) invalid task" % (self.instr, type(self.instr))

        # Set target
        self.target_color = []
        for i in range(5):
            self.target_color.append([0, 1, 0])
        self.target_color[self.task] = [1, 0, 0]

        # Timestep
        self.timesteps = 0

        # State
        self.state = self.start_postiion

        return self.get_obs()

    def get_obs(self):
        # Observation: state + instruction
        obs = np.concatenate((self.state[0:2], self.instr))
        #assert self.observation_space.contains(obs), "%r (%s) invalid task" % (obs, type(obs))
        return obs

    def render(self, mode='human'):
        # Parameters
        ###scale = self.room_y / (self.max_pos - self.min_pos)

        if self.viewer is None:
            
            from gym.envs.classic_control import rendering
            #print(self.room_x*self.viewer_scale_up, self.room_y*self.viewer_scale_up)
            self.viewer = rendering.Viewer(int(self.room_x*self.viewer_scale_up), int(self.room_y*self.viewer_scale_up))
            
            self.point_trans = rendering.Transform()
            
            # draw traget
            for i in range(self.num_targets):
                region = rendering.make_circle(self.target_size*self.viewer_scale_up)
                region_trans = rendering.Transform()
                region_trans.set_translation(self.target_coord[i][0]*self.viewer_scale_up+self.room_x*self.viewer_scale_up/2, self.target_coord[i][1]*self.viewer_scale_up+self.room_y*self.viewer_scale_up/2)
                ###region_trans.set_translation(self.target_coord[i][0]+self.room_x/2, self.target_coord[i][1]+self.room_y/2) #--
                region.add_attr(region_trans)
                self.targets.append(region)
                self.viewer.add_geom(region)
            
            # draw point
            point = rendering.make_circle(self.point_size*self.viewer_scale_up)
            point.set_color(.2, .2, 1)
            point.add_attr(self.point_trans)
            self.viewer.add_geom(point)

            # draw point head
            point_head = rendering.FilledPolygon([(0, -self.point_size*self.viewer_scale_up), (2*self.point_size*self.viewer_scale_up, 0), (0, self.point_size*self.viewer_scale_up)]) 
            point_head.set_color(1, 0, 0)
            point_head.add_attr(self.point_trans)
            self.viewer.add_geom(point_head)

        # Transform: agent
        xpos, ypos, xface, yface = self.state
        theta = np.arctan2(yface, xface)
        self.point_trans.set_translation(xpos*self.viewer_scale_up+self.room_x*self.viewer_scale_up/2, ypos*self.viewer_scale_up+self.room_y*self.viewer_scale_up/2)
        ###self.point_trans.set_translation(xpos+self.room_x/2*, ypos+self.room_y/2) #--
        self.point_trans.set_rotation(theta)

        # Transform: target
        for i in range(self.num_targets):
            r, g, b = self.target_color[i]
            self.targets[i].set_color(r, g, b)
        
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
