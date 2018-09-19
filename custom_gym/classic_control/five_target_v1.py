import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class FiveTargetEnv_v1(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, idx=0):
        # Parameters      
        self.min_pos = -1
        self.max_pos = 1
        self.speed_scale = 0.05
        self.rotate_scale = 0.3
        self.num_targets = 5
        
        # Define Instruction Space
        # one-hot
        self.high_instr = np.ones(self.num_targets)
        self.low_instr = np.zeros(self.num_targets)
        
        self.instr_space = spaces.Box(self.low_instr, self.high_instr, dtype=np.float32)

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
        # [xpos, ypos] + instruction
        self.high_obs = np.concatenate((self.high_state, self.high_instr))
        self.low_obs = np.concatenate((self.low_state, self.low_instr))

        self.observation_space = spaces.Box(self.low_obs, self.high_obs, dtype=np.float32)

        # Reset Env
        self.viewer = None

        # Target
        self.targets = []
        self.target_coord = range(18, 180, 36)
        self.target_coord = [np.deg2rad(x) for x in self.target_coord]
        self.target_coord = [(np.cos(x), np.sin(x)) for x in self.target_coord]
        print(self.target_coord)

        self.target_size = 0.05

        # Arena
        self.arena_size = 1

        # Timestep
        self.max_timesteps = 200
        self.timesteps = 0

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
        xpos, ypos, xface, yface = self.state
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
        self.state = [xpos, ypos, xface, yface]
        self.state = np.clip(self.state, self.low_state, self.high_state)

        # TODO Define reward function
        # TODO Define done
        done = False
        reward = 0
        xpos, ypos, xface, yface = self.state
        # time penalty(distance)
        vec = np.array([xpos-self.target_coord[self.task][0], ypos-self.target_coord[self.task][1]])
        dist = np.linalg.norm(vec)
        reward += -dist * 0.1
        
        done_status = ''
        # hit the target
        for i in range(self.num_targets):
            vec = np.array([xpos-self.target_coord[i][0], ypos-self.target_coord[i][1]])
            dist = np.linalg.norm(vec)
            if dist < self.target_size:
                done = True
                if i == self.task:
                    print('Right Target')
                    done_status = 'Right Target'
                    reward += 1
                else:
                    print('Wrong Target')
                    done_status = 'Wrong Target'
                    reward += -0.2
                break
        
        # hit the wall
        if not done:
            if xpos == 1 or xpos == -1 or ypos == 1 or ypos == -1:
            #if np.linalg.norm(np.array([xpos, ypos])) > self.arena_size:
                done = True
                reward += -1
                print('Hit the Wall')
                done_status = 'Hit the Wall'
        
        # times up
        self.timesteps += 1
        if not done and self.timesteps >= self.max_timesteps:
            done = True
            reward += -0.5
            print('Times Up')
            done_status = 'Times Up'

        return self.get_obs(), reward, done, {'done_status': done_status}

    def reset(self, task=None):
        
        # Task
        if task is None:
            #task = np.random.random_sample(np.shape(self.low_task))
            #task = task*(self.high_task-self.low_task)+self.low_task
            task = np.random.randint(self.num_targets)
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
        #theta = 2*np.pi*np.random.random_sample()
        #self.state = np.array([0, 0, np.cos(theta), np.sin(theta)])
        self.state = np.array([0, -.5, 0, 1])

        return self.get_obs()
        
    def get_obs(self):
        # Observation
        # [agent coord] + instruction
        obs = np.concatenate((self.state, self.instr))
        assert self.observation_space.contains(obs), "%r (%s) invalid task" % (obs, type(obs))
        return obs

    def render(self, mode='human'):
        # Parameters
        screen_size = 600
        world_size = self.max_pos - self.min_pos
        scale = screen_size/world_size
        #scale *= 0.8

        point_size = 15
        region_size = self.target_size*scale
	
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_size, screen_size)

            self.point_trans = rendering.Transform()
            #self.region_trans = rendering.Transform()
            '''            
            # draw arena
            border = rendering.make_circle(self.arena_size*scale*1.1)
            arena = rendering.make_circle(self.arena_size*scale)
            arena_trans = rendering.Transform()
            arena_trans.set_translation(screen_size/2, screen_size/2)
            border.set_color(0.5, 0.5, 0.5)
            arena.set_color(0.9, 0.9, 0.9)
            border.add_attr(arena_trans)
            arena.add_attr(arena_trans)
            self.viewer.add_geom(border)
            self.viewer.add_geom(arena)
            '''
            # draw traget
            for i in range(self.num_targets):
                region = rendering.make_circle(region_size)
                region_trans = rendering.Transform()
                region_trans.set_translation(self.target_coord[i][0]*scale+screen_size/2, self.target_coord[i][1]*scale+screen_size/2)
                region.add_attr(region_trans)
                self.targets.append(region)
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
        # agent
        xpos, ypos, xface, yface = self.state
        theta = np.arctan2(yface, xface)
        self.point_trans.set_translation(xpos*scale+screen_size/2, ypos*scale+screen_size/2)
        self.point_trans.set_rotation(theta)
        # target
        #print(len(self.targets))
        for i in range(self.num_targets):
            r, g, b = self.target_color[i]
            self.targets[i].set_color(r, g, b)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
