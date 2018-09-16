import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

def calDist(p1, p2):
    return np.linalg.norm(p1 - p2)

class FiveTargetEnv_v2(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, idx=0):
        self.Debug = False

        # Parameters
        self.num_targets = 5

        self.state_dim = 2
        self.instr_dim = self.num_targets
        self.action_dim = 2

        self.min_pos = -1
        self.max_pos = 1
        self.speed_scale = 0.05
        self.rotate_scale = 0.3
        
        # Define Instruction Space: one-hot
        high = np.array([np.inf] * self.instr_dim)
        self.instr_space = spaces.Box(-high, high, dtype=np.float32)

        # Define Action Space: [forward_speed, rotate]
        high = np.array([np.inf] * self.action_dim)
        self.action_space = spaces.Box(-high, high, dtype=np.float32)

        # Define State Space: [xpos, ypos, xface, yface]
        high = np.array([np.inf] * 4)
        self.state_space = spaces.Box(-high, high, dtype=np.float32)

        # Define Observation Space: [xpos, ypos] + instruction
        high = np.array([np.inf] * (2 + self.instr_dim))
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Reset Env
        self.viewer = None

        # Target
        self.targets = []
        self.target_coord = [(np.cos(np.deg2rad(x)), np.sin(np.deg2rad(x))) for x in range(18, 180, 36)]
        self.target_size = 0.05

        # Arena
        self.arena_size = 1

        # Timestep
        self.timesteps = 0
        self.max_timesteps = 200

        self.seed()
        self.reset()
        
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.timesteps += 1

        # Check action
        action = np.clip(action, [0, -3],  [2, 3])
        assert self.action_space.contains(action), "%r (%s) invalid action" % (action, type(action))
        
        # States before simulate
        xpos, ypos, xface, yface = self.state
        f_speed, rotate = action
        theta = np.arctan2(yface, xface)

        # Simulate
        # ------------------------------------------
        # update facing
        theta = theta + self.rotate_scale*rotate
        xface = np.cos(theta)
        yface = np.sin(theta)
        
        # update position
        xpos = xpos + xface*self.speed_scale*f_speed
        ypos = ypos + yface*self.speed_scale*f_speed

        # States after simulate
        self.state = [xpos, ypos, xface, yface]
    
        # TODO Define reward function, Define done
        reward, done = 0, False
        
        # time penalty(distance)
        p1, p2 = np.array([xpos, ypos]), np.array(self.target_coord[self.task])
        reward += np.linalg.norm(p1 - p2) * 0.1
        
        # Hit the target
        for i in range(self.num_targets):
            p1, p2 = np.array([xpos, ypos]), np.array(self.target_coord[i])
            if np.linalg.norm(p1 - p2) < self.target_size:
                done = True
                reward = reward + 1 if i == self.task else reward + (-0.2)
                
                if self.Debug:
                    if i == self.task: 
                        print('Right Target')
                    else: 
                        print('Wrong Target')
                break
        
        # Hit the wall
        if (not done) and (xpos > 1 or xpos < -1 or ypos > 1 or ypos < -1):
            done = True
            reward += -1
            
            if self.Debug: 
                print('Hit the Wall')

        # Times up
        if not done and self.timesteps == self.max_timesteps:
            done = True
            reward += -0.5
            
            if self.Debug: 
                print('Times Up')

        return self.get_obs(), reward, done, {}

    def reset(self, task=None):
        # Timestep
        self.timesteps = 0
        
        # Task
        self.task = np.random.randint(self.num_targets) if task is None else np.array(task)
        
        # Instruction: One-Hot
        self.instr = np.zeros(self.num_targets)
        self.instr[self.task] = 1
        assert self.instr_space.contains(self.instr), "%r (%s) invalid task" % (self.instr, type(self.instr))

        # Set target
        self.target_color = [[0, 1, 0] for _ in range(5)]
        self.target_color[self.task] = [1, 0, 0]

        # State
        self.state = np.array([0, 0, -.5, 1])

        return self.get_obs()
        
    def get_obs(self):
        # Observation: [agent coord] + instruction
        obs = np.concatenate((self.state[0:2], self.instr))
        assert self.observation_space.contains(obs), "%r (%s) invalid task" % (obs, type(obs))
        return obs

    def render(self, mode='human'):
        # Parameters
        screen_size = 600
        world_size = self.max_pos - self.min_pos
        scale = screen_size/world_size

        point_size = 15
        region_size = self.target_size*scale
	
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_size, screen_size)
            self.point_trans = rendering.Transform()

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

        # Transform: agent
        xpos, ypos, xface, yface = self.state
        theta = np.arctan2(yface, xface)
        self.point_trans.set_translation(xpos*scale+screen_size/2, ypos*scale+screen_size/2)
        self.point_trans.set_rotation(theta)
        
        # Transform: target
        for i in range(self.num_targets): 
            r, g, b = self.target_color[i]
            self.targets[i].set_color(r, g, b)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
