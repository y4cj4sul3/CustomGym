import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

def color_code2idx(colors, code):
    for i in range(5):
        if np.all(code == colors[i]): 
            return i
    ValueError("code not in colors")

def color_code2face(colors, code):
    for i in range(5):
        if np.all(code == colors[i]): 
            return 18 + (i * 72)
    ValueError("code not in colors")

def color_code2fixed(code):
    color_list = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ]
    for i in range(5):
        if np.all(code == color_list[i]): 
            return i
    ValueError("code not in colors")

class FiveTargetRandColorEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        # -------------------------------------------
        self.Debug = False
        self.counter = 0
        self.success = 0

        # Parameters
        # -------------------------------------------
        self.min_pos = -1
        self.max_pos = 1
        self.speed_scale = 0.05
        self.rotate_scale = 0.3
        self.num_targets = 5
        self.color_code_dim = 3
        
        # Define Spaces
        # -------------------------------------------
        # Define Skill Code
        self.high_instr = np.ones(self.color_code_dim)
        self.low_instr = np.zeros(self.color_code_dim)
        self.instr_space = spaces.Box(self.low_instr, self.high_instr, dtype=np.float32)

        # Define State Space: [xpos, ypos, xface, yface]
        self.high_state = np.array([1, 1, 1, 1])
        self.low_state = -self.high_state
        self.state_space = spaces.Box(self.low_state, self.high_state, dtype=np.float32)

        # Define Color Distance Space
        self.high_color_dist = np.array([np.inf] * 5)
        self.low_color_dist = np.array([-np.inf] * 5)
        self.color_dist_space = spaces.Box(self.low_color_dist, self.high_color_dist, dtype=np.float32)

        # Define Observation Space : [xpos, ypos, face, yface] + [r_d, g_d, b_d, p_d, c_d] + instruction
        self.high_obs = np.concatenate((self.high_state[0:4], self.high_color_dist, self.high_instr))
        self.low_obs = np.concatenate((self.low_state[0:4], self.low_color_dist, self.low_instr))
        self.observation_space = spaces.Box(self.low_obs, self.high_obs, dtype=np.float32)

        # Define Action Space: [forward_speed, rotate]
        self.high_action = np.array([1, 1])
        self.low_action = np.array([0, -1])
        self.action_space = spaces.Box(self.low_action, self.high_action, dtype=np.float32)

        # Reset Env
        # -------------------------------------------
        self.viewer = None

        # Target
        self.targets = []
        self.target_coord = [18, 90, 162, 234, 306]
        self.target_coord = [np.deg2rad(x) for x in self.target_coord]
        self.target_coord = [(np.cos(x), np.sin(x)) for x in self.target_coord]
        
        self.target_size = 0.2

        # Color Code
        self.colors = np.array([
            [1.0, 0.0, 0.0], 
            [0.0, 1.0, 0.0], 
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ])

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
        # --------------------------------------
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

        # Color Distance
        self.color_distances = np.zeros((self.num_targets,))
        for color in self.colors:
            idx, radius = color_code2fixed(color), np.deg2rad(color_code2face(self.colors, color))
            agent_position, target_position = self.state[0:2], np.array([np.cos(radius), np.sin(radius)])
            distance = np.linalg.norm(agent_position - target_position)
            self.color_distances[idx] = distance

        done = False
        reward = 0
        xpos, ypos, xface, yface = self.state
        
        # time penalty(distance)
        vec = np.array([xpos-self.target_coord[self.task][0], ypos-self.target_coord[self.task][1]])
        dist = np.linalg.norm(vec)
        reward += -dist * 0.01
        
        # hit the target
        for i in range(self.num_targets):
            vec = np.array([xpos-self.target_coord[i][0], ypos-self.target_coord[i][1]])
            dist = np.linalg.norm(vec)
            if dist < self.target_size:
                done = True
                if i == self.task:
                    if self.Debug: print('Right Target')
                    self.success += 1
                    reward += 0
                else:
                    if self.Debug: print('Wrong Target')
                    reward += -3
                break
        
        # hit the wall
        if not done:
            #if xpos == 1 or xpos == -1 or ypos == 1 or ypos == -1:
            if np.linalg.norm(np.array([xpos, ypos])) > self.arena_size:
                done = True
                reward += -3
                if self.Debug: print('Hit the Wall')
        
        # times up
        self.timesteps += 1
        if not done and self.timesteps >= self.max_timesteps:
            done = True
            reward += -0.5
            if self.Debug: print('Times Up')
        if done and self.Debug: 
            self.counter += 1
            print(reward, self.success / self.counter)
        return self.get_obs(), reward, done, {}

    def reset(self, task=None):
        # random shuffle
        np.random.shuffle(self.colors)

        # Task
        # task = [0, 0, 1]
        self.task = np.random.randint(5) if task is None else color_code2idx(self.colors, task)
        self.task = np.array(self.task)
        
        # Instruction
        self.instr = self.colors[self.task]
        assert self.instr_space.contains(self.instr), "%r (%s) invalid task" % (self.instr, type(self.instr))

        # Set Colors
        self.target_color = [color for color in self.colors]

        # Timestep
        self.timesteps = 0

        # State
        # theta = color_code2face(self.colors, self.instr) / 360 * 2 * np.pi
        theta = np.random.random_sample() * 2 * np.pi
        self.state = np.array([0, 0, np.cos(theta), np.sin(theta)])

        # Color Distance
        self.color_distances = np.zeros((self.num_targets,))
        for color in self.colors:
            idx, radius = color_code2fixed(color), np.deg2rad(color_code2face(self.colors, color))
            agent_position, target_position = self.state[0:2], np.array([np.cos(radius), np.sin(radius)])
            distance = np.linalg.norm(agent_position - target_position)
            self.color_distances[idx] = distance

        return self.get_obs()
        
    def get_obs(self):
        obs = np.concatenate((self.state[0:4], self.color_distances, self.instr))
        assert self.observation_space.contains(obs), "%r (%s) invalid task" % (obs, type(obs))
        return obs

    def render(self, mode='human'):
        # Parameters
        screen_size = 600
        world_size = self.max_pos - self.min_pos
        scale = screen_size / world_size

        point_size = 15
        region_size = self.target_size * scale

        # Visualize
        # -----------------------------------
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_size, screen_size)

            self.point_trans = rendering.Transform()
            self.task_target_trans = rendering.Transform()
            
            
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
            
            # draw traget
            for i in range(self.num_targets):
                region = rendering.make_circle(region_size)
                region_trans = rendering.Transform()
                region_trans.set_translation(self.target_coord[i][0]*scale+screen_size/2, self.target_coord[i][1]*scale+screen_size/2)
                region.add_attr(region_trans)
                self.targets.append(region)
                self.viewer.add_geom(region)

            # draw task traget
            region = rendering.make_circle(region_size * 0.9)
            region_trans = rendering.Transform()
            region.add_attr(self.task_target_trans)
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
        # -----------------------------------
        # agent
        xpos, ypos, xface, yface = self.state
        theta = np.arctan2(yface, xface)
        self.point_trans.set_translation((xpos+1)*scale, (ypos+1)*scale)
        self.point_trans.set_rotation(theta)
        self.task_target_trans.set_translation(self.target_coord[self.task][0]*scale+screen_size/2, self.target_coord[self.task][1]*scale+screen_size/2)
        
        # target
        for i in range(5):
            r, g, b = self.target_color[i]
            self.targets[i].set_color(r, g, b)
        r, g, b = self.target_color[self.task]
        self.targets[self.task].set_color(1.0, 0.8, 0.0)
        self.targets[self.num_targets].set_color(r, g, b)

        return self.viewer.render(return_rgb_array = (mode=='rgb_array'))

    def close(self):
        if self.viewer:
            self.viewer.close()
