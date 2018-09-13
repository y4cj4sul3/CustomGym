import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class FiveTargetEnv(gym.Env):
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
        self.high_instr = np.ones(self.num_targets)
        self.low_instr = np.zeros(self.num_targets)
        
        self.instr_space = spaces.Box(self.low_instr, self.high_instr, dtype=np.float32)

        # Define Action Space
        # [forward_speed, rotate]
        self.high_action = np.array([1, 1])
        self.low_action = np.array([0, -1])
        
        self.action_space = spaces.Box(self.low_action, self.high_action, dtype=np.float32)

        # Define Observation Space
        # [xpos, ypos, xface, yface] + instruction
        self.high_obs = np.concatenate(([1, 1, 1, 1], self.high_instr))
        self.low_obs = np.concatenate(([-1, -1, -1, -1], self.low_instr))

        self.observation_space = spaces.Box(self.low_obs, self.high_obs, dtype=np.float32)

        # Reset Env
        self.viewer = None

        # Target
        self.targets = []
        self.target_coord = [18, 90, 162, 234, 306]
        self.target_coord = [np.deg2rad(x) for x in self.target_coord]
        self.target_coord = [(np.cos(x), np.sin(x)) for x in self.target_coord]
        
        self.target_size = 0.1

        self.seed()
        self.reset()
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # Check action
        #print(action)
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
        self.obs = np.concatenate((self.obs, self.instr))
        self.obs = np.clip(self.obs, self.low_obs, self.high_obs)

        # TODO Define reward function
        # TODO Define done
        done = False
        reward = 0
        xpos, ypos, xface, yface = self.obs[0:4]
        # time penalty
        reward -= 0.005
        # hit the target
        for i in range(self.num_targets):
            vec = np.array([xpos-self.target_coord[i][0], ypos-self.target_coord[i][1]])
            dist = np.linalg.norm(vec)
            if dist < self.target_size:
                done = True
                if i == self.task:
                    reward += 1
                else:
                    reward += -0.2
                break
        # hit the wall
        if not done:
            if xpos == 1 or xpos == -1 or ypos == 1 or ypos == -1:
                done = True
                reward += -1

        return self.obs, reward, done, {}

    def reset(self, task=None):
        
        # Task
        if task is None:
            #task = np.random.random_sample(np.shape(self.low_task))
            #task = task*(self.high_task-self.low_task)+self.low_task
            task = np.random.randint(5)
        self.task = np.array(task)
        
        # Instruction
        self.instr = np.zeros(5)
        self.instr[self.task] = 1
        assert self.instr_space.contains(self.instr), "%r (%s) invalid task" % (self.instr, type(self.instr))

        # Set target
        self.target_color = []
        for i in range(5):
            self.target_color.append([0, 1, 0])
        self.target_color[self.task] = [1, 0, 0]

        # State
        theta = 2*np.pi*np.random.rand()
        self.obs = np.concatenate(([0, 0, np.cos(theta), np.sin(theta)], self.instr))

        return self.obs

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
            #self.region_trans = rendering.Transform()
            
            # draw traget
            for i in range(5):
                region = rendering.make_circle(region_size)
                region_trans = rendering.Transform()
                region_trans.set_translation((self.target_coord[i][0]+1)*scale, (self.target_coord[i][1]+1)*scale)
                #region.set_color(0.9, 0.9, 0)
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
        xpos, ypos, xface, yface = self.obs[0:4]
        theta = np.arctan2(yface, xface)
        self.point_trans.set_translation((xpos+1)*scale, (ypos+1)*scale)
        self.point_trans.set_rotation(theta)
        #xpos, ypos = self.task
        #self.region_trans.set_translation((xpos+1)*scale, (ypos+1)*scale)
        print(len(self.targets))
        for i in range(5):
            r, g, b = self.target_color[i]
            self.targets[i].set_color(r, g, b)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()





        






