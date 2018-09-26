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
        self.speed_scale = 0.06
        self.rotate_scale = 0.3
        self.num_targets = 5
        self.num_mid_targets = 2
        self.num_success = 0
        
        # Define Instruction Space
        # one-hot
        self.high_instr = np.ones(self.num_targets + self.num_mid_targets)
        self.low_instr = np.zeros(self.num_targets + self.num_mid_targets)
        
        self.instr_space = spaces.Box(self.low_instr, self.high_instr, dtype=np.float32)

        # Define Action Space
        # [forward_speed, rotate]
        #self.high_action = np.array([1, 1])
        #self.low_action = np.array([0, -1])
        self.high_action = np.array([1])
        self.low_action = np.array([-1])
        
        self.action_space = spaces.Box(self.low_action, self.high_action, dtype=np.float32)
        
        # Define State Space
        # [xpos, ypos, xface, yface]
        self.high_state = np.array([1, 1, 1, 1])
        self.low_state = self.high_state * -1
        
        self.state_space = spaces.Box(self.low_state, self.high_state, dtype=np.float32)

        # Define Observation Space
        # [xpos, ypos] + instruction
        self.high_obs = np.concatenate((self.high_state[0:2], self.high_instr[0:5]))
        self.low_obs = np.concatenate((self.low_state[0:2], self.low_instr[0:5]))

        self.observation_space = spaces.Box(self.low_obs, self.high_obs, dtype=np.float32)

        # Reset Env
        self.viewer = None

        # Testing Target
        self.targets = []
        self.target_coord = range(18, 180, 36)
        self.target_coord = [np.deg2rad(x) for x in self.target_coord]
        self.target_coord = [(np.cos(x), np.sin(x)) for x in self.target_coord]
        # Concate mid target
        mid_targets = [0, 180]
        mid_targets = [np.deg2rad(x) for x in mid_targets]
        mid_targets = [(np.cos(x) * 0.25, np.sin(x) * 0.25) for x in mid_targets]
        self.target_coord = np.concatenate((self.target_coord, mid_targets), axis=0)
        
        print(self.target_coord)

        self.target_size = 0.05

        # Training Target
        self.trainingTarget = np.zeros(4)

        # mid done to be used by Trajectory
        self.mid_done = True

        # Arena
        self.arena_size = 1

        # Timestep
        self.max_timesteps = 50
        self.timesteps = 0

        self.seed()
        self.reset()
        
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # Check action
        action = np.clip(action[0], self.low_action, self.high_action)
        assert self.action_space.contains(action), "%r (%s) invalid action" % (action, type(action))

        # States before simulate
        xpos, ypos, xface, yface = self.state
        #f_speed, rotate = action
        rotate = action[0]
        theta = np.arctan2(yface, xface)

        # Simulate
        # update facing
        theta = theta + self.rotate_scale*rotate
        xface = np.cos(theta)
        yface = np.sin(theta)
        # update position
        #xpos = xpos + xface*self.speed_scale*f_speed
        #ypos = ypos + yface*self.speed_scale*f_speed
        xpos = xpos + xface*self.speed_scale
        ypos = ypos + yface*self.speed_scale

        # States after simulate
        self.state = [xpos, ypos, xface, yface]
        self.state = np.clip(self.state, self.low_state, self.high_state)

        '''if self.timesteps == 10:
            self.rotate_scale = 0.3
        elif self.timesteps < 10:
            self.rotate_scale = self.rotate_scale * 0.9'''

        # TODO Define reward function
        # TODO Define done
        done = False
        reward = 0
        xpos, ypos, xface, yface = self.state
        # time penalty(distance)
        if self.id == 2 or self.id == 3:
            vec = np.array([xpos-self.target_coord[self.ins][0], ypos-self.target_coord[self.ins][1]])
        else:
            vec = np.array([xpos-self.trainingTarget[2], ypos-self.trainingTarget[3]])
        dist = np.linalg.norm(vec)
        reward += -dist * 0.1

        if self.id == 0 or self.id == 1: # goal traj trainingTarget [nan, nan, random, random]
            vec = np.array([xpos-self.trainingTarget[2], ypos-self.trainingTarget[3]])
            dist = np.linalg.norm(vec)
            if dist < self.target_size:
                done = True
                print('Success')
                #print('self.id',self.id)
        elif self.id == 2: # goal testing [envType, nan, instruction, nan, nan]
            # hit the target
            #for i in range(self.num_targets):
            vec = np.array([xpos-self.target_coord[self.ins][0], ypos-self.target_coord[self.ins][1]])
            dist = np.linalg.norm(vec)
            if dist < self.target_size:
                done = True
                    #print('self.id',self.id)
            '''if i == self.ins:
                print('Right Target')
                reward += 1
            else:
                print('Wrong Target')
                reward += -0.2
            break'''
        elif self.id == 3: # trajectory testing [envType, mid_instruction, instruction, nan, nan]
            # hit the target
            #for i in range(self.num_targets, self.num_targets + self.num_mid_targets):
            vec = np.array([xpos-self.target_coord[self.mid_ins][0], ypos-self.target_coord[self.mid_ins][1]])
            dist = np.linalg.norm(vec)
            if dist < self.target_size:
                self.mid_done = True
                    #print('self.id',self.id)
            '''if i == self.mid_ins:
                print('Right Target')
                reward += 1
            else:
                print('Wrong Target')
                reward += -0.2
            break'''
            # hit the target
            #for i in range(self.num_targets):
            vec = np.array([xpos-self.target_coord[self.ins][0], ypos-self.target_coord[self.ins][1]])
            dist = np.linalg.norm(vec)
            if dist < self.target_size and self.mid_done:
                done = True
                print('Success')
                    #print('self.id',self.id)
            '''if i == self.ins:
                print('Right Target')
                reward += 1
            else:
                print('Wrong Target')
                reward += -0.2
            break'''
        elif self.id == 4: # goal evaluating [envType, nan, nan, targetX, targetY]
            vec = np.array([xpos-self.trainingTarget[2], ypos-self.trainingTarget[3]])
            dist = np.linalg.norm(vec)
            if dist < self.target_size:
                done = True
                self.num_success += 1
                print('Success')
                print('Nsuccess', self.num_success)
                #print('self.id',self.id)
        elif self.id == 5: # trajtory evaluating [envType, mid_targetX, mid_targetY, targetX, targetY]
            vec = np.array([xpos-self.trainingTarget[0], ypos-self.trainingTarget[1]])
            dist = np.linalg.norm(vec)
            if dist < self.target_size:
                self.mid_done = True
                #print('midSuccess')
            vec = np.array([xpos-self.trainingTarget[2], ypos-self.trainingTarget[3]])
            dist = np.linalg.norm(vec)
            if dist < self.target_size and self.mid_done:
                done = True
                self.num_success += 1
                print('Success')
                print('Nsuccess', self.num_success)
                #print('self.id',self.id)
        else:
            print('Evelyn ErrorRRRRRRRRRRRRRRRRRRRRRRRRRR')
        
        # hit the wall
        if not done:
            if xpos == 1 or xpos == -1 or ypos == 1 or ypos == -1:
            #if np.linalg.norm(np.array([xpos, ypos])) > self.arena_size:
                done = True
                reward += -1
                print('Hit the Wall')
        
        # times up
        self.timesteps += 1
        if not done and self.timesteps >= self.max_timesteps:
            done = True
            reward += -0.5
            print('Times Up')

        return self.get_obs(), reward, done, {}

    def reset(self, task=None, max_timesteps = 50):
        self.ins = -1
        self.mid_ins = -1
        # Task
        if task is None:
            #task = np.random.random_sample(np.shape(self.low_task))
            #task = task*(self.high_task-self.low_task)+self.low_task
            self.task = np.random.randint(self.num_targets)
            self.id = -1
        else:
            self.task = task
            self.id = int(self.task[0])
            '''task = np.array[envType, informations]'''
            if self.id == 0: # goal training [envType, nan, nan, nan, nan]
                self.trainingTarget[2] = np.random.sample() * 2 - 1
                self.trainingTarget[3] = np.random.sample() * 2 - 1
                self.mid_done = True
            elif self.id == 1: # trajtory training [envType, nan, nan, nan, nan]
                self.trainingTarget[2] = np.random.sample() * 2 - 1
                self.trainingTarget[3] = np.random.sample() * 2 - 1
                self.mid_done = True
            elif self.id == 2: # goal testing [envType, nan, instruction, nan, nan]
                self.task[2] = task[2]
                self.ins = int(self.task[2])
                self.mid_done = True
            elif self.id == 3: # trajectory testing [envType, mid_instruction, instruction, nan, nan]
                self.task[1:3] = task[1:3]
                self.mid_ins = int(self.task[1])
                self.ins = int(self.task[2])
                self.mid_done = False
            elif self.id == 4: # goal evaluating [envType, nan, nan, targetX, targetY]
                self.trainingTarget[2:4] = task[3:5]
                self.mid_done = True
            elif self.id == 5: # trajtory evaluating [envType, mid_targetX, mid_targetY, targetX, targetY]
                self.trainingTarget = task[1:5]
                self.mid_done = False
            else:
                print('Evelyn ErrorRRRRRRRRRRRRRRRRRRRRRRRRRR')
                self.task = np.array(task)

        self.instr = np.zeros(self.num_targets+self.num_mid_targets)

        if self.id == 3:
            print('ins:', self.mid_ins, ' ', self.ins)
            self.instr = np.zeros(self.num_targets+self.num_mid_targets)
            self.instr[self.ins] = 1
            self.instr[self.mid_ins] = 1
            #assert self.instr_space.contains(self.instr), "%r (%s) invalid task" % (self.instr, type(self.instr))

            # Set target
            self.target_color = []
            for i in range(7):
                self.target_color.append([0, 1, 0])
            self.target_color[self.ins] = [1, 0, 0]
            self.target_color[self.mid_ins] = [1, 0, 0]
        '''
        elif self.id == 2:
            # Instruction
            self.instr = np.zeros(self.num_targets + self.num_mid_targets)
            self.instr[self.ins] = 1
            assert self.instr_space.contains(self.instr), "%r (%s) invalid task" % (self.instr, type(self.instr))

            # Set target
            self.target_color = []
            for i in range(7):
                self.target_color.append([0, 1, 0])
            self.target_color[self.ins] = [1, 0, 0]
        else:
            # Instruction
            self.instr = np.zeros(self.num_targets + self.num_mid_targets)
            #self.instr[self.task] = 1
            assert self.instr_space.contains(self.instr), "%r (%s) invalid task" % (self.instr, type(self.instr))

            # Set target
            self.target_color = []
            for i in range(7):
                self.target_color.append([0, 1, 0])
            #self.target_color[self.task] = [1, 0, 0]
        '''
        # Timestep
        self.max_timesteps = max_timesteps
        self.timesteps = 0

        # State
        #theta = 2*np.pi*np.random.random_sample()
        #self.state = np.array([0, 0, np.cos(theta), np.sin(theta)])
        self.state = np.array([0, -.5, 0, 1])

        # rotate will change in simulation
        #self.rotate_scale = 0.3

        return self.get_obs()
        
    def get_obs(self):
        # Observation
        # [agent coord] + instruction
        obs = np.concatenate((self.state[0:2], self.instr[0:5]))
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
            for i in range(self.num_targets + self.num_mid_targets):
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
        for i in range(self.num_targets + self.num_mid_targets):
            r, g, b = self.target_color[i]
            self.targets[i].set_color(r, g, b)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
