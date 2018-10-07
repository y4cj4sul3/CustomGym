import numpy as np
from gym import utils
from gym import spaces
#from gym.envs.mujoco import mujoco_env
from custom_gym.mujoco import mujoco_env

class ReacherGEPEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.timestep = 0
        self.maxtimestep = 20

        self.trainingTarget = np.zeros(4)
        self.zoneCheck = 0.019
        self.N_success = 0
        self.id = -1

        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)

        print('Frame Skip: {}'.format(self.frame_skip))

    def step(self, a):
        # Simulate
        #a *= 0.9
        self.do_simulation(a, self.frame_skip)

        # State
        vec = self.get_body_com("fingertip")[:2]-self.get_body_com("target")[:2]
        dist = np.linalg.norm(vec)
        reward_dist = - dist
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        
        # State after Sim
        done = False
        if self.id == 0 or self.id == 1: # goal traj trainingTarget [nan, nan, random, random]
            if dist <= self.zoneCheck:
                done = True
                reward += 1
                print("Success")
        elif self.id == 2: # goal testing [envType, nan, instruction, nan, nan]
            if dist <= self.zoneCheck:
                done = True
                reward += 1
                print("Success")
        elif self.id == 3: # trajectory testing [envType, mid_instruction, instruction, nan, nan]
            vec = self.get_body_com("fingertip")[:2]-self.targetList[self.mid_ins]
            dist = np.linalg.norm(vec)
            if dist <= self.zoneCheck:
                self.mid_done = True

            if self.mid_done:
                vec = self.get_body_com("fingertip")[:2]-self.targetList[self.ins]
                dist = np.linalg.norm(vec)
                if dist <= self.zoneCheck:
                    done = True
                    reward += 1
                    print("Success")
        elif self.id == 4: # goal evaluating [envType, nan, nan, targetX, targetY]
            if dist <= self.zoneCheck:
                done = True
                reward += 1
                self.N_success += 1
                print("Success")
                print('N_success', self.N_success)
        elif self.id == 5: # trajtory evaluating [envType, mid_targetX, mid_targetY, targetX, targetY]
            if dist <= self.zoneCheck:
                self.mid_done = True
            
            if self.mid_done:
                vec = self.get_body_com("fingertip")[:2]-self.trainingTarget[2:4]
                dist = np.linalg.norm(vec)
                if dist <= self.zoneCheck:
                    done = True
                    reward += 1
                    self.N_success += 1
                    print("Success")
                    print('N_success', self.N_success)
        else:
            print('Evelyn ErrorRRRRRRRRRRRRRRRRRRRRRRRRRR')

        self.timestep += 1
        if not done and self.timestep >= self.maxtimestep:
            done = True
            reward += -0.5
            print('Times Up')
        
        obs = self._get_obs()

        return obs, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl, t=self.timestep)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self, task=None, maxtimestep=20, isEval=False):
        # Position
        # [arm_angle_1, arm_angle_2, target_xpos, target_ypos]
        qpos = self.init_qpos
        self.ins = -1
        self.mid_ins = -1
        if task is None:
            self.id = -1
            '''while True:
                print('Evelyn ErrorRRRRRRRRRRRRRRRRRRRRRRRRRR')
                self.task = np.array(task)
                break'''
        else:
            self.task = task
            self.id = int(self.task[0])
            if self.id == 0: # goal training [envType, nan, nan, nan, nan]
                theta = np.random.sample() * 360
                rad = np.random.sample() * 0.21
                self.trainingTarget[2:4] = np.array([np.cos(np.deg2rad(theta)) * rad, np.sin(np.deg2rad(theta)) * rad])
                self.mid_done = True
            elif self.id == 1: # trajtory training [envType, nan, nan, nan, nan], same as goal training
                theta = np.random.sample() * 360
                rad = np.random.sample() * 0.21
                self.trainingTarget[2:4] = np.array([np.cos(np.deg2rad(theta)) * rad, np.sin(np.deg2rad(theta)) * rad])
                self.mid_done = True
            elif self.id == 2: # goal testing [envType, nan, instruction, nan, nan]
                self.targetList = np.array([[0, .15], [-.1, .1], [-.2, 0], [-.1, -.1], [0, -.15]])
                self.ins = int(self.task[2])
                self.mid_done = True
            elif self.id == 3: # trajectory testing [envType, mid_instruction, instruction, nan, nan]
                mid_goal = np.array([[np.cos(np.deg2rad(45)), np.sin(np.deg2rad(45))], [np.cos(np.deg2rad(315)), np.sin(np.deg2rad(315))]])
                f_goal = []
                for x in range(5):
                    f_goal.append([np.cos(np.deg2rad(180)) * x * 0.2 - 0.1, np.sin(np.deg2rad(180)) * x * 0.2])
                self.targetList = np.concatenate(([mid_goal, f_goal]), axis=0)
                self.targetList *= 0.21
                self.mid_ins = int(self.task[1])
                self.ins = int(self.task[2])
                #print('env receive ins:', self.mid_ins , ' ', self.ins)
                #print('env ins target pos:', self.target_coord[self.mid_ins], ' ', self.target_coord[self.ins])
                self.mid_done = False
            elif self.id == 4: # goal evaluating [envType, nan, nan, targetX, targetY]
                self.trainingTarget[2:4] = np.array([task[3]*0.21, task[4]*0.21])
                self.mid_done = True
            elif self.id == 5: # trajtory evaluating [envType, mid_targetX, mid_targetY, targetX, targetY]
                self.trainingTarget = np.array(task[1:5]) * 0.21
                self.mid_done = False
            else:
                print('Evelyn ErrorRRRRRRRRRRRRRRRRRRRRRRRRRR')
                self.task = np.array(task)

        # copy goal -> task
        if self.id == 2 or self.id == 3:
            qpos[-2:] = self.targetList[self.ins]
        else:
            qpos[-2:] = self.trainingTarget[2:4]

        # Velocity
        # [arm_aglvel_1, arm_aglvec_2, target_xvel, target_yvel]
        qvel = self.init_qvel
        qvel[-2:] = 0

        # State
        self.set_state(qpos, qvel)

        self.timestep = 0
        self.maxtimestep = maxtimestep
        self.isEval = isEval
        
        return self._get_obs()

    def _get_obs(self):
        #theta = self.sim.data.qpos.flat[:2]
        xpos = self.get_body_com("fingertip")[0]
        xpos /= 0.21
        ypos = self.get_body_com("fingertip")[1]
        ypos /= 0.21
        # Observation
        # [cos(arm_angle_1), cos(arm_angle_2),
        #  sin(arm_angle_1), sin(arm_angle_2),
        #  target_xpos, target_ypos
        #  arm_aglforce_1, arm_aglforce_2,
        #  dist_x, dist_y, dist_z]
        return np.concatenate([
            #np.cos(theta),
            #np.sin(theta),
            [xpos, ypos],
            #self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            #self.get_body_com("fingertip") - self.get_body_com("target")
        ])
