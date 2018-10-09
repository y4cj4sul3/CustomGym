import numpy as np
from gym import utils
from gym import spaces
#from gym.envs.mujoco import mujoco_env
from custom_gym.mujoco import mujoco_env

class ReacherGEPTrajTestEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.timestep = 0
        self.maxtimestep = 20

        mid_goal = np.array([[np.cos(np.deg2rad(45)), np.sin(np.deg2rad(45))], [np.cos(np.deg2rad(315)), np.sin(np.deg2rad(315))]])
        f_goal = []
        for x in range(5):
            f_goal.append([np.cos(np.deg2rad(180)) * x * 0.2 - 0.1, np.sin(np.deg2rad(180)) * x * 0.2])
        self.targetList = np.concatenate(([mid_goal, f_goal]), axis=0)
        self.targetList *= 0.21
        #self.targetList = np.array([[0, .15], [-.1, .1], [-.2, 0], [-.1, -.1], [0, -.15], [.1, .1], [.1, -.1]])
        self.task = np.array([-1, -1])
        self.mid_done = False
        self.zoneCheck = 0.019

        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)

        print('Frame Skip: {}'.format(self.frame_skip))

    def step(self, a):
        # Simulate
        self.do_simulation(a, self.frame_skip)

        # State before Sim
        vec = self.get_body_com("fingertip")[:2]-self.get_body_com("target")[:2]
        dist = np.linalg.norm(vec)
        reward_dist = - dist
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        
        done = False
        vec = self.get_body_com("fingertip")[0:2]-self.targetList[self.task[0]]
        m_dist = np.linalg.norm(vec)
        if m_dist < self.zoneCheck:
            self.mid_done = True
            #print("Mid Success")

        vec = self.get_body_com("fingertip")[0:2]-self.targetList[self.task[1]]
        m_dist = np.linalg.norm(vec)
        if dist < self.zoneCheck and self.mid_done:
            done = True
            reward += 1
            print("Success")
        
        self.timestep += 1
        if not done and self.timestep >= self.maxtimestep:
            done = True
            reward += -0.5
            print('Times Up')

        ob = self._get_obs()

        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl, t=self.timestep)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self, task=None):
        # Position
        # [arm_angle_1, arm_angle_2, target_xpos, target_ypos]
        # replace goal -> task
        qpos = self.init_qpos
        if task is not None:
            self.task = task
            qpos[2], qpos[3] = self.targetList[self.task[1]][0], self.targetList[self.task[1]][1]
            print('ins:', task)
            print('target pos:', self.targetList[task])
        #else:
            #print('Evelyn ErrorrrrRRR')
        

        # Velocity
        # [arm_aglvel_1, arm_aglvec_2, target_xvel, target_yvel]
        qvel = self.init_qvel
        qvel[-2:] = 0

        # State
        self.set_state(qpos, qvel)

        self.timestep = 0
        self.maxtimestep = 20
        self.mid_done = False
        
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
