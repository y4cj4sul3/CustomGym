import numpy as np
from gym import utils
from gym import spaces
#from gym.envs.mujoco import mujoco_env
from custom_gym.mujoco import mujoco_env

class ReacherGEPTrajEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.task_traj = np.array([1, 1, 1, 1])
        self.mid_done = False
        self.timestep = 0
        self.maxtimestep = 15
        self.zoneCheck = 0.019
        self.N_success = 0
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)

        print('Frame Skip: {}'.format(self.frame_skip))

    def step(self, a):
        # Simulate
        self.do_simulation(a, self.frame_skip)

        # State
        vec = self.get_body_com("fingertip")[:2]-self.get_body_com("target")[:2]
        dist = np.linalg.norm(vec)
        reward_dist = - dist
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        
        # Check if close enough to the target
        done = False
        vec = self.get_body_com("fingertip")[0:2]-self.task_traj[0:2]
        m_dist = np.linalg.norm(vec)
        if m_dist < self.zoneCheck:
            self.mid_done = True
            #print("Mid Success")
        if dist < self.zoneCheck and self.mid_done:
            done = True
            reward += 1
            print("Success")
            if self.isEval:
                self.N_success += 1
                print('N_success', self.N_success)
        
        self.timestep += 1
        if not done and self.timestep >= self.maxtimestep:
            done = True
            reward += -0.5
            print('Times Up')

        ob = self._get_obs()

        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl, t=self.timestep)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self, task=None, maxtimestep=15, isEval=False):
        # Position
        # [arm_angle_1, arm_angle_2, target_xpos, target_ypos]
        # replace goal -> task
        qpos = self.init_qpos
        while True:
            if task is None:
                theta = np.random.sample() * 360
                rad = np.random.sample() * 0.21
                m_theta = np.random.sample() * 360
                m_rad = np.random.sample() * 0.21
                self.task_traj = np.array([np.cos(np.deg2rad(m_theta)) * m_rad, np.sin(np.deg2rad(m_theta)) * m_rad, np.cos(np.deg2rad(theta)) * rad, np.sin(np.deg2rad(theta)) * rad])
            else:
                self.task_traj = np.array(task) * 0.21
                #print(str(self.task_traj))
            if np.linalg.norm(self.task_traj[0:2]) <= 0.21 and np.linalg.norm(self.task_traj[2:4]) <= 0.21:
                break
        self.goal = self.task_traj[-2:]
        qpos[-2:] = self.goal

        # Velocity
        # [arm_aglvel_1, arm_aglvec_2, target_xvel, target_yvel]
        qvel = self.init_qvel
        qvel[-2:] = 0

        # State
        self.set_state(qpos, qvel)

        self.timestep = 0
        self.maxtimestep = maxtimestep
        self.mid_done = False
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
