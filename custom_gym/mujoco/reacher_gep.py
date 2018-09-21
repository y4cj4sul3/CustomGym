import numpy as np
from gym import utils
from gym import spaces
#from gym.envs.mujoco import mujoco_env
from custom_gym.mujoco import mujoco_env

class ReacherGEPEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)

        print('Frame Skip: {}'.format(self.frame_skip))

    def step(self, a):
        # State before Sim
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl

        #print('action: ' + str(a))
        # Simulate
        self.do_simulation(a, self.frame_skip)
        
        # State after Sim
        obs = self._get_obs()
        done = False

        xpos, ypos = self.get_body_com("fingertip")[0], self.get_body_com("fingertip")[1]
        if abs(xpos - self.task_goal[0]) <= 0.01 and abs(ypos - self.task_goal[1]) <=0.01:
            done = True
            print("Success")
        #print('xpos ypos:' + str(xpos) + ' ' + str(ypos))
        #print('goal:' + str(self.task_goal[0]) + ' ' + str(self.task_goal[1]))

        self.timestep += 1

        return obs, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl, t=self.timestep)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self, task=None):
        # Position
        # [arm_angle_1, arm_angle_2, target_xpos, target_ypos]
        qpos = self.init_qpos
        while True:
            if task is None:
                theta = np.random.sample() * 360
                rad = np.random.sample() * 0.21
                self.goal = np.array([np.cos(np.deg2rad(theta)) * rad, np.sin(np.deg2rad(theta)) * rad])
                #self.goal = np.array([1, 1])
            else:
                self.goal = np.array(task) * 0.21
                print(str(self.goal))
            if np.linalg.norm(self.goal) <= 0.21:
                break
        # copy goal -> task
        self.task_goal = self.goal
        #print(str(self.goal))
        qpos[-2:] = self.goal

        # Velocity
        # [arm_aglvel_1, arm_aglvec_2, target_xvel, target_yvel]
        qvel = self.init_qvel
        qvel[-2:] = 0

        # State
        self.set_state(qpos, qvel)

        self.timestep = 0
        self.maxtimestep = 50
        
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
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
