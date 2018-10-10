import numpy as np
from gym import utils
from custom_gym.mujoco import mujoco_env
from custom_gym.utils import Recoder

def one_hot(size, target):
    one_hot_vec = np.zeros(size)
    one_hot_vec[target] = 1
    return one_hot_vec

class ReacherOverCookedEnv_v3(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, is_record=True, save_path="Dataset/ReacherTraj/demo"):
        # environment setup, set properties
        self.save_path = save_path
        self.max_timesteps = 20 # max-timesteps
        self.timesteps = 0

        # set flags
        self.is_record = is_record
        self.show_info = True

        # set recoder, etc.
        self.set_target()  # random init target
        self._call_recorder("init")
        
        # set mujoco
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher_over_cooked.xml', 2)

    def step(self, a):
        # simulation & after simulation
        self.do_simulation(a, self.frame_skip)
        dist1 = np.linalg.norm(self.get_body_com("fingertip") - self.get_body_com("checkpoint"))
        dist2 = np.linalg.norm(self.get_body_com("fingertip") - self.get_body_com("true_target"))

        # target & reward ( num_task_left = len(self.target_id) ) 
        dist = dist1 if len(self.target_id) == 2 else dist2 # first or second task
        reward_dist, reward_ctrl = (-dist), (-np.square(a).sum())
        reward = reward_dist + reward_ctrl + -0.4 * (len(self.target_id) - 1) # reward_dist + reward_ctrl + task
    
        # check done
        self.timesteps += 1
        reward, done, done_status = self._collision_detection(dist, reward)
        
        # recording
        self.a, self.reward, self.done = a, reward, done
        self._call_recorder('init')
        min_dist_cp, min_dist_ft = self._call_recorder("step")

        # info
        info = dict(
            reward_dist=reward_dist, reward_ctrl=reward_ctrl, 
            done_status=done_status, 
            coord=self.get_body_com('fingertip'), 
            dist=dist, min_dist_cp=min_dist_cp, min_dist_ft=min_dist_ft
        )

        # return values for the step
        return self._get_obs(), reward, done, info

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def set_target(self, target_id=None):
        self.target_id = [np.random.randint(2), 2+np.random.randint(5)] if target_id == None else target_id
        self.instr = one_hot(7, self.target_id)
        print('Instruction: {}'.format(self.instr))

    def reset_model(self, target_id=None):
        # setup 
        self.set_target(target_id) # random init target
        self.timesteps = 0 # timesteps

        # qpos (16 dim): [(finger_2_angle), (true_target_xy), (t1_xy), (t2_xy), (t3_xy), (t4_xy), (checkpoint), (t5_xy)]
        cp_coord = np.sqrt(0.5)
        qpos = 0.21 * np.array([0, 0, -.1, 0, -.3, 0, -.5, 0, -.7, 0, -.9, 0, cp_coord, cp_coord, cp_coord, -cp_coord])
        # set target position
        qpos[12], qpos[self.target_id[0]*2+12] = qpos[self.target_id[0]*2+12], qpos[12]
        qpos[13], qpos[self.target_id[0]*2+13] = qpos[self.target_id[0]*2+13], qpos[13]
        # set target position
        qpos[2], qpos[(self.target_id[1]-2)*2+2] = qpos[(self.target_id[1]-2)*2+2], qpos[2]
        qpos[3], qpos[(self.target_id[1]-2)*2+3] = qpos[(self.target_id[1]-2)*2+3], qpos[3]

        # qvel (16 dim): [(finger), (true_target), (t1), (t2), (t3), (t4), (checkpoint), (t5_xy)]
        qvel = np.zeros(16)
        self.set_state(qpos, qvel)
        
        # recorder
        self._call_recorder("reset_model")

        # return ob
        return self._get_obs()

    def _get_obs(self):
        pos = self.get_body_com("fingertip") / 0.21
        q_vel = self.sim.data.qvel.flat[:2]
        return np.concatenate([pos, q_vel, self.instr])

    def _collision_detection(self, dist, reward):        
        if dist < 0.019 and len(self.target_id) == 2:
            self.target_id = self.target_id[1:]
            reward += 0.5
            return reward, False, 'Right Target'
        if dist < 0.019:
            reward += 1
            return reward, True, 'Finish Task'
        if self.timesteps >= self.max_timesteps:
            reward += -0.5
            return reward, True, 'Times Up'
        return reward, False, ''

    def _call_recorder(self, command):
        if command == "init":
            self.recorder = Recoder(self.save_path)
            self.recorder.traj['reward'] = 0
            self.recorder.traj['coord'] = []
        if command == "reset_model":
            # save traj
            if self.is_record:
                self.recorder.save() 
            # reset traj
            self.recorder.reset_traj()
            self.recorder.traj['reward'] = 0
            self.recorder.traj['coord'] = []
            # save first step
            self.recorder.step(self._get_obs())
        if command == "step":
            self.recorder.step(self._get_obs(), self.a)
            self.recorder.traj['reward'] += self.reward
            self.recorder.traj['coord'].append(self.get_body_com("fingertip").tolist())
            return 0, 0
        if command == "step" and self.done:  
            # find dist closest to checkpoint
            ctcp = np.argmin(np.linalg.norm(np.array(self.recorder.traj['coord'])-self.get_body_com('checkpoint'), axis=1))
            ctft = ctcp+np.argmin(np.linalg.norm(np.array(self.recorder.traj['coord'])[ctcp:]-self.get_body_com('true_target'), axis=1))
            min_dist_cp = np.linalg.norm(np.array(self.recorder.traj['coord'][ctcp]-self.get_body_com('checkpoint')))
            min_dist_ft = np.linalg.norm(np.array(self.recorder.traj['coord'][ctft]-self.get_body_com('true_target')))
            # save traj
            if self.is_record:
                self.recorder.save() 
            return min_dist_cp, min_dist_ft
