import numpy as np
from gym import utils
from custom_gym.mujoco import mujoco_env
from custom_gym.utils import Recoder

class ReacherOverCookedEnv_v1(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        # random init target
        self.set_target()
        # timestep
        self.max_timesteps = 50
        self.timesteps = 0
        
        # recorder
        self.is_record = True
        if self.is_record:
            self.recorder = Recoder('Dataset/ReacherOverCooked-v1/test/')
            self.recorder.traj['reward'] = 0
            self.recorder.traj['coord'] = []
        
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher_over_cooked.xml', 2)

    def step(self, a):
        
        # sim
        self.do_simulation(a, self.frame_skip)

        # target
        num_task_left = len(self.target_id)

        # after sim
        vec1 = self.get_body_com("fingertip")-self.get_body_com("checkpoint")
        vec2 = self.get_body_com("fingertip")-self.get_body_com("true_target")
        dist1 = np.linalg.norm(vec1)
        dist2 = np.linalg.norm(vec2)
        if num_task_left == 2:
            dist = dist1
        else:
            dist = dist2
        reward_dist = - dist
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl

        # task
        reward += -0.4*(num_task_left-1)

        # check done
        done = False
        done_status = ''
        # collision detection
        if dist < 0.019:
            if num_task_left == 2:
                self.target_id = self.target_id[1:]
                reward += 0.5
                print('Right Target')
                done_status = 'Right Target'
            else:
                done = True
                reward += 1
                print('Finish Task')
                done_status = 'Finish Task'
        '''
        # TODO: wrong target
        # currently only true target can be detect
        else:
            print('Wrong Target')
            reward += -0
        '''
        # times up
        self.timesteps += 1
        if not done and self.timesteps >= self.max_timesteps:
            done = True
            reward += -0.5
            print('Times Up')
            done_status = 'Times Up'
        
        # record
        min_dist_cp = 0
        min_dist_ft = 0
        if self.is_record:
            self.recorder.step(self._get_obs(), a)
            #if hasattr(self.recorder.traj, 'reward'):
            self.recorder.traj['reward'] += reward
            self.recorder.traj['coord'].append(self.get_body_com("fingertip").tolist())
            #if done_status == 'Finish Task':
            if done:
                # find dist closest to checkpoint
                ctcp = np.argmin(np.linalg.norm(np.array(self.recorder.traj['coord'])-self.get_body_com('checkpoint'), axis=1))
                ctft = ctcp+np.argmin(np.linalg.norm(np.array(self.recorder.traj['coord'])[ctcp:]-self.get_body_com('true_target'), axis=1))
                min_dist_cp = np.linalg.norm(np.array(self.recorder.traj['coord'][ctcp]-self.get_body_com('checkpoint')))
                min_dist_ft = np.linalg.norm(np.array(self.recorder.traj['coord'][ctft]-self.get_body_com('true_target')))
                #print(min_dist_cp, min_dist_ft)
                #self.recorder.save()
        
        # get obs
        ob = self._get_obs()
        
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl, done_status=done_status, coord=self.get_body_com('fingertip'), dist=dist, min_dist_cp=min_dist_cp, min_dist_ft=min_dist_ft)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def set_target(self, target_id=None):
        if target_id == None:
            # random init target
            self.target_id = [np.random.randint(2), 2+np.random.randint(5)]
        else:
            self.target_id = target_id
        #print('Current Target: {}'.format(self.target_id))

        # instruction (one-hot)
        self.instr = np.zeros(7)
        self.instr[self.target_id] = 1
        print('Instruction: {}'.format(self.instr))

    def reset_model(self, target_id=None):
        # random init target
        self.set_target(target_id)
        
        # timestep
        self.timesteps = 0

        # qpos (16 dim)
        # [(finger_2_angle), (true_target_xy), (t1_xy), (t2_xy), (t3_xy), (t4_xy), (checkpoint), (t5_xy)]
        #qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        cp_coord = np.sqrt(0.5)
        qpos = .21*np.array([0, 0, -.1, 0, -.3, 0, -.5, 0, -.7, 0, -.9, 0, cp_coord, cp_coord, cp_coord, -cp_coord])
        qpos[12], qpos[self.target_id[0]*2+12] = qpos[self.target_id[0]*2+12], qpos[12]
        qpos[13], qpos[self.target_id[0]*2+13] = qpos[self.target_id[0]*2+13], qpos[13]
        # set target position
        qpos[2], qpos[(self.target_id[1]-2)*2+2] = qpos[(self.target_id[1]-2)*2+2], qpos[2]
        qpos[3], qpos[(self.target_id[1]-2)*2+3] = qpos[(self.target_id[1]-2)*2+3], qpos[3]

        # qvel (16 dim)
        # [(finger), (true_target), (t1), (t2), (t3), (t4), (checkpoint), (t5_xy)]
        #qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel = np.zeros(16)

        self.set_state(qpos, qvel)
        
        # recorder
        if self.is_record:
            # save traj
            #self.recorder.save()
            # reset traj
            self.recorder.reset_traj()
            self.recorder.traj['reward'] = 0
            self.recorder.traj['coord'] = []
            # save first step
            self.recorder.step(self._get_obs())
        
        return self._get_obs()

    def _get_obs(self):
        #print(self.sim.data.qpos)
        theta = self.sim.data.qpos.flat[:2]
        # Observation (13 dim)
        # [cos(angle_1), cos(angle_2),
        #  sin(angle_2), sin(angle_2),
        #  angle_vec_1, angle_vec_2,
        #  one_hot_instruction]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qvel.flat[:2],
            self.instr
        ])
