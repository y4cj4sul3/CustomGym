import numpy as np
from gym import utils
from custom_gym.mujoco import mujoco_env
from custom_gym.utils import Recoder

class ReacherFiveTargetEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        # random init target
        self.set_target()
        # timestep
        self.max_timesteps = 50
        self.timesteps = 0
        '''
        self.hit = False
        # recorder
        self.recorder = Recoder('Dataset/ReacherFiveTarget-v0/ppo2_2e6/')
        self.recorder.traj['reward'] = 0
        '''
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher_five_target.xml', 2)

    def step(self, a):
        # before sim
        vec = self.get_body_com("fingertip")-self.get_body_com("true_target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        '''
        # record
        self.recorder.step(self._get_obs())
        self.recorder.traj['reward'] += reward
        '''
        # sim
        self.do_simulation(a, self.frame_skip)

        # after sim
        ob = self._get_obs()

        # check done
        #done = self.hit
        done = False
        done_status = ''
        # collision detection
        if self.data.ncon > 0:
            done = True
            #self.hit = True
            #print('=========Contact========')
            for coni in range(self.data.ncon):
                #print('--------{}--------'.format(coni))
                con = self.data.contact[coni]
                #print('  geom1  = %d' % (con.geom1))
                #print('  geom2  = %d' % (con.geom2))

                if con.geom2 == 9:
                    print('Right Target')
                    done_status = 'Right Target'
                    reward += 1
                '''
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
        
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl, done_status=done_status, coord=self.get_body_com('fingertip'))

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def set_target(self, target_id=None):
        if target_id == None:
            # random init target
            self.target_id = np.random.randint(5)
        else:
            self.target_id = target_id
        print('Current Target: {}'.format(self.target_id))

        # instruction (one-hot)
        self.target_one_hot = np.zeros(5)
        self.target_one_hot[self.target_id] = 1

    def reset_model(self, target_id=None):
        # random init target
        self.set_target(target_id)
        
        # timestep
        self.timesteps = 0

        '''
        self.hit = False
        # save traj
        self.recorder.save()
        # reset traj
        self.recorder.traj['reward'] = 0
        # save file
        self.recorder.step(self._get_obs())
        '''
        # qpos (12 dim)
        # [(finger_2_angle), (true_target_xy), (t1_xy), (t2_xy), (t3_xy), (t4_xy)]
        #qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        qpos = np.array([0, 0, 0, .15, -.1, .1, -.2, 0, -.1, -.1, 0, -.15])
        # set target position
        qpos[2], qpos[self.target_id*2+2] = qpos[self.target_id*2+2], qpos[2]
        qpos[3], qpos[self.target_id*2+3] = qpos[self.target_id*2+3], qpos[3]

        # qvel (12 dim)
        # [(finger), (true_target), (t1), (t2), (t3), (t4)]
        #qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel = np.zeros(12)

        self.set_state(qpos, qvel)

        return self._get_obs()

    def _get_obs(self):
        #print(self.sim.data.qpos)
        theta = self.sim.data.qpos.flat[:2]
        # Observation (11 dim)
        # [cos(angle_1), cos(angle_2),
        #  sin(angle_2), sin(angle_2),
        #  angle_vec_1, angle_vec_2,
        #  one_hot_instruction]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qvel.flat[:2],
            self.target_one_hot
        ])
