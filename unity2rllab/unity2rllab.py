
import gym

from rllab.envs.base import Env, Step
from rllab.envs.env_spec import EnvSpec

from unity2rllab.unity_env_pos import UnityEnvPos
from unity2rllab.unity_env_no_pos import UnityEnvNoPos

# Unity Env with position - v0
# ----------------------------------------
class UDColor_Position(Env):
    def __init__(self, idx=0):
        self.env = UnityEnvPos("/home/recharrs/UnityExec/UnityDiversity_color/kobuki.x86_64", idx=128+idx)
        self.spec = gym.spec(0)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

class UDColorObj_Position(UnityEnvPos):
    def __init__(self, idx=1):
        super().__init__("/home/recharrs/UnityExec/UnityDiversity_colorObj/kobuki.x86_64", idx=128+idx)

class UDColor2_Position(UnityEnvPos):
    def __init__(self, idx=2):
        super().__init__("/home/recharrs/UnityExec/UnityDiversity_color-2/kobuki.x86_64", idx=128+idx)

# Unity Env without position - v1
# ----------------------------------------
class UDColor_NoPosition(UnityEnvNoPos):
    def __init__(self, idx=3):
        super().__init__("/home/recharrs/UnityExec/UnityDiversity_color/kobuki.x86_64", idx=128+idx)

class UDColorObj_NoPosition(UnityEnvNoPos):
    def __init__(self, idx=4):
        super().__init__("/home/recharrs/UnityExec/UnityDiversity_colorObj/kobuki.x86_64", idx=128+idx)

class UDColor2_NoPosition(UnityEnvNoPos):
    def __init__(self, idx=5):
        super().__init__("/home/recharrs/UnityExec/UnityDiversity_color-2/kobuki.x86_64", idx=128+idx)
