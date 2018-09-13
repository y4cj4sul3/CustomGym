import gym

from unity2rllab.unity_env_v0 import UnityEnvV0
from unity2rllab.unity_env_v1 import UnityEnvV1

# with position
# ----------------------------------------
class UnityColor7_v0(UnityEnvV0):
    def __init__(self, idx=0):
        super().__init__("/home/recharrs/UnityExec/UnityDiversity_C2+5/kobuki.x86_64", idx=128+idx)

# Unity Env without position - v1
# ----------------------------------------
class UnityColor7_v1(UnityEnvV1):
    def __init__(self, idx=0):
        super().__init__("/home/recharrs/UnityExec/UnityDiversity_C2+5/kobuki.x86_64", idx=128+idx)

# make
# ----------------------------------------
def make_env(env_name, idx):
    print(env_name)
    if env_name == 'UnityColor7_v0':
        return UnityColor7_v0(idx=idx)
    if env_name == 'UnityColor7_v1':
        return UnityColor7_v1(idx=idx)
