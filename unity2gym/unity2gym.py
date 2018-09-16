from unity2gym.unity_env_v0 import UnityEnvV0
from unity2gym.unity_env_v1 import UnityEnvV1
from unity2gym.unity_env_v3 import UnityEnvV3

# ----------------------------------------
class UnitySkillRL_v0(UnityEnvV0):
    def __init__(self, idx=0):
        super().__init__("/home/recharrs/UnityExec/UnityRL_skill/kobuki.x86_64", idx=idx)

class UnitySkillRLUnit_v0(UnityEnvV0):
    def __init__(self, idx=1):
        super().__init__("/home/recharrs/UnityExec/UnityRL_skill_unit/kobuki.x86_64", idx=idx)

# ----------------------------------------
class UnitySkillRL_v1(UnityEnvV1):
    def __init__(self, idx=10):
        super().__init__("/home/recharrs/UnityExec/UnityRL_skill/kobuki.x86_64", idx=idx)

# ----------------------------------------
class UnitySkillRL_v2(UnityEnvV0):
    ''' coord, rgb skill. with reward shaping '''
    def __init__(self, idx=20):
        super().__init__("/home/recharrs/UnityExec/UnityRL_skill_v2/kobuki.x86_64", idx=idx)

# ----------------------------------------
class UnitySkillRL_v3(UnityEnvV3):
    ''' coord, rgb skill. with reward shaping '''
    def __init__(self, idx=30):
        super().__init__("/home/recharrs/UnityExec/UnityRL_skill_v2/kobuki.x86_64", idx=idx)

def make_env(env_name, idx=0):
    if env_name == "UnitySkillRL_v0":
        return UnitySkillRL_v0(idx=idx)
    if env_name == "UnitySkillRLUnit_v0":
        return UnitySkillRLUnit_v0(idx=idx)
    if env_name == "UnitySkillRL_v1":
        return UnitySkillRL_v1(idx=idx)
    if env_name == "UnitySkillRL_v2":
        return UnitySkillRL_v2(idx=idx)
    if env_name == "UnitySkillRL_v3":
        return UnitySkillRL_v3(idx=idx)
